#include "targets/qwen3_6/impl/frontend/encoded_history_cache.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

bool asan_build() noexcept {
#if defined(__SANITIZE_ADDRESS__)
    return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
}

bool bytes_equal_prefix(std::string_view stored, std::string_view full) noexcept {
    return stored.size() <= full.size() &&
           std::memcmp(stored.data(), full.data(), stored.size()) == 0;
}

std::uint32_t checked_frontier(std::size_t value, std::string_view what) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(what) + " exceeds uint32");
    }
    return static_cast<std::uint32_t>(value);
}

bool ids_end_with(std::span<const int> ids, std::span<const int> tail) noexcept {
    if (tail.size() > ids.size()) { return false; }
    return std::equal(tail.begin(), tail.end(), ids.end() - static_cast<std::ptrdiff_t>(tail.size()));
}

bool try_committed_ids_from_full(const Tokenizer& tokenizer, std::string_view committed,
                                 std::string_view full, std::span<const int> full_ids,
                                 const std::optional<CopiedCommitted>& hit, std::vector<int>& out) {
    const std::size_t n = committed.size();
    if (n == 0 || n > kHostEncodeCacheMaxBytes) { return false; }
    if (!full.starts_with(committed)) { return false; }
    if (!tokenizer.is_encode_loop_pos(full, n)) { return false; }

    if (hit && hit->bytes.size() <= n && bytes_equal_prefix(hit->bytes, committed)) {
        std::vector<int> extra = tokenizer.encode(committed.substr(hit->bytes.size()));
        out.clear();
        out.reserve(hit->ids.size() + extra.size());
        out.insert(out.end(), hit->ids.begin(), hit->ids.end());
        out.insert(out.end(), extra.begin(), extra.end());
    } else {
        const std::vector<int> tail = tokenizer.encode(full.substr(n));
        if (!ids_end_with(full_ids, tail)) { return false; }
        out.assign(full_ids.begin(),
                   full_ids.end() - static_cast<std::ptrdiff_t>(tail.size()));
    }
    if (out.empty() || out.size() > kHostEncodeCacheMaxIds) { return false; }
    return true;
}

} // namespace

bool host_encode_verify_enabled() noexcept {
    if (const char* env = std::getenv("NINFER_VERIFY_HOST_ENCODE")) {
        return env[0] != '\0' && std::strcmp(env, "0") != 0;
    }
    if (asan_build()) { return true; }
#ifndef NDEBUG
    return true;
#else
    return false;
#endif
}

std::optional<CopiedCommitted>
EncodedHistoryCache::copy_longest_prefix(std::string_view full, const Tokenizer& tokenizer,
                                         std::optional<std::size_t> checkpoint_offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t best_index = entries_.size();
    std::size_t best_n     = 0;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        const Entry& entry = entries_[i];
        const std::size_t n = entry.bytes.size();
        if (n == 0 || n > full.size() || n <= best_n) { continue; }
        if (std::memcmp(entry.bytes.data(), full.data(), n) != 0) { continue; }
        if (!tokenizer.is_encode_loop_pos(full, n)) { continue; }
        if (checkpoint_offset && *checkpoint_offset < n) { continue; }
        best_index = i;
        best_n     = n;
    }
    if (best_index == entries_.size()) { return std::nullopt; }
    entries_[best_index].stamp = ++clock_;
    return CopiedCommitted{.bytes = entries_[best_index].bytes, .ids = entries_[best_index].ids};
}

void EncodedHistoryCache::insert_committed(std::string bytes, std::vector<int> ids) {
    if (bytes.empty() || ids.empty() || bytes.size() > kHostEncodeCacheMaxBytes ||
        ids.size() > kHostEncodeCacheMaxIds) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (Entry& entry : entries_) {
        if (entry.bytes == bytes) {
            entry.ids   = std::move(ids);
            entry.stamp = ++clock_;
            return;
        }
    }
    if (entries_.size() >= kHostEncodeCacheEntries) {
        std::size_t lru = 0;
        for (std::size_t i = 1; i < entries_.size(); ++i) {
            if (entries_[i].stamp < entries_[lru].stamp) { lru = i; }
        }
        entries_[lru] = Entry{.bytes = std::move(bytes), .ids = std::move(ids), .stamp = ++clock_};
        return;
    }
    entries_.push_back(
        Entry{.bytes = std::move(bytes), .ids = std::move(ids), .stamp = ++clock_});
}

void EncodedHistoryCache::drop_committed(std::string_view bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::erase_if(entries_, [&](const Entry& entry) { return entry.bytes == bytes; });
}

void EncodedHistoryCache::poison_committed_ids(std::string_view bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Entry& entry : entries_) {
        if (entry.bytes == bytes) {
            for (int& id : entry.ids) { id ^= 0x00ffffff; }
            return;
        }
    }
}

void EncodedHistoryCache::scramble_committed_bytes(std::string_view bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Entry& entry : entries_) {
        if (entry.bytes == bytes && !entry.bytes.empty()) {
            entry.bytes.front() =
                static_cast<char>(static_cast<unsigned char>(entry.bytes.front()) ^ 0xff);
            return;
        }
    }
}

std::size_t EncodedHistoryCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

std::optional<EncodedChat>
try_splice_encoded_chat(const Tokenizer& tokenizer, std::span<const int> committed_ids,
                        std::string_view full, std::size_t n,
                        const std::optional<RewriteCheckpointByteSpec>& checkpoint) {
    if (n > full.size() || !tokenizer.is_encode_loop_pos(full, n)) { return std::nullopt; }
    const std::string_view suffix = full.substr(n);

    EncodedChat encoded;
    auto append_suffix = [&](const EncodedText& extra) {
        encoded.input_ids.reserve(committed_ids.size() + extra.ids.size());
        encoded.input_ids.assign(committed_ids.begin(), committed_ids.end());
        encoded.input_ids.insert(encoded.input_ids.end(), extra.ids.begin(), extra.ids.end());
    };

    if (!checkpoint) {
        append_suffix(tokenizer.encode(suffix, std::nullopt));
        return encoded;
    }
    if (checkpoint->offset < n || checkpoint->offset > full.size()) { return std::nullopt; }
    const std::size_t rel = checkpoint->offset - n;
    if (rel == 0) {
        append_suffix(tokenizer.encode(suffix, std::nullopt));
        encoded.rewrite_checkpoint = RewriteCheckpointSpec{
            .kind     = checkpoint->kind,
            .frontier = checked_frontier(committed_ids.size(), "rewrite checkpoint token frontier"),
        };
        return encoded;
    }

    EncodedText extra = tokenizer.encode(suffix, rel);
    if (!extra.prefix_tokens || *extra.prefix_tokens == 0 ||
        *extra.prefix_tokens > extra.ids.size()) {
        return std::nullopt;
    }
    append_suffix(extra);
    encoded.rewrite_checkpoint = RewriteCheckpointSpec{
        .kind     = checkpoint->kind,
        .frontier = checked_frontier(committed_ids.size() + *extra.prefix_tokens,
                                     "rewrite checkpoint token frontier"),
    };
    return encoded;
}

EncodedChat encode_chat_with_cache(const Tokenizer& tokenizer,
                                   const CompiledChatTemplate& chat_template,
                                   const std::vector<ChatMessage>& messages,
                                   const ChatRenderOptions& options, EncodedHistoryCache& cache) {
    last_host_encode_observation = {};

    ChatRenderOptions committed_options     = options;
    committed_options.add_generation_prompt = false;
    const RenderedChat committed            = chat_template.render(messages, committed_options);
    const RenderedChat full =
        options.add_generation_prompt ? chat_template.render(messages, options) : committed;

    if (!full.text.starts_with(committed.text)) {
        return encode_rendered_chat(tokenizer, full);
    }

    std::optional<std::size_t> checkpoint_offset;
    if (full.rewrite_checkpoint) { checkpoint_offset = full.rewrite_checkpoint->offset; }

    std::optional<CopiedCommitted> hit =
        cache.copy_longest_prefix(full.text, tokenizer, checkpoint_offset);
    EncodedChat encoded;
    if (hit) {
        last_host_encode_observation.attempted_prefix = true;
        last_host_encode_observation.prefix_bytes     = hit->bytes.size();
        if (auto spliced = try_splice_encoded_chat(tokenizer, hit->ids, full.text, hit->bytes.size(),
                                                   full.rewrite_checkpoint)) {
            encoded                              = std::move(*spliced);
            last_host_encode_observation.cache_hit = true;
        }
    }
    if (!last_host_encode_observation.cache_hit) { encoded = encode_rendered_chat(tokenizer, full); }

    if (last_host_encode_observation.cache_hit && host_encode_verify_enabled()) {
        const EncodedChat cold = encode_rendered_chat(tokenizer, full);
        if (cold.input_ids != encoded.input_ids ||
            cold.rewrite_checkpoint != encoded.rewrite_checkpoint) {
            last_host_encode_observation.verified_mismatch = true;
            last_host_encode_observation.cache_hit         = false;
            if (hit) { cache.drop_committed(hit->bytes); }
            hit.reset();
            encoded = cold;
        }
    }

    std::vector<int> committed_ids;
    if (try_committed_ids_from_full(tokenizer, committed.text, full.text, encoded.input_ids, hit,
                                    committed_ids)) {
        cache.insert_committed(committed.text, std::move(committed_ids));
        last_host_encode_observation.inserted = true;
    }
    return encoded;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
