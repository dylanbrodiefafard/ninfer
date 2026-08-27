#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/processor.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

inline constexpr std::size_t kHostEncodeCacheEntries     = 16;
inline constexpr std::size_t kHostEncodeCacheMaxIds      = 262144;
inline constexpr std::size_t kHostEncodeCacheMaxBytes    = 2 * 1024 * 1024;

struct HostEncodeObservation {
    bool cache_hit          = false;
    bool attempted_prefix   = false;
    bool verified_mismatch  = false;
    bool inserted           = false;
    std::size_t prefix_bytes = 0;
};

inline thread_local HostEncodeObservation last_host_encode_observation{};

[[nodiscard]] bool host_encode_verify_enabled() noexcept;

struct CopiedCommitted {
    std::string bytes;
    std::vector<int> ids;
};

class EncodedHistoryCache {
public:
    [[nodiscard]] std::optional<CopiedCommitted>
    copy_longest_prefix(std::string_view full, const Tokenizer& tokenizer,
                        std::optional<std::size_t> checkpoint_offset);

    void insert_committed(std::string bytes, std::vector<int> ids);
    void drop_committed(std::string_view bytes);
    void poison_committed_ids(std::string_view bytes);
    void scramble_committed_bytes(std::string_view bytes);

    [[nodiscard]] std::size_t size() const;

private:
    struct Entry {
        std::string bytes;
        std::vector<int> ids;
        std::uint64_t stamp = 0;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
    std::uint64_t clock_ = 0;
};

[[nodiscard]] std::optional<EncodedChat>
try_splice_encoded_chat(const Tokenizer& tokenizer, std::span<const int> committed_ids,
                        std::string_view full, std::size_t n,
                        const std::optional<RewriteCheckpointByteSpec>& checkpoint);

[[nodiscard]] EncodedChat encode_chat_with_cache(const Tokenizer& tokenizer,
                                                 const CompiledChatTemplate& chat_template,
                                                 const std::vector<ChatMessage>& messages,
                                                 const ChatRenderOptions& options,
                                                 EncodedHistoryCache& cache);

} // namespace ninfer::targets::qwen3_6::frontend_internal

namespace ninfer::targets::qwen3_6 {

class EncodedHistoryPrepare {
public:
    [[nodiscard]] static PreparedPrompt prepare(const Frontend& frontend, PromptInput input,
                                                frontend_internal::EncodedHistoryCache& cache);
    [[nodiscard]] static std::uint32_t count_tokens(const Frontend& frontend, PromptInput input,
                                                    frontend_internal::EncodedHistoryCache& cache);
};

} // namespace ninfer::targets::qwen3_6
