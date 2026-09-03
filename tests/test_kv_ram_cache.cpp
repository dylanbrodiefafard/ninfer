#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct PlannedPagedCache {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

PlannedPagedCache
plan_paged_cache(std::uint32_t pages, std::uint32_t logical_pages, std::int32_t rows,
                 std::vector<ninfer::PagedKVPlaneSpec> planes,
                 ninfer::PagedKVPlaneOrder order = ninfer::PagedKVPlaneOrder::PageMajor) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = pages,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .plane_order           = order,
                                                       .planes                = std::move(planes)});
    return PlannedPagedCache{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

std::optional<std::uint64_t>
capture_or_evict(ninfer::targets::qwen3_6::detail::KVRamCache& cache,
                 const ninfer::targets::qwen3_6::detail::RamCaptureSource& source) {
    for (;;) {
        if (auto id = cache.capture(source)) { return id; }
        const auto victim = cache.peek_oldest_unpinned();
        if (!victim) {
            cache.record_drop();
            return std::nullopt;
        }
        cache.evict_one_unpinned(*victim);
    }
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

void fill_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                        unsigned char seed) {
    const auto pages                    = allocation.page_ids();
    const ninfer::PagedKVPlaneOrder order = pool.plane_order();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes(), 0);
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U);
            if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
                std::fill(host.begin() + static_cast<std::ptrdiff_t>(begin),
                          host.begin() + static_cast<std::ptrdiff_t>(begin + tensor.nb[3]), value);
            } else {
                const std::size_t bpp =
                    static_cast<std::size_t>(tensor.ne[3]) * static_cast<std::size_t>(tensor.nb[2]);
                for (std::int32_t head = 0; head < tensor.ne[3]; ++head) {
                    const std::size_t begin = static_cast<std::size_t>(
                        head * tensor.nb[3] + pages[i] * tensor.nb[2]);
                    std::fill(host.begin() + static_cast<std::ptrdiff_t>(begin),
                              host.begin() + static_cast<std::ptrdiff_t>(begin + tensor.nb[2]),
                              static_cast<unsigned char>(value + static_cast<unsigned>(head)));
                }
                (void)bpp;
            }
        }
        CUDA_CHECK(cudaMemcpy(tensor.data, host.data(), host.size(), cudaMemcpyHostToDevice));
    }
}

int expect_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                         unsigned char seed, const char* label) {
    const auto pages                    = allocation.page_ids();
    const ninfer::PagedKVPlaneOrder order = pool.plane_order();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U);
            if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
                for (std::int64_t byte = 0; byte < tensor.nb[3]; ++byte) {
                    if (host[begin + static_cast<std::size_t>(byte)] != value) {
                        std::cerr << label << " PageMajor logical page " << i << " plane " << plane
                                  << " mismatch\n";
                        return 1;
                    }
                }
            } else {
                for (std::int32_t head = 0; head < tensor.ne[3]; ++head) {
                    const unsigned char expected =
                        static_cast<unsigned char>(value + static_cast<unsigned>(head));
                    const std::size_t begin = static_cast<std::size_t>(
                        head * tensor.nb[3] + pages[i] * tensor.nb[2]);
                    for (std::int64_t byte = 0; byte < tensor.nb[2]; ++byte) {
                        if (host[begin + static_cast<std::size_t>(byte)] != expected) {
                            std::cerr << label << " HeadMajor logical page " << i << " head "
                                      << head << " mismatch\n";
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int expect_host_page_major_layout(const unsigned char* image, const ninfer::PagedKVPool& pool,
                                  std::uint32_t page_count, unsigned char seed, const char* label) {
    std::size_t offset = 0;
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        const std::size_t bpp        = static_cast<std::size_t>(tensor.nb[3]);
        for (std::uint32_t i = 0; i < page_count; ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + i + 1U);
            for (std::size_t byte = 0; byte < bpp; ++byte) {
                if (image[offset + i * bpp + byte] != value) {
                    std::cerr << label << " host PageMajor layout mismatch\n";
                    return 1;
                }
            }
        }
        offset += static_cast<std::size_t>(page_count) * bpp;
    }
    return 0;
}

int expect_host_head_major_layout(const unsigned char* image, const ninfer::PagedKVPool& pool,
                                  std::uint32_t page_count, unsigned char seed, const char* label) {
    const ninfer::Tensor& tensor = pool.plane(0);
    const std::size_t bpp =
        static_cast<std::size_t>(tensor.ne[3]) * static_cast<std::size_t>(tensor.nb[2]);
    for (std::uint32_t i = 0; i < page_count; ++i) {
        const unsigned char value = static_cast<unsigned char>(seed + i + 1U);
        for (std::int32_t head = 0; head < tensor.ne[3]; ++head) {
            const unsigned char expected =
                static_cast<unsigned char>(value + static_cast<unsigned>(head));
            const std::size_t begin = static_cast<std::size_t>(i) * bpp +
                                      static_cast<std::size_t>(head) * tensor.nb[2];
            for (std::int64_t byte = 0; byte < tensor.nb[2]; ++byte) {
                if (image[begin + static_cast<std::size_t>(byte)] != expected) {
                    std::cerr << label << " host HeadMajor page " << i << " is not i*bpp packed\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int round_trip_pool(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool,
                    std::uint32_t entitlement, std::uint32_t mapped, unsigned char seed,
                    const char* label) {
    auto source = pool.reserve(entitlement);
    source.materialize_pages(mapped);
    fill_logical_pages(pool, source, seed);
    const std::uint32_t captured = source.mapped_page_count();
    const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(pool, captured);
    ninfer::HostPinnedArena host(std::max<std::size_t>(image_bytes, 256));
    void* image = host.try_alloc(image_bytes, 256);
    if (image == nullptr) {
        std::cerr << label << " host image allocation failed\n";
        return 1;
    }
    ninfer::pack_paged_kv_allocation_to_host(source, pool, image, ctx.stream);
    ctx.synchronize_all();
    int failures = 0;
    if (pool.plane_order() == ninfer::PagedKVPlaneOrder::PageMajor) {
        failures += expect_host_page_major_layout(static_cast<const unsigned char*>(image), pool,
                                                  captured, seed, label);
    } else {
        failures += expect_host_head_major_layout(static_cast<const unsigned char*>(image), pool,
                                                  captured, seed, label);
    }
    source.release();

    auto destination = pool.reserve(entitlement);
    destination.materialize_pages(mapped);
    ninfer::unpack_paged_kv_allocation_from_host(destination, pool, image, captured, mapped,
                                                 ctx.stream);
    ctx.synchronize_all();
    failures += expect_logical_pages(pool, destination, seed, label);
    destination.release();
    return failures;
}

ninfer::targets::qwen3_6::PreparedPromptData text_prompt(std::vector<ninfer::TokenId> tokens) {
    ninfer::targets::qwen3_6::PreparedPromptData prompt;
    prompt.token_ids   = std::move(tokens);
    prompt.token_types.assign(prompt.token_ids.size(), 0);
    prompt.positions.resize(3 * prompt.token_ids.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < prompt.token_ids.size(); ++i) {
            prompt.positions[static_cast<std::size_t>(axis) * prompt.token_ids.size() + i] =
                static_cast<std::int32_t>(i);
        }
    }
    return prompt;
}

int capture_text_entry(ninfer::targets::qwen3_6::detail::KVRamCache& cache,
                       ninfer::PagedKVPool& pool, ninfer::PagedKVAllocation& alloc,
                       const ninfer::targets::qwen3_6::PreparedPromptData& prompt,
                       cudaStream_t stream, std::uint32_t checkpoint_frontier = 0) {
    ninfer::targets::qwen3_6::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    ninfer::targets::qwen3_6::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    ninfer::targets::qwen3_6::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = ninfer::targets::qwen3_6::detail::prefix_hash_at(
        retained.token_ids, identity, source.execution_frontier);
    if (checkpoint_frontier != 0) {
        source.rewrite_valid     = true;
        source.rewrite_kind      = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure;
        source.rewrite_frontier  = checkpoint_frontier;
        source.hash_c_valid      = true;
        source.hash_c            = ninfer::targets::qwen3_6::detail::prefix_hash_at(
            retained.token_ids, identity, checkpoint_frontier);
    }
    source.text      = &alloc;
    source.text_pool = &pool;
    source.stream    = stream;
    return capture_or_evict(cache, source) ? 0 : 1;
}

int capture_with_hidden(ninfer::targets::qwen3_6::detail::KVRamCache& cache,
                        ninfer::PagedKVPool& pool, ninfer::PagedKVAllocation& alloc,
                        const ninfer::targets::qwen3_6::PreparedPromptData& prompt,
                        const ninfer::Tensor& hidden, cudaStream_t stream) {
    ninfer::targets::qwen3_6::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    ninfer::targets::qwen3_6::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    ninfer::targets::qwen3_6::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = ninfer::targets::qwen3_6::detail::prefix_hash_at(
        retained.token_ids, identity, source.execution_frontier);
    source.text        = &alloc;
    source.text_pool   = &pool;
    source.tail_hidden = &hidden;
    source.stream      = stream;
    return capture_or_evict(cache, source) ? 0 : 1;
}

int test_kv_ram_index(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    int failures  = 0;
    auto alloc    = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ctx.synchronize_all();

    const auto prompt_a = text_prompt({10, 11, 12, 13});
    const auto prompt_b = text_prompt({20, 21, 22, 23});
    const auto prompt_c = text_prompt({30, 31, 32, 33});
    const auto chain_a  = q36::detail::prefix_hash_chain(prompt_a);
    const auto chain_b  = q36::detail::prefix_hash_chain(prompt_b);
    const auto chain_c  = q36::detail::prefix_hash_chain(prompt_c);

    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        std::cerr << "RAM probe capture failed\n";
        alloc.release();
        return 1;
    }
    ctx.synchronize_all();
    const std::size_t entry_bytes = probe.snapshot().used_bytes;
    if (entry_bytes == 0) {
        std::cerr << "RAM probe capture used zero bytes\n";
        alloc.release();
        return 1;
    }

    q36::detail::KVRamCache oversize(4096);
    if (capture_text_entry(oversize, pool, alloc, prompt_a, ctx.copy_stream) == 0 ||
        oversize.snapshot().drops == 0) {
        std::cerr << "oversize capture did not drop\n";
        ++failures;
    }

    q36::detail::KVRamCache pinned(entry_bytes);
    if (capture_text_entry(pinned, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        std::cerr << "single-entry RAM capture failed\n";
        alloc.release();
        return failures + 1;
    }
    ctx.synchronize_all();
    const auto pinned_match = pinned.plan_match(prompt_a, chain_a);
    if (!pinned_match) {
        std::cerr << "single-entry RAM capture did not match\n";
        ++failures;
    } else {
        pinned.claim(pinned_match->entry_id);
        if (capture_text_entry(pinned, pool, alloc, prompt_b, ctx.copy_stream) == 0 ||
            pinned.snapshot().drops == 0 || pinned.snapshot().entry_count != 1) {
            std::cerr << "claimed RAM entry did not survive a capture storm\n";
            ++failures;
        }
        if (pinned.plan_match(prompt_a, chain_a)) {
            std::cerr << "plan_match returned a claimed RAM entry during a capture storm\n";
            ++failures;
        }
        pinned.release(pinned_match->entry_id);
        if (!pinned.plan_match(prompt_a, chain_a)) {
            std::cerr << "release after a capture storm did not restore the match\n";
            ++failures;
        }
        pinned.claim(pinned_match->entry_id);
        pinned.consume(pinned_match->entry_id);
        if (pinned.plan_match(prompt_a, chain_a) || pinned.snapshot().restores != 1) {
            std::cerr << "consume did not remove the RAM entry\n";
            ++failures;
        }
    }

    q36::detail::KVRamCache cache(4ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt_a, ctx.copy_stream) != 0 ||
        capture_text_entry(cache, pool, alloc, prompt_b, ctx.copy_stream) != 0) {
        std::cerr << "RAM index capture failed\n";
        alloc.release();
        return failures + 1;
    }
    ctx.synchronize_all();

    const auto match_a = cache.plan_match(prompt_a, chain_a);
    const auto match_b = cache.plan_match(prompt_b, chain_b);
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id ||
        match_a->reuse_base != prompt_a.token_ids.size() ||
        match_b->reuse_base != prompt_b.token_ids.size()) {
        std::cerr << "RAM index failed to match captured prompts\n";
        ++failures;
    }

    auto changed = prompt_a;
    changed.token_ids.back() += 1;
    const auto before_exact = cache.exact_comparisons();
    const auto miss_hash    = cache.plan_match(changed, q36::detail::prefix_hash_chain(changed));
    if (miss_hash || cache.exact_comparisons() != before_exact) {
        std::cerr << "hash-prefilter miss performed an exact comparison\n";
        ++failures;
    }

    q36::PreparedPromptData vision = prompt_a;
    q36::VisionItem item;
    item.modality    = q36::PromptModality::Image;
    item.grid        = {.temporal = 1, .height = 2, .width = 4};
    item.patch_count = 8;
    item.token_spans = {{.begin = 0, .count = 4}};
    item.content_digest.fill(1);
    vision.token_types.assign(4, static_cast<std::uint8_t>(q36::PromptModality::Image));
    vision.vision_items.push_back(item);
    q36::detail::KVRamCache vision_cache(4ULL << 20);
    if (capture_text_entry(vision_cache, pool, alloc, vision, ctx.copy_stream) != 0) {
        std::cerr << "vision RAM capture failed\n";
        alloc.release();
        return failures + 1;
    }
    ctx.synchronize_all();
    const auto vision_match =
        vision_cache.plan_match(vision, q36::detail::prefix_hash_chain(vision));
    if (!vision_match) {
        std::cerr << "vision RAM entry did not match itself\n";
        ++failures;
    } else {
        vision_cache.test_tamper_identity_digest(vision_match->entry_id, 9);
        const auto before_tamper = vision_cache.exact_comparisons();
        const auto tampered =
            vision_cache.plan_match(vision, q36::detail::prefix_hash_chain(vision));
        if (tampered || vision_cache.exact_comparisons() <= before_tamper) {
            std::cerr << "exact-gate digest tamper did not miss via prefix_matches\n";
            ++failures;
        }
    }

    q36::detail::KVRamCache two(entry_bytes * 2 + 256);
    if (capture_text_entry(two, pool, alloc, prompt_a, ctx.copy_stream) != 0 ||
        capture_text_entry(two, pool, alloc, prompt_b, ctx.copy_stream) != 0) {
        std::cerr << "two-entry RAM capture failed\n";
        ++failures;
    } else {
        ctx.synchronize_all();
        const auto keep = two.plan_match(prompt_a, chain_a);
        if (!keep) {
            std::cerr << "two-entry cache lost prompt A\n";
            ++failures;
        } else {
            const auto version_before_claim = two.index_version();
            two.claim(keep->entry_id);
            if (two.index_version() == version_before_claim) {
                std::cerr << "claim did not bump the RAM index version\n";
                ++failures;
            }
            if (two.plan_match(prompt_a, chain_a)) {
                std::cerr << "plan_match returned a claimed RAM entry\n";
                ++failures;
            }
            (void)capture_text_entry(two, pool, alloc, prompt_c, ctx.copy_stream);
            ctx.synchronize_all();
            if (two.plan_match(prompt_a, chain_a) || two.plan_match(prompt_b, chain_b) ||
                two.snapshot().entry_count != 2) {
                std::cerr << "pinned FIFO eviction removed the claimed entry\n";
                ++failures;
            }
            (void)chain_c;
            two.release(keep->entry_id);
            if (!two.plan_match(prompt_a, chain_a)) {
                std::cerr << "release did not make the claimed entry matchable\n";
                ++failures;
            }
        }
    }

    q36::detail::KVRamCache fifo(4ULL << 20);
    if (capture_text_entry(fifo, pool, alloc, prompt_a, ctx.copy_stream) != 0 ||
        capture_text_entry(fifo, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        std::cerr << "duplicate RAM capture failed\n";
        ++failures;
    } else {
        ctx.synchronize_all();
        const auto oldest = fifo.plan_match(prompt_a, chain_a);
        if (!oldest || oldest->reuse_base != prompt_a.token_ids.size()) {
            std::cerr << "duplicate RAM captures did not match the captured frontier\n";
            ++failures;
        }
    }

    auto longer = prompt_a;
    longer.token_ids.push_back(99);
    longer.token_types.push_back(0);
    longer.positions.resize(3 * longer.token_ids.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < longer.token_ids.size(); ++i) {
            longer.positions[static_cast<std::size_t>(axis) * longer.token_ids.size() + i] =
                static_cast<std::int32_t>(i);
        }
    }
    const auto longer_match =
        fifo.plan_match(longer, q36::detail::prefix_hash_chain(longer));
    if (!longer_match || longer_match->reuse_base != prompt_a.token_ids.size()) {
        std::cerr << "RAM index did not match a longer prompt at the captured frontier\n";
        ++failures;
    }

    q36::detail::KVRamCache lengths(8ULL << 20);
    if (capture_text_entry(lengths, pool, alloc, prompt_a, ctx.copy_stream) != 0 ||
        capture_text_entry(lengths, pool, alloc, longer, ctx.copy_stream) != 0) {
        std::cerr << "different-length RAM capture failed\n";
        ++failures;
    } else {
        const auto longest = lengths.plan_match(longer, q36::detail::prefix_hash_chain(longer));
        if (!longest || longest->reuse_base != longer.token_ids.size()) {
            std::cerr << "RAM index did not prefer the longer captured frontier\n";
            ++failures;
        }
    }

    auto checkpoint_prompt = text_prompt({10, 11, 12, 13});
    q36::detail::KVRamCache checkpoint(8ULL << 20);
    if (capture_text_entry(checkpoint, pool, alloc, checkpoint_prompt, ctx.copy_stream, 2) != 0) {
        std::cerr << "checkpoint RAM capture failed\n";
        ++failures;
    } else {
        const auto short_prompt = text_prompt({10, 11});
        const auto restored =
            checkpoint.plan_match(short_prompt, q36::detail::prefix_hash_chain(short_prompt));
        if (!restored || restored->reuse_base != 2 ||
            restored->reuse != ninfer::PrefixReusePath::RestoreTurnCheckpoint) {
            std::cerr << "RAM index did not match the rewrite checkpoint prefix\n";
            ++failures;
        }
    }

    q36::detail::KVRamCache tight(entry_bytes + 256);
    if (capture_text_entry(tight, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        std::cerr << "tight FIFO first capture failed\n";
        ++failures;
    } else {
        ctx.synchronize_all();
        const auto evictions_before = tight.snapshot().evictions;
        if (capture_text_entry(tight, pool, alloc, prompt_b, ctx.copy_stream) != 0) {
            std::cerr << "tight FIFO second capture failed\n";
            ++failures;
        } else {
            ctx.synchronize_all();
            if (tight.plan_match(prompt_a, chain_a) || !tight.plan_match(prompt_b, chain_b) ||
                tight.snapshot().evictions <= evictions_before) {
                std::cerr << "unpinned FIFO eviction did not drop the oldest entry\n";
                ++failures;
            }
        }
    }

    alloc.release();
    return failures;
}

int test_unpack_consume_and_drop(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    int failures  = 0;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 13);
    ctx.synchronize_all();

    const auto prompt = text_prompt({40, 41, 42, 43});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, source, prompt, ctx.copy_stream) != 0) {
        std::cerr << "unpack-consume capture failed\n";
        source.release();
        return 1;
    }
    const auto saved = cache.harvest_copy_seconds();
    if (saved.save < 0.0 || saved.load != 0.0 || cache.snapshot().save_seconds != saved.save) {
        std::cerr << "capture copy elapsed was not harvested as save\n";
        source.release();
        return 1;
    }
    source.release();

    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        std::cerr << "unpack-consume capture did not index\n";
        return 1;
    }

    auto full = pool.reserve(2);
    full.materialize_pages(2, ctx.stream);
    q36::detail::RamRestoreTarget full_target;
    full_target.text           = &full;
    full_target.text_pool      = &pool;
    full_target.text_dst_pages = 2;
    full_target.stream         = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, full_target);
    const auto loaded = cache.harvest_copy_seconds();
    if (loaded.save != 0.0 || loaded.load < 0.0 || cache.snapshot().load_seconds != loaded.load) {
        std::cerr << "restore copy elapsed was not harvested as load\n";
        ++failures;
    }
    cache.consume(match->entry_id);
    if (cache.snapshot().entry_count != 0 || cache.snapshot().restores != 1 ||
        cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        std::cerr << "consume did not drop the RAM entry from the index\n";
        ++failures;
    }
    failures += expect_logical_pages(pool, full, 13, "unpack then consume");
    full.release();

    auto again = pool.reserve(2);
    again.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, again, 14);
    ctx.synchronize_all();
    q36::detail::KVRamCache prefix_cache(8ULL << 20);
    if (capture_text_entry(prefix_cache, pool, again, prompt, ctx.copy_stream) != 0) {
        std::cerr << "prefix-unpack capture failed\n";
        again.release();
        return failures + 1;
    }
    again.release();
    const auto prefix_match =
        prefix_cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!prefix_match) {
        std::cerr << "prefix-unpack capture did not index\n";
        return failures + 1;
    }
    auto prefix_dest = pool.reserve(2);
    prefix_dest.materialize_pages(1, ctx.stream);
    q36::detail::RamRestoreTarget prefix_target;
    prefix_target.text           = &prefix_dest;
    prefix_target.text_pool      = &pool;
    prefix_target.text_dst_pages = 1;
    prefix_target.stream         = ctx.copy_stream;
    prefix_cache.claim(prefix_match->entry_id);
    (void)prefix_cache.unpack_device(prefix_match->entry_id, prefix_target);
    prefix_cache.consume(prefix_match->entry_id);
    failures += expect_logical_pages(pool, prefix_dest, 14, "checkpoint-sized unpack prefix");
    prefix_dest.release();

    q36::detail::KVRamCache tiny(256);
    auto tiny_source = pool.reserve(2);
    tiny_source.materialize_pages(2, ctx.stream);
    const auto drops_before = tiny.snapshot().drops;
    if (capture_text_entry(tiny, pool, tiny_source, prompt, ctx.copy_stream) == 0 ||
        tiny.snapshot().drops <= drops_before || tiny.snapshot().entry_count != 0) {
        std::cerr << "over-capacity capture did not drop\n";
        ++failures;
    }
    tiny_source.release();
    return failures;
}

int expect_logical_page(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                        std::size_t logical_index, unsigned char seed, const char* label) {
    const auto pages = allocation.page_ids();
    if (logical_index >= pages.size()) {
        std::cerr << label << " logical page index out of range\n";
        return 1;
    }
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        const unsigned char value = static_cast<unsigned char>(
            seed + plane * 17U + static_cast<unsigned>(logical_index) + 1U);
        const std::size_t begin =
            static_cast<std::size_t>(pages[logical_index] * tensor.nb[3]);
        for (std::int64_t byte = 0; byte < tensor.nb[3]; ++byte) {
            if (host[begin + static_cast<std::size_t>(byte)] != value) {
                std::cerr << label << " logical page " << logical_index << " plane " << plane
                          << " mismatch\n";
                return 1;
            }
        }
    }
    return 0;
}

int test_frontier_beats_checkpoint(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto alloc    = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    const auto prompt = text_prompt({10, 11, 12, 13});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt, ctx.copy_stream, 2) != 0) {
        alloc.release();
        return fail("frontier-beats-checkpoint capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    alloc.release();
    if (!match || match->reuse != ninfer::PrefixReusePath::AppendAtFrontier ||
        match->reuse_base != prompt.token_ids.size()) {
        std::cerr << "frontier match lost to the shorter checkpoint\n";
        return 1;
    }
    return 0;
}

int test_asymmetric_fifo(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    const auto prompt_a = text_prompt({10, 11, 12, 13});
    const auto prompt_b = text_prompt({20, 21, 22, 23});
    const auto prompt_c = text_prompt({30, 31, 32, 33});
    auto small          = pool.reserve(1);
    small.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, small, prompt_a, ctx.copy_stream) != 0) {
        small.release();
        return fail("asymmetric FIFO probe capture failed");
    }
    ctx.synchronize_all();
    const std::size_t small_bytes = probe.snapshot().used_bytes;
    auto large                    = pool.reserve(2);
    large.materialize_pages(2, ctx.stream);
    q36::detail::KVRamCache large_probe(8ULL << 20);
    if (capture_text_entry(large_probe, pool, large, prompt_c, ctx.copy_stream) != 0) {
        small.release();
        large.release();
        return fail("asymmetric FIFO large probe capture failed");
    }
    ctx.synchronize_all();
    const std::size_t large_bytes = large_probe.snapshot().used_bytes;
    if (large_bytes <= small_bytes) {
        small.release();
        large.release();
        return fail("two-page capture was not larger than one-page capture");
    }
    q36::detail::KVRamCache cache(small_bytes * 2 + 256);
    if (capture_text_entry(cache, pool, small, prompt_a, ctx.copy_stream) != 0 ||
        capture_text_entry(cache, pool, small, prompt_b, ctx.copy_stream) != 0) {
        small.release();
        large.release();
        return fail("asymmetric FIFO small captures failed");
    }
    ctx.synchronize_all();
    const auto evictions_before = cache.snapshot().evictions;
    const auto drops_before     = cache.snapshot().drops;
    if (capture_text_entry(cache, pool, large, prompt_c, ctx.copy_stream) != 0) {
        small.release();
        large.release();
        return fail("asymmetric FIFO large capture failed");
    }
    ctx.synchronize_all();
    int failures = 0;
    if (cache.snapshot().evictions < evictions_before + 2 || cache.snapshot().drops != drops_before ||
        cache.snapshot().entry_count != 1 ||
        cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a)) ||
        cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b)) ||
        !cache.plan_match(prompt_c, q36::detail::prefix_hash_chain(prompt_c))) {
        std::cerr << "asymmetric FIFO did not evict both small entries: evictions="
                  << cache.snapshot().evictions - evictions_before
                  << " drops=" << cache.snapshot().drops - drops_before
                  << " entries=" << cache.snapshot().entry_count << '\n';
        ++failures;
    }
    small.release();
    large.release();
    return failures;
}

int test_fifo_consume_middle(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    const auto prompt_a = text_prompt({10, 11, 12, 13});
    const auto prompt_b = text_prompt({20, 21, 22, 23});
    const auto prompt_c = text_prompt({30, 31, 32, 33});
    auto alloc          = pool.reserve(1);
    alloc.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        alloc.release();
        return fail("FIFO middle first capture failed");
    }
    ctx.synchronize_all();
    const std::size_t bytes_a = cache.snapshot().used_bytes;
    if (capture_text_entry(cache, pool, alloc, prompt_b, ctx.copy_stream) != 0) {
        alloc.release();
        return fail("FIFO middle second capture failed");
    }
    ctx.synchronize_all();
    const std::size_t bytes_ab = cache.snapshot().used_bytes;
    if (capture_text_entry(cache, pool, alloc, prompt_c, ctx.copy_stream) != 0) {
        alloc.release();
        return fail("FIFO middle third capture failed");
    }
    ctx.synchronize_all();
    const std::size_t bytes_abc = cache.snapshot().used_bytes;
    const std::size_t bytes_c   = bytes_abc - bytes_ab;
    const auto match_b          = cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_b) {
        alloc.release();
        return fail("FIFO middle did not match B");
    }
    cache.claim(match_b->entry_id);
    cache.consume(match_b->entry_id);
    int failures = 0;
    if (cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b)) ||
        !cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a)) ||
        !cache.plan_match(prompt_c, q36::detail::prefix_hash_chain(prompt_c)) ||
        cache.snapshot().entry_count != 2 || cache.snapshot().restores != 1 ||
        cache.snapshot().used_bytes != bytes_a + bytes_c) {
        std::cerr << "FIFO consume did not drop the middle host resident: entries="
                  << cache.snapshot().entry_count << " used=" << cache.snapshot().used_bytes
                  << " expected=" << bytes_a + bytes_c << '\n';
        ++failures;
    }
    alloc.release();
    return failures;
}

int test_fifo_evict_after_middle_consume(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    const auto prompt_a = text_prompt({10, 11, 12, 13});
    const auto prompt_b = text_prompt({20, 21, 22, 23});
    const auto prompt_c = text_prompt({30, 31, 32, 33});
    const auto prompt_d = text_prompt({40, 41, 42, 43});
    auto small          = pool.reserve(1);
    small.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, small, prompt_a, ctx.copy_stream) != 0) {
        small.release();
        return fail("FIFO hole probe capture failed");
    }
    ctx.synchronize_all();
    const std::size_t small_bytes = probe.snapshot().used_bytes;
    auto large                    = pool.reserve(2);
    large.materialize_pages(2, ctx.stream);
    q36::detail::KVRamCache large_probe(8ULL << 20);
    if (capture_text_entry(large_probe, pool, large, prompt_d, ctx.copy_stream) != 0) {
        small.release();
        large.release();
        return fail("FIFO hole large probe capture failed");
    }
    ctx.synchronize_all();
    const std::size_t large_bytes = large_probe.snapshot().used_bytes;
    if (large_bytes <= small_bytes || large_bytes > small_bytes * 2) {
        small.release();
        large.release();
        return fail("FIFO hole large capture does not force exactly one eviction after a hole");
    }
    q36::detail::KVRamCache cache(small_bytes * 3 + 256);
    if (capture_text_entry(cache, pool, small, prompt_a, ctx.copy_stream) != 0 ||
        capture_text_entry(cache, pool, small, prompt_b, ctx.copy_stream) != 0 ||
        capture_text_entry(cache, pool, small, prompt_c, ctx.copy_stream) != 0) {
        small.release();
        large.release();
        return fail("FIFO hole small captures failed");
    }
    ctx.synchronize_all();
    const auto match_b = cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_b) {
        small.release();
        large.release();
        return fail("FIFO hole did not match B");
    }
    cache.claim(match_b->entry_id);
    cache.consume(match_b->entry_id);
    const auto evictions_before = cache.snapshot().evictions;
    const auto drops_before     = cache.snapshot().drops;
    if (capture_text_entry(cache, pool, large, prompt_d, ctx.copy_stream) != 0) {
        small.release();
        large.release();
        return fail("FIFO hole large capture failed");
    }
    ctx.synchronize_all();
    int failures = 0;
    if (cache.snapshot().evictions != evictions_before + 1 || cache.snapshot().drops != drops_before ||
        cache.snapshot().entry_count != 2 ||
        cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a)) ||
        cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b)) ||
        !cache.plan_match(prompt_c, q36::detail::prefix_hash_chain(prompt_c)) ||
        !cache.plan_match(prompt_d, q36::detail::prefix_hash_chain(prompt_d))) {
        std::cerr << "FIFO hole did not evict the oldest remaining resident: evictions="
                  << cache.snapshot().evictions - evictions_before
                  << " drops=" << cache.snapshot().drops - drops_before
                  << " entries=" << cache.snapshot().entry_count << '\n';
        ++failures;
    }
    small.release();
    large.release();
    return failures;
}

int test_prefix_unpack_preserves_tail(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 31);
    ctx.synchronize_all();
    const auto prompt = text_prompt({40, 41, 42, 43});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, source, prompt, ctx.copy_stream) != 0) {
        source.release();
        return fail("prefix-tail capture failed");
    }
    ctx.synchronize_all();
    source.release();
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) { return fail("prefix-tail capture did not index"); }
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, dest, 99);
    ctx.synchronize_all();
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    ctx.synchronize_all();
    int failures = 0;
    failures += expect_logical_page(pool, dest, 0, 31, "prefix unpack page 0");
    failures += expect_logical_page(pool, dest, 1, 99, "prefix unpack must not write dest page 1");
    dest.release();
    return failures;
}

int test_consume_reaps_for_next_capture(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 61);
    const auto prompt_a = text_prompt({60, 61, 62, 63});
    const auto prompt_b = text_prompt({70, 71, 72, 73});
    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, source, prompt_a, ctx.copy_stream) != 0) {
        source.release();
        return fail("consume-reap probe capture failed");
    }
    ctx.synchronize_all();
    const std::size_t entry_bytes = probe.snapshot().used_bytes;
    q36::detail::KVRamCache cache(entry_bytes + 256);
    if (capture_text_entry(cache, pool, source, prompt_a, ctx.copy_stream) != 0) {
        source.release();
        return fail("consume-reap first capture failed");
    }
    const auto match = cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    if (!match) {
        source.release();
        return fail("consume-reap first capture did not index");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    const auto drops_before     = cache.snapshot().drops;
    const auto evictions_before = cache.snapshot().evictions;
    if (capture_text_entry(cache, pool, source, prompt_b, ctx.copy_stream) != 0) {
        source.release();
        dest.release();
        return fail("consume-reap second capture failed");
    }
    int failures = 0;
    if (cache.snapshot().drops != drops_before || cache.snapshot().evictions != evictions_before ||
        cache.snapshot().entry_count != 1 ||
        cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a)) ||
        !cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b))) {
        std::cerr << "consume did not return host budget to the next capture\n";
        ++failures;
    }
    ctx.synchronize_all();
    failures += expect_logical_pages(pool, dest, 61, "consume-reap dest after next capture");
    source.release();
    dest.release();
    return failures;
}

int test_copies_ready_query(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    q36::detail::KVRamCache empty(8ULL << 20);
    if (!empty.copies_ready(0) || !empty.copies_ready(1)) {
        return fail("missing RAM entry copies_ready should be true");
    }
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ctx.synchronize_all();
    const auto prompt = text_prompt({70, 71, 72, 73});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt, ctx.copy_stream) != 0) {
        alloc.release();
        return fail("copies_ready capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("copies_ready capture did not match");
    }
    ctx.synchronize_all();
    int failures = 0;
    if (!cache.copies_ready(match->entry_id)) {
        std::cerr << "captured RAM entry copies_ready was false after synchronize_all\n";
        ++failures;
    }
    cache.claim(match->entry_id);
    cache.consume(match->entry_id);
    if (!cache.copies_ready(match->entry_id)) {
        std::cerr << "retired RAM entry copies_ready should be true\n";
        ++failures;
    }
    alloc.release();
    return failures;
}

int test_copy_compute_stream_overlap(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 77);
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr std::size_t kBulkBytes = 32ULL << 20;
    ninfer::DeviceBuffer bulk(kBulkBytes);
    bulk.fill(0x5a);
    ninfer::Tensor hidden(bulk.p, ninfer::DType::U8,
                          {static_cast<std::int32_t>(kBulkBytes)});

    const auto prompt = text_prompt({70, 71, 72, 73});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    q36::detail::RamCaptureSource cap;
    cap.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    cap.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    cap.text_kv_valid      = cap.execution_frontier;
    cap.tail_hidden_valid  = true;
    cap.ledger             = retained.token_ids;
    cap.identity           = &identity;
    cap.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                         cap.execution_frontier);
    cap.text               = &source;
    cap.text_pool          = &pool;
    cap.tail_hidden        = &hidden;
    cap.stream             = ctx.copy_stream;

    q36::detail::KVRamCache cache(64ULL << 20);
    if (!capture_or_evict(cache, cap)) {
        source.release();
        return fail("copy/compute overlap capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        source.release();
        return fail("copy/compute overlap capture did not index");
    }

    ninfer::DeviceBuffer scratch(256);
    cudaEvent_t compute_done{};
    CUDA_CHECK(cudaEventCreate(&compute_done));
    CUDA_CHECK(cudaMemsetAsync(scratch.p, 0, 1, ctx.stream));
    CUDA_CHECK(cudaEventRecord(compute_done, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(compute_done));
    if (cache.copies_ready(match->entry_id)) {
        CUDA_CHECK(cudaEventDestroy(compute_done));
        source.release();
        return fail("capture D2H was drained by a compute-only wait; copies are on device.stream");
    }
    CUDA_CHECK(cudaEventDestroy(compute_done));

    try {
        auto extra = pool.reserve(static_cast<std::uint32_t>(pool.page_group_count()));
        extra.release();
        source.release();
        return fail("in-flight capture still allowed a full-pool reserve");
    } catch (const std::bad_alloc&) {}

    ctx.synchronize_all();
    if (!cache.copies_ready(match->entry_id)) {
        source.release();
        return fail("capture copies_ready stayed false after synchronize_all");
    }
    source.release();

    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out(kBulkBytes);
    ninfer::Tensor hidden_out_t(hidden_out.p, ninfer::DType::U8,
                                {static_cast<std::int32_t>(kBulkBytes)});
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.tail_hidden    = &hidden_out_t;
    target.stream         = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);

    CUDA_CHECK(cudaEventCreate(&compute_done));
    CUDA_CHECK(cudaMemsetAsync(scratch.p, 1, 1, ctx.stream));
    CUDA_CHECK(cudaEventRecord(compute_done, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(compute_done));
    CUDA_CHECK(cudaEventDestroy(compute_done));
    if (cache.copies_ready(match->entry_id)) {
        dest.release();
        return fail("restore H2D was drained by a compute-only wait; copies are on device.stream");
    }
    ctx.synchronize_all();
    cache.consume(match->entry_id);
    const int failures = expect_logical_pages(pool, dest, 77, "copy/compute overlap restore KV");
    dest.release();
    return failures;
}

int test_unpack_without_harvest_keeps_save_and_load(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 33);
    constexpr std::size_t kBulkBytes = 32ULL << 20;
    ninfer::DeviceBuffer bulk(kBulkBytes);
    bulk.fill(0x3c);
    ninfer::Tensor hidden(bulk.p, ninfer::DType::U8, {static_cast<std::int32_t>(kBulkBytes)});
    const auto prompt = text_prompt({33, 34, 35, 36});
    q36::detail::KVRamCache cache(64ULL << 20);
    if (capture_with_hidden(cache, pool, source, prompt, hidden, ctx.copy_stream) != 0) {
        source.release();
        return fail("save/load harvest capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        source.release();
        return fail("save/load harvest capture did not index");
    }
    {
        ninfer::DeviceBuffer scratch(256);
        cudaEvent_t compute_done{};
        CUDA_CHECK(cudaEventCreate(&compute_done));
        CUDA_CHECK(cudaMemsetAsync(scratch.p, 0, 1, ctx.stream));
        CUDA_CHECK(cudaEventRecord(compute_done, ctx.stream));
        CUDA_CHECK(cudaEventSynchronize(compute_done));
        CUDA_CHECK(cudaEventDestroy(compute_done));
        if (cache.pending_copies_ready()) {
            source.release();
            return fail("pending_copies_ready was true during in-flight capture D2H");
        }
    }
    source.release();
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out(kBulkBytes);
    ninfer::Tensor hidden_out_t(hidden_out.p, ninfer::DType::U8,
                                {static_cast<std::int32_t>(kBulkBytes)});
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.tail_hidden    = &hidden_out_t;
    target.stream         = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    const auto copies = cache.harvest_copy_seconds();
    int failures      = 0;
    if (copies.save <= 0.0) {
        std::cerr << "unpack without prior harvest billed D2H as load or dropped it, save="
                  << copies.save << " load=" << copies.load << '\n';
        ++failures;
    }
    if (copies.load <= 0.0) {
        std::cerr << "unpack without prior harvest dropped H2D load timing, save=" << copies.save
                  << " load=" << copies.load << '\n';
        ++failures;
    }
    cache.consume(match->entry_id);
    ctx.synchronize_all();
    failures += expect_logical_pages(pool, dest, 33, "unpack without prior harvest");
    dest.release();
    return failures;
}

int test_consume_without_harvest_clears_pending(ninfer::DeviceContext& ctx,
                                                ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 34);
    constexpr std::size_t kBulkBytes = 32ULL << 20;
    ninfer::DeviceBuffer bulk(kBulkBytes);
    bulk.fill(0x4d);
    ninfer::Tensor hidden(bulk.p, ninfer::DType::U8, {static_cast<std::int32_t>(kBulkBytes)});
    const auto prompt = text_prompt({44, 45, 46, 47});
    q36::detail::KVRamCache cache(64ULL << 20);
    if (capture_with_hidden(cache, pool, source, prompt, hidden, ctx.copy_stream) != 0) {
        source.release();
        return fail("consume-pending capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        source.release();
        return fail("consume-pending capture did not index");
    }
    cache.claim(match->entry_id);
    cache.consume(match->entry_id);
    source.release();
    if (cache.test_pending_copy_count() != 0) {
        return fail("consume left a ghost id in pending_save_ids_ or pending_load_id_");
    }
    if (!cache.pending_copies_ready()) {
        return fail("consume left pending copies unready after retiring the entry");
    }
    const auto copies = cache.harvest_copy_seconds();
    if (copies.save <= 0.0 || copies.load != 0.0) {
        std::cerr << "consume without harvest did not fold D2H into save, save=" << copies.save
                  << " load=" << copies.load << '\n';
        return 1;
    }
    if (cache.snapshot().save_seconds != copies.save || cache.snapshot().entry_count != 0) {
        return fail("consume without harvest left stale snapshot save/entry accounting");
    }
    return 0;
}

int test_consume_after_unpack_folds_load(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 35);
    constexpr std::size_t kBulkBytes = 32ULL << 20;
    ninfer::DeviceBuffer bulk(kBulkBytes);
    bulk.fill(0x5e);
    ninfer::Tensor hidden(bulk.p, ninfer::DType::U8, {static_cast<std::int32_t>(kBulkBytes)});
    const auto prompt = text_prompt({55, 56, 57, 58});
    q36::detail::KVRamCache cache(64ULL << 20);
    if (capture_with_hidden(cache, pool, source, prompt, hidden, ctx.copy_stream) != 0) {
        source.release();
        return fail("consume-after-unpack capture failed");
    }
    const auto saved = cache.harvest_copy_seconds();
    if (saved.save <= 0.0) {
        source.release();
        return fail("consume-after-unpack capture save was not harvested");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        source.release();
        return fail("consume-after-unpack capture did not index");
    }
    source.release();
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out(kBulkBytes);
    ninfer::Tensor hidden_out_t(hidden_out.p, ninfer::DType::U8,
                                {static_cast<std::int32_t>(kBulkBytes)});
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.tail_hidden    = &hidden_out_t;
    target.stream         = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    if (cache.test_pending_copy_count() != 0) {
        dest.release();
        return fail("consume after unpack left a ghost pending copy id");
    }
    const auto loaded = cache.harvest_copy_seconds();
    int failures      = 0;
    if (loaded.save != 0.0 || loaded.load <= 0.0) {
        std::cerr << "consume after unpack did not fold H2D into harvest load, save="
                  << loaded.save << " load=" << loaded.load << '\n';
        ++failures;
    }
    failures += expect_logical_pages(pool, dest, 35, "consume after unpack");
    dest.release();
    return failures;
}

int test_event_overlap_unpack(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 44);
    const auto prompt = text_prompt({50, 51, 52, 53});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, source, prompt, ctx.copy_stream) != 0) {
        source.release();
        return fail("overlap capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        source.release();
        return fail("overlap capture did not index before D2H completion");
    }
    ctx.synchronize_all();
    if (!cache.copies_ready(match->entry_id)) {
        source.release();
        return fail("overlap capture copies_done did not complete after synchronize_all");
    }
    source.release();
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    ctx.synchronize_all();
    const int failures = expect_logical_pages(pool, dest, 44, "event-overlapped unpack");
    dest.release();
    return failures;
}

int test_irregular_page_major_runs(ninfer::DeviceContext& ctx) {
    auto plan = plan_paged_cache(8, 8, 2,
                                 {{ninfer::DType::I8, 64, 2},
                                  {ninfer::DType::I8, 64, 2},
                                  {ninfer::DType::FP16, 1, 2},
                                  {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    std::vector<ninfer::PagedKVAllocation> held;
    for (int i = 0; i < 8; ++i) {
        auto page = pool.reserve(1);
        page.materialize_pages(1, ctx.stream);
        held.push_back(std::move(page));
    }
    held[0].release();
    held[2].release();
    held[3].release();
    held[5].release();
    held[7].release();
    auto source = pool.reserve(5);
    source.materialize_pages(5, ctx.stream);
    const auto ids = source.page_ids();
    bool mixed     = false;
    if (ids.size() >= 3) {
        const std::int32_t delta = ids[1] - ids[0];
        for (std::size_t i = 2; i < ids.size(); ++i) {
            if (ids[i] - ids[i - 1] != delta) {
                mixed = true;
                break;
            }
        }
    }
    if (!mixed) {
        std::cerr << "irregular PageMajor test did not produce mixed strides:";
        for (std::int32_t id : ids) { std::cerr << ' ' << id; }
        std::cerr << '\n';
        source.release();
        return 1;
    }
    fill_logical_pages(pool, source, 73);
    const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(pool, 5);
    ninfer::HostPinnedArena host(std::max<std::size_t>(image_bytes, 256));
    void* image = host.try_alloc(image_bytes, 256);
    ninfer::pack_paged_kv_allocation_to_host(source, pool, image, ctx.stream);
    ctx.synchronize_all();
    source.release();
    auto dest = pool.reserve(5);
    dest.materialize_pages(5, ctx.stream);
    ninfer::unpack_paged_kv_allocation_from_host(dest, pool, image, 5, 5, ctx.stream);
    ctx.synchronize_all();
    const int failures = expect_logical_pages(pool, dest, 73, "irregular PageMajor runs");
    dest.release();
    return failures;
}

int test_restore_throw_then_replay(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 81);
    const auto prompt = text_prompt({80, 81, 82, 83});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xb1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_buf(128);
    rewrite_buf.fill(0xb2);
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {128});
    q36::detail::RamCaptureSource cap;
    cap.execution_frontier        = static_cast<std::uint32_t>(prompt.token_ids.size());
    cap.ledger_frontier           = static_cast<std::uint32_t>(tokens);
    cap.text_kv_valid             = cap.execution_frontier;
    cap.tail_hidden_valid         = true;
    cap.rewrite_valid             = true;
    cap.rewrite_kind              = q36::RewriteCheckpointKind::ResponseReplay;
    cap.rewrite_frontier          = 2;
    cap.hash_c_valid              = true;
    cap.ledger                    = retained.token_ids;
    cap.identity                  = &identity;
    cap.hash_f                    = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                                cap.execution_frontier);
    cap.hash_c                    = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    cap.text                      = &source;
    cap.text_pool                 = &pool;
    cap.tail_hidden               = &hidden;
    cap.rewrite_checkpoint_hidden = &rewrite;
    cap.stream                    = ctx.copy_stream;
    q36::detail::KVRamCache cache(8ULL << 20);
    if (!capture_or_evict(cache, cap)) {
        source.release();
        return fail("throw-after-H2D capture failed");
    }
    ctx.synchronize_all();
    source.release();
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_bad_buf(64);
    rewrite_bad_buf.fill(0);
    ninfer::Tensor rewrite_bad(rewrite_bad_buf.p, ninfer::DType::U8, {64});
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        dest.release();
        return fail("throw-after-H2D capture did not match");
    }
    q36::detail::RamRestoreTarget bad;
    bad.text                      = &dest;
    bad.text_pool                 = &pool;
    bad.text_dst_pages            = 2;
    bad.tail_hidden               = &hidden_out;
    bad.rewrite_checkpoint_hidden = &rewrite_bad;
    bad.stream                    = ctx.copy_stream;
    cache.claim(match->entry_id);
    bool threw = false;
    try {
        (void)cache.unpack_device(match->entry_id, bad);
    } catch (const std::logic_error&) {
        threw = true;
    }
    ctx.synchronize_all();
    cache.release(match->entry_id);
    dest.release();
    if (!threw) { return fail("mismatched rewrite hidden did not throw after KV H2D"); }

    auto dest2 = pool.reserve(2);
    dest2.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer rewrite_ok_buf(128);
    rewrite_ok_buf.fill(0);
    ninfer::Tensor rewrite_ok(rewrite_ok_buf.p, ninfer::DType::U8, {128});
    hidden_out_buf.fill(0);
    q36::detail::RamRestoreTarget ok;
    ok.text                      = &dest2;
    ok.text_pool                 = &pool;
    ok.text_dst_pages            = 2;
    ok.tail_hidden               = &hidden_out;
    ok.rewrite_checkpoint_hidden = &rewrite_ok;
    ok.stream                    = ctx.copy_stream;
    cache.claim(match->entry_id);
    const auto host = cache.unpack_device(match->entry_id, ok);
    cache.consume(match->entry_id);
    ctx.synchronize_all();
    int failures = 0;
    if (!host.rewrite_valid || host.rewrite_kind != q36::RewriteCheckpointKind::ResponseReplay ||
        host.rewrite_frontier != 2) {
        std::cerr << "response-checkpoint host metadata mismatch after failed unpack\n";
        ++failures;
    }
    failures += expect_logical_pages(pool, dest2, 81, "replay after thrown unpack");
    dest2.release();
    return failures;
}

int test_destructor_with_inflight_copies(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 91);
    const auto prompt = text_prompt({90, 91, 92, 93});
    auto dest         = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    {
        q36::detail::KVRamCache cache(8ULL << 20);
        if (capture_text_entry(cache, pool, source, prompt, ctx.copy_stream) != 0) {
            source.release();
            dest.release();
            return fail("destructor-inflight capture failed");
        }
        const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            source.release();
            dest.release();
            return fail("destructor-inflight capture did not match");
        }
        q36::detail::RamRestoreTarget target;
        target.text           = &dest;
        target.text_pool      = &pool;
        target.text_dst_pages = 2;
        target.stream         = ctx.copy_stream;
        cache.claim(match->entry_id);
        (void)cache.unpack_device(match->entry_id, target);
    }
    ctx.synchronize_all();
    const int failures = expect_logical_pages(pool, dest, 91, "destructor drained in-flight H2D");
    source.release();
    dest.release();
    return failures;
}

int test_spill_drop_keeps_indexed_source(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto alloc    = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    const auto prompt_a = text_prompt({11, 12, 13, 14});
    const auto prompt_b = text_prompt({21, 22, 23, 24});
    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        alloc.release();
        return fail("spill-drop probe capture failed");
    }
    ctx.synchronize_all();
    const std::size_t entry_bytes = probe.snapshot().used_bytes;
    q36::detail::KVRamCache cache(entry_bytes + 256);
    if (capture_text_entry(cache, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        alloc.release();
        return fail("spill-drop first capture failed");
    }
    const auto first = cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    if (!first) {
        alloc.release();
        return fail("spill-drop first capture did not index");
    }
    cache.claim(first->entry_id);
    const auto drops_before = cache.snapshot().drops;
    if (capture_text_entry(cache, pool, alloc, prompt_b, ctx.copy_stream) == 0) {
        alloc.release();
        return fail("pinned one-entry budget captured a second bundle");
    }
    int failures = 0;
    if (cache.snapshot().drops != drops_before + 1 || cache.snapshot().entry_count != 1 ||
        cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b))) {
        std::cerr << "spill drop indexed the dropped victim\n";
        ++failures;
    }
    cache.release(first->entry_id);
    if (!cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a))) {
        std::cerr << "spill drop lost the pinned source\n";
        ++failures;
    }
    alloc.release();
    return failures;
}

int test_full_state_image(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto backend_plan =
        plan_paged_cache(4, 4, 1, {{ninfer::DType::I8, 8, 1}, {ninfer::DType::I8, 8, 1}});
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool backend_pool({backend_arena.base(), backend_arena.capacity()},
                                     backend_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 3,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    ninfer::LayoutBuilder cyclic_builder;
    const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 16, 2, 8, 2);
    ninfer::DeviceArena cyclic_arena(cyclic_builder.finish(256));
    ninfer::CyclicKVCache dflash_local({cyclic_arena.base(), cyclic_arena.capacity()},
                                       cyclic_layout);
    ninfer::LayoutBuilder cyclic_ckpt_builder;
    const auto cyclic_ckpt_layout =
        ninfer::plan_cyclic_kv_cache(cyclic_ckpt_builder, 2, 16, 2, 8, 2);
    ninfer::DeviceArena cyclic_ckpt_arena(cyclic_ckpt_builder.finish(256));
    ninfer::CyclicKVCache dflash_ckpt({cyclic_ckpt_arena.base(), cyclic_ckpt_arena.capacity()},
                                      cyclic_ckpt_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 7);
    auto backend = backend_pool.reserve(2);
    backend.materialize_pages(1, ctx.stream);
    fill_logical_pages(backend_pool, backend, 8);

    std::vector<unsigned char> conv_cur(gdn.conv_slot(0, 0).bytes());
    std::vector<unsigned char> conv_ckpt(gdn.conv_slot(0, 1).bytes());
    for (std::size_t i = 0; i < conv_cur.size(); ++i) {
        conv_cur[i]  = static_cast<unsigned char>(i + 1);
        conv_ckpt[i] = static_cast<unsigned char>(i + 9);
    }
    CUDA_CHECK(cudaMemcpy(gdn.conv_slot(0, 0).data, conv_cur.data(), conv_cur.size(),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gdn.conv_slot(2, 0).data, conv_cur.data(), conv_cur.size(),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gdn.conv_slot(0, 1).data, conv_ckpt.data(), conv_ckpt.size(),
                           cudaMemcpyHostToDevice));
    std::vector<unsigned char> rec_cur(gdn.recurrent_slot(0, 0).bytes(), 0x21);
    std::vector<unsigned char> rec_ckpt(gdn.recurrent_slot(0, 1).bytes(), 0x22);
    CUDA_CHECK(cudaMemcpy(gdn.recurrent_slot(1, 0).data, rec_cur.data(), rec_cur.size(),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gdn.recurrent_slot(1, 1).data, rec_ckpt.data(), rec_ckpt.size(),
                           cudaMemcpyHostToDevice));

    ninfer::CyclicKVCacheLayerView local_view = dflash_local.layer_view(0);
    std::vector<unsigned char> k_local(local_view.k.slice(3, 0, 1).bytes(), 0x3c);
    std::vector<unsigned char> v_local(local_view.v.slice(3, 0, 1).bytes(), 0x3d);
    CUDA_CHECK(cudaMemcpy(local_view.k.slice(3, 0, 1).data, k_local.data(), k_local.size(),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(local_view.v.slice(3, 0, 1).data, v_local.data(), v_local.size(),
                           cudaMemcpyHostToDevice));
    ninfer::CyclicKVCacheLayerView ckpt_view = dflash_ckpt.layer_view(0);
    std::vector<unsigned char> k_ckpt(ckpt_view.k.slice(3, 0, 1).bytes(), 0x4c);
    std::vector<unsigned char> v_ckpt(ckpt_view.v.slice(3, 0, 1).bytes(), 0x4d);
    CUDA_CHECK(cudaMemcpy(ckpt_view.k.slice(3, 0, 1).data, k_ckpt.data(), k_ckpt.size(),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ckpt_view.v.slice(3, 0, 1).data, v_ckpt.data(), v_ckpt.size(),
                           cudaMemcpyHostToDevice));

    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_buf(128);
    rewrite_buf.fill(0xa2);
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.rope_delta         = 7;
    source.text_kv_valid      = source.execution_frontier;
    source.mtp_kv_valid       = 3;
    source.tail_hidden_valid  = true;
    source.rewrite_valid      = true;
    source.rewrite_kind       = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier   = 2;
    source.hash_c_valid       = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.hash_c             = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.backend            = &backend;
    source.backend_pool       = &backend_pool;
    source.gdn                = &gdn;
    source.gdn_current_slot   = 0;
    source.gdn_checkpoint_slot = 1;
    source.tail_hidden        = &hidden;
    source.rewrite_checkpoint_hidden = &rewrite;
    source.dflash_local       = &dflash_local;
    source.dflash_checkpoint  = &dflash_ckpt;
    source.dflash_lane        = 0;
    source.stream             = ctx.copy_stream;
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) { return fail("full-state capture failed"); }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    auto backend_dest = backend_pool.reserve(2);
    backend_dest.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_out_buf(128);
    rewrite_out_buf.fill(0);
    ninfer::Tensor rewrite_out(rewrite_out_buf.p, ninfer::DType::U8, {128});
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) { return fail("full-state capture did not match"); }
    q36::detail::RamRestoreTarget target;
    target.text                    = &text_dest;
    target.text_pool               = &text_pool;
    target.text_dst_pages          = 2;
    target.backend                 = &backend_dest;
    target.backend_pool            = &backend_pool;
    target.backend_dst_pages       = 1;
    target.gdn                     = &gdn;
    target.gdn_current_slot        = 2;
    target.gdn_checkpoint_slot     = 3;
    target.tail_hidden             = &hidden_out;
    target.rewrite_checkpoint_hidden = &rewrite_out;
    target.dflash_local            = &dflash_local;
    target.dflash_checkpoint       = &dflash_ckpt;
    target.dflash_lane             = 1;
    target.stream                  = ctx.copy_stream;
    cache.claim(match->entry_id);
    const q36::detail::RamRestoredHost host = cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    ctx.synchronize_all();

    int failures = 0;
    if (host.rope_delta != 7 || host.mtp_kv_valid != 3 || !host.backend_image_present ||
        !host.rewrite_valid || host.rewrite_frontier != 2 ||
        host.ledger.size() != tokens) {
        std::cerr << "full-state host metadata mismatch\n";
        ++failures;
    }
    failures += expect_logical_pages(text_pool, text_dest, 7, "full-state text KV");
    failures += expect_logical_pages(backend_pool, backend_dest, 8, "full-state backend KV");
    std::vector<unsigned char> conv_out(conv_cur.size());
    CUDA_CHECK(cudaMemcpy(conv_out.data(), gdn.conv_slot(0, 2).data, conv_out.size(),
                           cudaMemcpyDeviceToHost));
    if (conv_out != conv_cur) {
        std::cerr << "full-state GDN conv current did not round-trip\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(conv_out.data(), gdn.conv_slot(0, 3).data, conv_out.size(),
                           cudaMemcpyDeviceToHost));
    if (conv_out != conv_ckpt) {
        std::cerr << "full-state GDN conv checkpoint did not round-trip\n";
        ++failures;
    }
    std::vector<unsigned char> rec_out(rec_cur.size());
    CUDA_CHECK(cudaMemcpy(rec_out.data(), gdn.recurrent_slot(1, 2).data, rec_out.size(),
                           cudaMemcpyDeviceToHost));
    if (rec_out != rec_cur) {
        std::cerr << "full-state GDN recurrent current did not round-trip\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(rec_out.data(), gdn.recurrent_slot(1, 3).data, rec_out.size(),
                           cudaMemcpyDeviceToHost));
    if (rec_out != rec_ckpt) {
        std::cerr << "full-state GDN recurrent checkpoint did not round-trip\n";
        ++failures;
    }
    std::vector<unsigned char> hidden_host(128);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != std::vector<unsigned char>(128, 0xa1)) {
        std::cerr << "full-state tail hidden did not round-trip\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), rewrite_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != std::vector<unsigned char>(128, 0xa2)) {
        std::cerr << "full-state rewrite hidden did not round-trip\n";
        ++failures;
    }
    std::vector<unsigned char> k_out(k_local.size());
    CUDA_CHECK(cudaMemcpy(k_out.data(), dflash_local.layer_view(0).k.slice(3, 1, 1).data,
                           k_out.size(), cudaMemcpyDeviceToHost));
    std::vector<unsigned char> v_out(v_local.size());
    CUDA_CHECK(cudaMemcpy(v_out.data(), dflash_local.layer_view(0).v.slice(3, 1, 1).data,
                           v_out.size(), cudaMemcpyDeviceToHost));
    if (k_out != k_local || v_out != v_local) {
        std::cerr << "full-state DFlash local lane did not round-trip\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(k_out.data(), dflash_ckpt.layer_view(0).k.slice(3, 1, 1).data, k_out.size(),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_out.data(), dflash_ckpt.layer_view(0).v.slice(3, 1, 1).data, v_out.size(),
                           cudaMemcpyDeviceToHost));
    if (k_out != k_ckpt || v_out != v_ckpt) {
        std::cerr << "full-state DFlash checkpoint lane did not round-trip\n";
        ++failures;
    }
    text.release();
    backend.release();
    text_dest.release();
    backend_dest.release();
    return failures;
}

int test_context_checkpoint_middle_head(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 3,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 7);

    std::vector<unsigned char> conv_exec(gdn.conv_host_image_bytes(), 0xe1);
    std::vector<unsigned char> rec_exec(gdn.recurrent_host_image_bytes(), 0xe2);
    std::vector<unsigned char> conv_rewrite(gdn.conv_host_image_bytes(), 0xc1);
    std::vector<unsigned char> rec_rewrite(gdn.recurrent_host_image_bytes(), 0xc2);
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemcpy(gdn.conv_slot(layer, 0).data, conv_exec.data(), gdn.conv_slot_bytes(),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gdn.recurrent_slot(layer, 0).data, rec_exec.data(),
                               gdn.recurrent_slot_bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gdn.conv_slot(layer, 1).data, conv_rewrite.data(),
                               gdn.conv_slot_bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gdn.recurrent_slot(layer, 1).data, rec_rewrite.data(),
                               gdn.recurrent_slot_bytes(), cudaMemcpyHostToDevice));
    }

    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_buf(128);
    rewrite_buf.fill(0xa2);
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    std::vector<unsigned char> conv_f2(gdn.conv_host_image_bytes(), 0x21);
    std::vector<unsigned char> rec_f2(gdn.recurrent_host_image_bytes(), 0x22);
    std::vector<unsigned char> hid_f2(128, 0xb2);
    std::vector<unsigned char> conv_f4(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec_f4(gdn.recurrent_host_image_bytes(), 0x42);
    std::vector<unsigned char> hid_f4(128, 0xb4);
    std::vector<unsigned char> conv_f6(gdn.conv_host_image_bytes(), 0x61);
    std::vector<unsigned char> rec_f6(gdn.recurrent_host_image_bytes(), 0x62);
    std::vector<unsigned char> hid_f6(128, 0xb6);
    const auto head_at = [&](std::uint32_t frontier, const void* conv, const void* rec,
                             const void* hid) {
        q36::detail::RamLadderHead head;
        head.frontier        = frontier;
        head.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, frontier);
        head.conv            = conv;
        head.recurrent       = rec;
        head.hidden          = hid;
        head.conv_bytes      = conv_f4.size();
        head.recurrent_bytes = rec_f4.size();
        head.hidden_bytes    = hid_f4.size();
        return head;
    };

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.rope_delta         = 7;
    source.text_kv_valid      = source.execution_frontier;
    source.mtp_kv_valid       = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.rewrite_valid      = true;
    source.rewrite_kind       = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier   = 6;
    source.hash_c_valid       = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.hash_c             = q36::detail::prefix_hash_at(retained.token_ids, identity, 6);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.gdn                = &gdn;
    source.gdn_current_slot   = 0;
    source.gdn_checkpoint_slot = 1;
    source.tail_hidden        = &hidden;
    source.rewrite_checkpoint_hidden = &rewrite;
    source.ladder_heads       = {head_at(2, conv_f2.data(), rec_f2.data(), hid_f2.data()),
                                 head_at(4, conv_f4.data(), rec_f4.data(), hid_f4.data()),
                                 head_at(6, conv_f6.data(), rec_f6.data(), hid_f6.data())};
    source.stream             = ctx.copy_stream;
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) { return fail("context-checkpoint capture failed"); }

    const auto prefix_f2 = text_prompt({1, 2});
    const auto prefix_f4 = text_prompt({1, 2, 3, 4});
    const auto prefix_f6 = text_prompt({1, 2, 3, 4, 5, 6});
    const auto match    = cache.plan_match(prefix_f4, q36::detail::prefix_hash_chain(prefix_f4));
    const auto match_f2 = cache.plan_match(prefix_f2, q36::detail::prefix_hash_chain(prefix_f2));
    const auto match_f6 = cache.plan_match(prefix_f6, q36::detail::prefix_hash_chain(prefix_f6));
    if (!match || match->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        match->reuse_base != 4) {
        std::cerr << "four-token prefix did not select ladder head F=4 (shorter than rewrite F=6)\n";
        text.release();
        return 1;
    }
    if (!match_f2 || match_f2->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        match_f2->reuse_base != 2 || match_f2->entry_id != match->entry_id) {
        std::cerr << "shortest ladder prefix did not match head F=2\n";
        text.release();
        return 1;
    }
    if (!match_f6 || match_f6->reuse != ninfer::PrefixReusePath::RestoreTurnCheckpoint ||
        match_f6->reuse_base != 6 || match_f6->entry_id != match->entry_id) {
        std::cerr << "six-token prefix did not keep rewrite on a same-frontier tie with ladder F=6\n";
        text.release();
        return 1;
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_out_buf(128);
    rewrite_out_buf.fill(0xee);
    ninfer::Tensor rewrite_out(rewrite_out_buf.p, ninfer::DType::U8, {128});
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemset(gdn.conv_slot(layer, 2).data, 0, gdn.conv_slot(layer, 2).bytes()));
        CUDA_CHECK(
            cudaMemset(gdn.recurrent_slot(layer, 2).data, 0, gdn.recurrent_slot(layer, 2).bytes()));
        CUDA_CHECK(cudaMemset(gdn.conv_slot(layer, 3).data, 0xee, gdn.conv_slot(layer, 3).bytes()));
        CUDA_CHECK(cudaMemset(gdn.recurrent_slot(layer, 3).data, 0xee,
                              gdn.recurrent_slot(layer, 3).bytes()));
    }

    q36::detail::RamRestoreTarget target;
    target.text                    = &text_dest;
    target.text_pool               = &text_pool;
    target.text_dst_pages          = ninfer::pages_for_tokens(match->reuse_base);
    target.gdn                     = &gdn;
    target.gdn_current_slot        = 2;
    target.gdn_checkpoint_slot     = 3;
    target.tail_hidden             = &hidden_out;
    target.rewrite_checkpoint_hidden = &rewrite_out;
    target.reuse                   = match->reuse;
    target.reuse_base              = match->reuse_base;
    target.stream                  = ctx.copy_stream;
    cache.claim(match->entry_id);
    const q36::detail::RamRestoredHost host = cache.unpack_device(match->entry_id, target);
    ctx.synchronize_all();

    int failures = 0;
    if (host.ladders.size() != 3) {
        std::cerr << "RAM image dropped ladder heads instead of keeping F and neighbors\n";
        ++failures;
    } else {
        const bool keep_eq =
            host.ladders[0].frontier == 2 && host.ladders[1].frontier == 4 &&
            host.ladders[2].frontier == 6;
        if (!keep_eq) {
            std::cerr << "RAM image ladder frontiers are not 2/4/6\n";
            ++failures;
        }
    }

    std::vector<unsigned char> conv_packed(gdn.conv_host_image_bytes(), 0);
    std::vector<unsigned char> rec_packed(gdn.recurrent_host_image_bytes(), 0);
    gdn.pack_slot_to_host(2, conv_packed.data(), rec_packed.data(), ctx.copy_stream);
    ctx.synchronize_all();
    if (conv_packed != conv_f4 || rec_packed != rec_f4) {
        std::cerr << "middle-head unpack did not install that head's GDN into current\n";
        ++failures;
    }
    gdn.pack_slot_to_host(3, conv_packed.data(), rec_packed.data(), ctx.copy_stream);
    ctx.synchronize_all();
    if (conv_packed != std::vector<unsigned char>(conv_packed.size(), 0xee) ||
        rec_packed != std::vector<unsigned char>(rec_packed.size(), 0xee)) {
        std::cerr << "middle-head unpack loaded rewrite GDN sitting ahead of F\n";
        ++failures;
    }
    std::vector<unsigned char> hidden_host(128);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != hid_f4) {
        std::cerr << "middle-head unpack did not install that head's hidden into tail_hidden\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), rewrite_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != std::vector<unsigned char>(128, 0xee)) {
        std::cerr << "middle-head unpack wrote rewrite_checkpoint_hidden\n";
        ++failures;
    }

    auto unpack_head = [&](std::uint32_t frontier, const std::vector<unsigned char>& conv_expect,
                           const std::vector<unsigned char>& rec_expect,
                           const std::vector<unsigned char>& hid_expect, bool expect_rewrite) {
        CUDA_CHECK(cudaMemset(hidden_out.data, 0, hidden_out.bytes()));
        CUDA_CHECK(cudaMemset(rewrite_out.data, 0xee, rewrite_out.bytes()));
        for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
            CUDA_CHECK(cudaMemset(gdn.conv_slot(layer, 2).data, 0, gdn.conv_slot(layer, 2).bytes()));
            CUDA_CHECK(cudaMemset(gdn.recurrent_slot(layer, 2).data, 0,
                                  gdn.recurrent_slot(layer, 2).bytes()));
            CUDA_CHECK(
                cudaMemset(gdn.conv_slot(layer, 3).data, 0xee, gdn.conv_slot(layer, 3).bytes()));
            CUDA_CHECK(cudaMemset(gdn.recurrent_slot(layer, 3).data, 0xee,
                                  gdn.recurrent_slot(layer, 3).bytes()));
        }
        target.reuse_base     = frontier;
        target.text_dst_pages = ninfer::pages_for_tokens(frontier);
        (void)cache.unpack_device(match->entry_id, target);
        ctx.synchronize_all();
        gdn.pack_slot_to_host(2, conv_packed.data(), rec_packed.data(), ctx.copy_stream);
        ctx.synchronize_all();
        if (conv_packed != conv_expect || rec_packed != rec_expect) {
            std::cerr << "unpack of ladder F=" << frontier << " installed the wrong GDN\n";
            ++failures;
        }
        CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                               cudaMemcpyDeviceToHost));
        if (hidden_host != hid_expect) {
            std::cerr << "unpack of ladder F=" << frontier << " installed the wrong hidden\n";
            ++failures;
        }
        gdn.pack_slot_to_host(3, conv_packed.data(), rec_packed.data(), ctx.copy_stream);
        ctx.synchronize_all();
        const std::vector<unsigned char> rewrite_conv_fill(conv_packed.size(), 0xee);
        const std::vector<unsigned char> rewrite_rec_fill(rec_packed.size(), 0xee);
        if (expect_rewrite) {
            if (conv_packed != conv_rewrite || rec_packed != rec_rewrite) {
                std::cerr << "unpack of ladder F=" << frontier
                          << " did not restore rewrite GDN at or before F\n";
                ++failures;
            }
            CUDA_CHECK(cudaMemcpy(hidden_host.data(), rewrite_out.data, hidden_host.size(),
                                   cudaMemcpyDeviceToHost));
            if (hidden_host != std::vector<unsigned char>(128, 0xa2)) {
                std::cerr << "unpack of ladder F=" << frontier
                          << " did not restore rewrite hidden at or before F\n";
                ++failures;
            }
        } else if (conv_packed != rewrite_conv_fill || rec_packed != rewrite_rec_fill) {
            std::cerr << "unpack of ladder F=" << frontier
                      << " loaded rewrite GDN sitting ahead of F\n";
            ++failures;
        }
    };
    unpack_head(2, conv_f2, rec_f2, hid_f2, false);
    unpack_head(6, conv_f6, rec_f6, hid_f6, true);
    cache.consume(match->entry_id);

    text.release();
    text_dest.release();
    return failures;
}

int test_context_checkpoint_two_ram_entries(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(6, 6, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 2,
                      .value_head_dim = 2,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text_a = text_pool.reserve(2);
    auto text_b = text_pool.reserve(2);
    text_a.materialize_pages(2, ctx.stream);
    text_b.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text_a, 1);
    fill_logical_pages(text_pool, text_b, 2);

    const auto make_retained = [](std::vector<ninfer::TokenId> tokens) {
        auto prompt = text_prompt(std::move(tokens));
        q36::PreparedPromptData retained = prompt;
        retained.token_ids.push_back(0);
        retained.token_types.push_back(0);
        const std::size_t n = retained.token_ids.size();
        retained.positions.resize(3 * n);
        for (int axis = 0; axis < 3; ++axis) {
            for (std::size_t i = 0; i < n; ++i) {
                retained.positions[static_cast<std::size_t>(axis) * n + i] =
                    static_cast<std::int32_t>(i);
            }
        }
        return retained;
    };
    q36::PreparedPromptData retained_a = make_retained({1, 2, 3, 4});
    q36::PreparedPromptData retained_b = make_retained({9, 8, 7, 6});
    q36::detail::ResidentPrefixIdentity identity_a;
    q36::detail::ResidentPrefixIdentity identity_b;
    identity_a.assign(retained_a);
    identity_b.assign(retained_b);

    std::vector<unsigned char> conv_a(gdn.conv_host_image_bytes(), 0xa1);
    std::vector<unsigned char> rec_a(gdn.recurrent_host_image_bytes(), 0xa2);
    std::vector<unsigned char> hid_a(32, 0xa3);
    std::vector<unsigned char> conv_b(gdn.conv_host_image_bytes(), 0xb1);
    std::vector<unsigned char> rec_b(gdn.recurrent_host_image_bytes(), 0xb2);
    std::vector<unsigned char> hid_b(32, 0xb3);
    const auto head = [](std::uint32_t frontier, q36::detail::PrefixHash128 hash, const void* conv,
                         const void* rec, const void* hid, std::size_t conv_n, std::size_t rec_n,
                         std::size_t hid_n) {
        q36::detail::RamLadderHead out;
        out.frontier        = frontier;
        out.hash            = hash;
        out.conv            = conv;
        out.recurrent       = rec;
        out.hidden          = hid;
        out.conv_bytes      = conv_n;
        out.recurrent_bytes = rec_n;
        out.hidden_bytes    = hid_n;
        return out;
    };

    ninfer::DeviceBuffer hidden_a_buf(32);
    hidden_a_buf.fill(0xa3);
    ninfer::Tensor hidden_a(hidden_a_buf.p, ninfer::DType::U8, {32});
    ninfer::DeviceBuffer hidden_b_buf(32);
    hidden_b_buf.fill(0xb3);
    ninfer::Tensor hidden_b(hidden_b_buf.p, ninfer::DType::U8, {32});

    auto capture = [&](q36::PreparedPromptData& retained, q36::detail::ResidentPrefixIdentity& id,
                       ninfer::PagedKVAllocation& text, ninfer::Tensor& hidden,
                       std::vector<q36::detail::RamLadderHead> ladders) {
        q36::detail::RamCaptureSource source;
        source.execution_frontier = static_cast<std::uint32_t>(retained.token_ids.size() - 1);
        source.ledger_frontier    = static_cast<std::uint32_t>(retained.token_ids.size());
        source.text_kv_valid      = source.execution_frontier;
        source.tail_hidden_valid  = true;
        source.ledger             = retained.token_ids;
        source.identity           = &id;
        source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, id,
                                                                source.execution_frontier);
        source.text               = &text;
        source.text_pool          = &text_pool;
        source.gdn                = &gdn;
        source.gdn_current_slot   = 0;
        source.tail_hidden        = &hidden;
        source.ladder_heads       = std::move(ladders);
        source.stream             = ctx.copy_stream;
        return source;
    };

    q36::detail::KVRamCache cache(16ULL << 20);
    auto source_a = capture(retained_a, identity_a, text_a, hidden_a,
                            {head(2, q36::detail::prefix_hash_at(retained_a.token_ids, identity_a, 2),
                                  conv_a.data(), rec_a.data(), hid_a.data(), conv_a.size(),
                                  rec_a.size(), hid_a.size())});
    if (!capture_or_evict(cache, source_a)) {
        text_a.release();
        text_b.release();
        return fail("two-entry capture A failed");
    }
    auto source_b = capture(retained_b, identity_b, text_b, hidden_b,
                            {head(2, q36::detail::prefix_hash_at(retained_b.token_ids, identity_b, 2),
                                  conv_b.data(), rec_b.data(), hid_b.data(), conv_b.size(),
                                  rec_b.size(), hid_b.size())});
    if (!capture_or_evict(cache, source_b)) {
        text_a.release();
        text_b.release();
        return fail("two-entry capture B failed");
    }
    ctx.synchronize_all();

    const auto prefix_a = text_prompt({1, 2});
    const auto prefix_b = text_prompt({9, 8});
    const auto match_a  = cache.plan_match(prefix_a, q36::detail::prefix_hash_chain(prefix_a));
    const auto match_b  = cache.plan_match(prefix_b, q36::detail::prefix_hash_chain(prefix_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id ||
        match_a->reuse_base != 2 || match_b->reuse_base != 2 ||
        match_a->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        match_b->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        text_a.release();
        text_b.release();
        std::cerr << "two RAM entries did not match as distinct ladder heads\n";
        return 1;
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(32);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {32});
    std::vector<unsigned char> conv_packed(gdn.conv_host_image_bytes(), 0);
    std::vector<unsigned char> rec_packed(gdn.recurrent_host_image_bytes(), 0);
    std::vector<unsigned char> hidden_host(32, 0);

    auto unpack_entry = [&](const q36::detail::RamMatch& match,
                            const std::vector<unsigned char>& conv_expect,
                            const std::vector<unsigned char>& rec_expect,
                            const std::vector<unsigned char>& hid_expect, const char* label) {
        for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
            CUDA_CHECK(cudaMemset(gdn.conv_slot(layer, 2).data, 0, gdn.conv_slot(layer, 2).bytes()));
            CUDA_CHECK(cudaMemset(gdn.recurrent_slot(layer, 2).data, 0,
                                  gdn.recurrent_slot(layer, 2).bytes()));
        }
        CUDA_CHECK(cudaMemset(hidden_out.data, 0, hidden_out.bytes()));
        q36::detail::RamRestoreTarget target;
        target.text             = &text_dest;
        target.text_pool        = &text_pool;
        target.text_dst_pages   = ninfer::pages_for_tokens(match.reuse_base);
        target.gdn              = &gdn;
        target.gdn_current_slot = 2;
        target.tail_hidden      = &hidden_out;
        target.reuse            = match.reuse;
        target.reuse_base       = match.reuse_base;
        target.stream           = ctx.copy_stream;
        cache.claim(match.entry_id);
        (void)cache.unpack_device(match.entry_id, target);
        cache.consume(match.entry_id);
        ctx.synchronize_all();
        gdn.pack_slot_to_host(2, conv_packed.data(), rec_packed.data(), ctx.copy_stream);
        ctx.synchronize_all();
        int local = 0;
        if (conv_packed != conv_expect || rec_packed != rec_expect) {
            std::cerr << label << " unpacked the other entry's GDN\n";
            local = 1;
        }
        CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                               cudaMemcpyDeviceToHost));
        if (hidden_host != hid_expect) {
            std::cerr << label << " unpacked the other entry's hidden\n";
            local = 1;
        }
        return local;
    };

    int failures = 0;
    failures += unpack_entry(*match_a, conv_a, rec_a, hid_a, "entry A");
    failures += unpack_entry(*match_b, conv_b, rec_b, hid_b, "entry B");
    text_a.release();
    text_b.release();
    text_dest.release();
    return failures;
}

int test_context_checkpoint_ladder_beats_rewrite(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 5);

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8, 9});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    std::vector<unsigned char> conv(64, 0x11);
    std::vector<unsigned char> rec(64, 0x22);
    std::vector<unsigned char> hid(32, 0x33);
    q36::detail::RamLadderHead ladder;
    ladder.frontier        = 8;
    ladder.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 8);
    ladder.conv            = conv.data();
    ladder.recurrent       = rec.data();
    ladder.hidden          = hid.data();
    ladder.conv_bytes      = conv.size();
    ladder.recurrent_bytes = rec.size();
    ladder.hidden_bytes    = hid.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.rewrite_valid      = true;
    source.rewrite_kind       = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier   = 4;
    source.hash_c_valid       = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.hash_c             = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.ladder_heads       = {ladder};
    source.stream             = ctx.copy_stream;
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("ladder-beats-rewrite capture failed");
    }

    const auto prefix = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    const auto match  = cache.plan_match(prefix, q36::detail::prefix_hash_chain(prefix));
    text.release();
    if (!match || match->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        match->reuse_base != 8) {
        std::cerr << "longer ladder head did not beat shorter rewrite in plan_match\n";
        return 1;
    }
    return 0;
}

int test_context_checkpoint_equal_execution_is_append(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 5);

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    std::vector<unsigned char> conv(64, 0x11);
    std::vector<unsigned char> rec(64, 0x22);
    std::vector<unsigned char> hid(32, 0x33);
    q36::detail::RamLadderHead ladder;
    ladder.frontier        = 8;
    ladder.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 8);
    ladder.conv            = conv.data();
    ladder.recurrent       = rec.data();
    ladder.hidden          = hid.data();
    ladder.conv_bytes      = conv.size();
    ladder.recurrent_bytes = rec.size();
    ladder.hidden_bytes    = hid.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.rewrite_valid      = true;
    source.rewrite_kind       = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier   = 4;
    source.hash_c_valid       = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.hash_c             = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.ladder_heads       = {ladder};
    source.stream             = ctx.copy_stream;
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("equal-execution capture failed");
    }

    const auto prefix = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    const auto match  = cache.plan_match(prefix, q36::detail::prefix_hash_chain(prefix));
    text.release();
    if (!match || match->reuse != ninfer::PrefixReusePath::AppendAtFrontier ||
        match->reuse_base != 8) {
        std::cerr << "E==F RAM match did not keep AppendAtFrontier over the ladder head\n";
        return 1;
    }
    return 0;
}

int test_context_checkpoint_hash_mismatch(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    std::vector<unsigned char> conv(64, 0x11);
    std::vector<unsigned char> rec(64, 0x22);
    std::vector<unsigned char> hid(32, 0x33);
    q36::detail::RamLadderHead ladder;
    ladder.frontier        = 4;
    ladder.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    ladder.conv            = conv.data();
    ladder.recurrent       = rec.data();
    ladder.hidden          = hid.data();
    ladder.conv_bytes      = conv.size();
    ladder.recurrent_bytes = rec.size();
    ladder.hidden_bytes    = hid.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.ladder_heads       = {ladder};
    source.stream             = ctx.copy_stream;
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("hash-mismatch capture failed");
    }

    auto mutated = text_prompt({1, 2, 3, 99});
    const auto miss = cache.plan_match(mutated, q36::detail::prefix_hash_chain(mutated));
    text.release();
    if (miss) {
        std::cerr << "hash mismatch at ladder F still produced a RAM hit\n";
        return 1;
    }
    return 0;
}

int test_context_checkpoint_rollback_recapture(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 9);

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    std::vector<unsigned char> conv2(64, 0x21), rec2(64, 0x22), hid2(32, 0xb2);
    std::vector<unsigned char> conv4(64, 0x41), rec4(64, 0x42), hid4(32, 0xb4);
    std::vector<unsigned char> conv6(64, 0x61), rec6(64, 0x62), hid6(32, 0xb6);
    const auto head_at = [&](std::uint32_t frontier, const void* conv, const void* rec,
                             const void* hid) {
        q36::detail::RamLadderHead head;
        head.frontier        = frontier;
        head.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, frontier);
        head.conv            = conv;
        head.recurrent       = rec;
        head.hidden          = hid;
        head.conv_bytes      = conv4.size();
        head.recurrent_bytes = rec4.size();
        head.hidden_bytes    = hid4.size();
        return head;
    };

    auto fill_source = [&](std::vector<q36::detail::RamLadderHead> heads) {
        q36::detail::RamCaptureSource source;
        source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
        source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
        source.text_kv_valid      = source.execution_frontier;
        source.tail_hidden_valid  = false;
        source.ledger             = retained.token_ids;
        source.identity           = &identity;
        source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                                source.execution_frontier);
        source.text               = &text;
        source.text_pool          = &text_pool;
        source.ladder_heads       = std::move(heads);
        source.stream             = ctx.copy_stream;
        return source;
    };

    q36::detail::KVRamCache cache(16ULL << 20);
    auto fat = fill_source({head_at(2, conv2.data(), rec2.data(), hid2.data()),
                            head_at(4, conv4.data(), rec4.data(), hid4.data()),
                            head_at(6, conv6.data(), rec6.data(), hid6.data())});
    if (!capture_or_evict(cache, fat)) {
        text.release();
        return fail("rollback fat capture failed");
    }
    const auto prefix6 = text_prompt({1, 2, 3, 4, 5, 6});
    const auto hit6    = cache.plan_match(prefix6, q36::detail::prefix_hash_chain(prefix6));
    if (!hit6 || hit6->reuse_base != 6) {
        text.release();
        return fail("fat RAM image did not index all three GDN heads");
    }
    const auto prefix4 = text_prompt({1, 2, 3, 4});
    const auto hit4    = cache.plan_match(prefix4, q36::detail::prefix_hash_chain(prefix4));
    if (!hit4 || hit4->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        hit4->reuse_base != 4) {
        text.release();
        return fail("fat RAM image did not independently index the middle GDN head");
    }
    cache.claim(hit4->entry_id);
    cache.consume(hit4->entry_id);

    auto trimmed = fill_source({head_at(2, conv2.data(), rec2.data(), hid2.data()),
                                head_at(4, conv4.data(), rec4.data(), hid4.data())});
    if (!capture_or_evict(cache, trimmed)) {
        text.release();
        return fail("rollback recapture of remaining heads failed");
    }
    const auto after = cache.plan_match(prefix6, q36::detail::prefix_hash_chain(prefix6));
    text.release();
    if (!after || after->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        after->reuse_base != 4) {
        std::cerr << "recapture after rollback still served an infeasible F=6 GDN head\n";
        return 1;
    }
    const auto still4 = cache.plan_match(prefix4, q36::detail::prefix_hash_chain(prefix4));
    if (!still4 || still4->reuse_base != 4) {
        std::cerr << "recapture after rollback dropped the still-hittable F=4 GDN head\n";
        return 1;
    }
    return 0;
}

int test_context_checkpoint_consume_waits_copies(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 9);

    constexpr std::size_t kBulkBytes = 32ULL << 20;
    ninfer::DeviceBuffer bulk(kBulkBytes);
    bulk.fill(0x5a);
    ninfer::Tensor hidden(bulk.p, ninfer::DType::U8, {static_cast<std::int32_t>(kBulkBytes)});

    const auto prompt = text_prompt({1, 2, 3, 4});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    std::vector<unsigned char> conv(64, 0x21);
    std::vector<unsigned char> rec(64, 0x22);
    std::vector<unsigned char> hid(32, 0x23);
    const auto head_at = [&](std::uint32_t frontier) {
        q36::detail::RamLadderHead head;
        head.frontier        = frontier;
        head.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, frontier);
        head.conv            = conv.data();
        head.recurrent       = rec.data();
        head.hidden          = hid.data();
        head.conv_bytes      = conv.size();
        head.recurrent_bytes = rec.size();
        head.hidden_bytes    = hid.size();
        return head;
    };

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.tail_hidden        = &hidden;
    source.ladder_heads       = {head_at(2), head_at(4)};
    source.stream             = ctx.copy_stream;
    q36::detail::KVRamCache cache(64ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("consume-wait capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        text.release();
        return fail("consume-wait capture did not index");
    }

    cache.claim(match->entry_id);
    cache.consume(match->entry_id);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    if (!cache.copies_ready(match->entry_id)) {
        text.release();
        return fail("consume did not wait copies_done before freeing the ladder image");
    }
    text.release();
    return 0;
}

int test_context_checkpoint_catch_up_frontier(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 2,
                      .value_head_dim = 2,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 3);

    std::vector<unsigned char> conv_live(gdn.conv_host_image_bytes(), 0xe1);
    std::vector<unsigned char> rec_live(gdn.recurrent_host_image_bytes(), 0xe2);
    std::vector<unsigned char> conv_head(gdn.conv_host_image_bytes(), 0x51);
    std::vector<unsigned char> rec_head(gdn.recurrent_host_image_bytes(), 0x52);
    std::vector<unsigned char> hid_head(32, 0x53);
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemcpy(gdn.conv_slot(layer, 0).data, conv_live.data(), gdn.conv_slot_bytes(),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gdn.recurrent_slot(layer, 0).data, rec_live.data(),
                               gdn.recurrent_slot_bytes(), cudaMemcpyHostToDevice));
    }

    ninfer::DeviceBuffer hidden_buf(32);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {32});

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    q36::detail::RamLadderHead ladder;
    ladder.frontier        = 5;
    ladder.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 5);
    ladder.conv            = conv_head.data();
    ladder.recurrent       = rec_head.data();
    ladder.hidden          = hid_head.data();
    ladder.conv_bytes      = conv_head.size();
    ladder.recurrent_bytes = rec_head.size();
    ladder.hidden_bytes    = hid_head.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.mtp_kv_valid       = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.gdn                = &gdn;
    source.gdn_current_slot   = 0;
    source.tail_hidden        = &hidden;
    source.ladder_heads       = {ladder};
    source.stream             = ctx.copy_stream;
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("catch-up frontier capture failed");
    }

    const auto prefix = text_prompt({1, 2, 3, 4, 5});
    const auto match  = cache.plan_match(prefix, q36::detail::prefix_hash_chain(prefix));
    if (!match || match->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        match->reuse_base != 5) {
        text.release();
        std::cerr << "off-grid ladder F=5 did not match\n";
        return 1;
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(32);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {32});
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemset(gdn.conv_slot(layer, 2).data, 0, gdn.conv_slot(layer, 2).bytes()));
        CUDA_CHECK(
            cudaMemset(gdn.recurrent_slot(layer, 2).data, 0, gdn.recurrent_slot(layer, 2).bytes()));
    }
    q36::detail::RamRestoreTarget target;
    target.text           = &text_dest;
    target.text_pool      = &text_pool;
    target.text_dst_pages = ninfer::pages_for_tokens(match->reuse_base);
    target.gdn            = &gdn;
    target.gdn_current_slot = 2;
    target.tail_hidden    = &hidden_out;
    target.reuse          = match->reuse;
    target.reuse_base     = match->reuse_base;
    target.stream         = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    ctx.synchronize_all();

    std::vector<unsigned char> conv_packed(gdn.conv_host_image_bytes(), 0);
    std::vector<unsigned char> rec_packed(gdn.recurrent_host_image_bytes(), 0);
    gdn.pack_slot_to_host(2, conv_packed.data(), rec_packed.data(), ctx.copy_stream);
    ctx.synchronize_all();
    int failures = 0;
    if (conv_packed != conv_head || rec_packed != rec_head) {
        std::cerr << "catch-up unpack installed dump current GDN instead of the F=5 head\n";
        ++failures;
    }
    std::vector<unsigned char> hidden_host(32);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != hid_head) {
        std::cerr << "catch-up unpack installed the wrong hidden\n";
        ++failures;
    }
    cache.consume(match->entry_id);
    text.release();
    text_dest.release();
    return failures;
}

int test_turn_rollback_kind_roundtrip(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 3,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 7);

    std::vector<unsigned char> conv_exec(gdn.conv_host_image_bytes(), 0xaa);
    std::vector<unsigned char> rec_exec(gdn.recurrent_host_image_bytes(), 0xab);
    std::vector<unsigned char> conv_head(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec_head(gdn.recurrent_host_image_bytes(), 0x42);
    std::vector<unsigned char> hid_head(128, 0x43);
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemcpy(gdn.conv_slot(layer, 0).data, conv_exec.data(), gdn.conv_slot_bytes(),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gdn.recurrent_slot(layer, 0).data, rec_exec.data(),
                               gdn.recurrent_slot_bytes(), cudaMemcpyHostToDevice));
    }
    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xaa);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 4;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv_head.data();
    rollback.recurrent       = rec_head.data();
    rollback.hidden          = hid_head.data();
    rollback.conv_bytes      = conv_head.size();
    rollback.recurrent_bytes = rec_head.size();
    rollback.hidden_bytes    = hid_head.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.mtp_kv_valid       = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.rewrite_valid      = true;
    source.rewrite_kind       = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier   = 2;
    source.hash_c_valid       = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.hash_c             = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.gdn                = &gdn;
    source.gdn_current_slot   = 0;
    source.gdn_checkpoint_slot = 1;
    source.tail_hidden        = &hidden;
    source.ladder_heads       = {rollback};
    source.stream             = ctx.copy_stream;
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("turn-rollback kind capture failed");
    }

    const auto prefix = text_prompt({1, 2, 3, 4});
    const auto match  = cache.plan_match(prefix, q36::detail::prefix_hash_chain(prefix));
    if (!match || match->reuse != ninfer::PrefixReusePath::RestoreTurnRollback ||
        match->reuse_base != 4) {
        text.release();
        std::cerr << "turn-rollback RAM head did not report RestoreTurnRollback\n";
        return 1;
    }
    const q36::detail::RamRestoredHost loaded = cache.load_host(match->entry_id);
    if (loaded.ladders.size() != 1 ||
        loaded.ladders[0].kind != q36::detail::ContextCheckpointKind::TurnRollback) {
        text.release();
        std::cerr << "RAM image dropped turn_rollback kind\n";
        return 1;
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemset(gdn.conv_slot(layer, 2).data, 0xee, gdn.conv_slot(layer, 2).bytes()));
        CUDA_CHECK(
            cudaMemset(gdn.recurrent_slot(layer, 2).data, 0xee, gdn.recurrent_slot(layer, 2).bytes()));
    }
    q36::detail::RamRestoreTarget target;
    target.text             = &text_dest;
    target.text_pool        = &text_pool;
    target.text_dst_pages   = ninfer::pages_for_tokens(match->reuse_base);
    target.gdn              = &gdn;
    target.gdn_current_slot = 2;
    target.gdn_checkpoint_slot = 3;
    target.tail_hidden      = &hidden_out;
    target.reuse            = match->reuse;
    target.reuse_base       = match->reuse_base;
    target.stream           = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    ctx.synchronize_all();

    std::vector<unsigned char> conv_packed(gdn.conv_host_image_bytes(), 0);
    std::vector<unsigned char> rec_packed(gdn.recurrent_host_image_bytes(), 0);
    gdn.pack_slot_to_host(2, conv_packed.data(), rec_packed.data(), ctx.copy_stream);
    ctx.synchronize_all();
    int failures = 0;
    if (conv_packed != conv_head || rec_packed != rec_head) {
        std::cerr << "RestoreTurnRollback unpacked eviction current GDN instead of the head\n";
        ++failures;
    }
    std::vector<unsigned char> hidden_host(128);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != hid_head) {
        std::cerr << "RestoreTurnRollback unpacked eviction hidden instead of the head\n";
        ++failures;
    }
    text.release();
    text_dest.release();
    return failures;
}

int test_mixed_checkpoint_gdn_isolation(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 3,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 7);

    std::vector<unsigned char> conv_cur(gdn.conv_host_image_bytes(), 0xaa);
    std::vector<unsigned char> rec_cur(gdn.recurrent_host_image_bytes(), 0xab);
    std::vector<unsigned char> conv_rw(gdn.conv_host_image_bytes(), 0xbb);
    std::vector<unsigned char> rec_rw(gdn.recurrent_host_image_bytes(), 0xbc);
    std::vector<unsigned char> conv_rb(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec_rb(gdn.recurrent_host_image_bytes(), 0x42);
    std::vector<unsigned char> hid_rb(128, 0x43);
    std::vector<unsigned char> conv_ld(gdn.conv_host_image_bytes(), 0x51);
    std::vector<unsigned char> rec_ld(gdn.recurrent_host_image_bytes(), 0x52);
    std::vector<unsigned char> hid_ld(128, 0x53);
    gdn.unpack_slot_from_host(0, conv_cur.data(), rec_cur.data(), ctx.stream);
    gdn.unpack_slot_from_host(1, conv_rw.data(), rec_rw.data(), ctx.stream);
    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_buf(128);
    rewrite_buf.fill(0xa2);
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 4;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv_rb.data();
    rollback.recurrent       = rec_rb.data();
    rollback.hidden          = hid_rb.data();
    rollback.conv_bytes      = conv_rb.size();
    rollback.recurrent_bytes = rec_rb.size();
    rollback.hidden_bytes    = hid_rb.size();
    q36::detail::RamLadderHead ladder;
    ladder.frontier        = 6;
    ladder.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 6);
    ladder.kind            = q36::detail::ContextCheckpointKind::Ladder;
    ladder.conv            = conv_ld.data();
    ladder.recurrent       = rec_ld.data();
    ladder.hidden          = hid_ld.data();
    ladder.conv_bytes      = conv_ld.size();
    ladder.recurrent_bytes = rec_ld.size();
    ladder.hidden_bytes    = hid_ld.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier      = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier         = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid           = source.execution_frontier;
    source.mtp_kv_valid            = source.execution_frontier;
    source.tail_hidden_valid       = true;
    source.rewrite_valid           = true;
    source.rewrite_kind            = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier        = 2;
    source.hash_c_valid            = true;
    source.ledger                  = retained.token_ids;
    source.identity                = &identity;
    source.hash_f                  = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                                source.execution_frontier);
    source.hash_c                  = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    source.text                    = &text;
    source.text_pool               = &text_pool;
    source.gdn                     = &gdn;
    source.gdn_current_slot        = 0;
    source.gdn_checkpoint_slot     = 1;
    source.tail_hidden             = &hidden;
    source.rewrite_checkpoint_hidden = &rewrite;
    source.ladder_heads            = {rollback, ladder};
    source.stream                  = ctx.copy_stream;
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("mixed checkpoint capture failed");
    }

    const auto prefix4 = text_prompt({1, 2, 3, 4});
    const auto prefix6 = text_prompt({1, 2, 3, 4, 5, 6});
    const auto prefix8 = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    const auto match4  = cache.plan_match(prefix4, q36::detail::prefix_hash_chain(prefix4));
    const auto match6  = cache.plan_match(prefix6, q36::detail::prefix_hash_chain(prefix6));
    const auto match8  = cache.plan_match(prefix8, q36::detail::prefix_hash_chain(prefix8));
    if (!match4 || match4->reuse != ninfer::PrefixReusePath::RestoreTurnRollback ||
        match4->reuse_base != 4) {
        text.release();
        std::cerr << "mixed image prefix4 did not select RestoreTurnRollback\n";
        return 1;
    }
    if (!match6 || match6->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        match6->reuse_base != 6 || match6->entry_id != match4->entry_id) {
        text.release();
        std::cerr << "mixed image prefix6 did not select the longer ladder head on the same entry\n";
        return 1;
    }
    if (!match8 || match8->reuse != ninfer::PrefixReusePath::AppendAtFrontier ||
        match8->reuse_base != 8 || match8->entry_id != match4->entry_id) {
        text.release();
        std::cerr << "fully paged chat did not select AppendAtFrontier at eviction E\n";
        return 1;
    }
    int failures = 0;
    const q36::detail::RamRestoredHost loaded = cache.load_host(match4->entry_id);
    if (loaded.ladders.size() != 2 || loaded.ladders[0].hash != rollback.hash ||
        loaded.ladders[1].hash != ladder.hash) {
        std::cerr << "RAM ledger hashes for mixed heads do not match prefix_hash_at\n";
        ++failures;
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    std::vector<unsigned char> conv_wipe(gdn.conv_host_image_bytes(), 0xee);
    std::vector<unsigned char> rec_wipe(gdn.recurrent_host_image_bytes(), 0xee);
    auto wipe_dest = [&] {
        hidden_out_buf.fill(0);
        gdn.unpack_slot_from_host(2, conv_wipe.data(), rec_wipe.data(), ctx.stream);
        gdn.unpack_slot_from_host(3, conv_wipe.data(), rec_wipe.data(), ctx.stream);
        ctx.synchronize_all();
    };
    auto packed_slot = [&](std::int32_t slot) {
        std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0);
        std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0);
        gdn.pack_slot_to_host(slot, conv.data(), rec.data(), ctx.copy_stream);
        ctx.synchronize_all();
        return std::pair(std::move(conv), std::move(rec));
    };

    wipe_dest();
    std::vector<unsigned char> conv_poison(gdn.conv_host_image_bytes(), 0x99);
    std::vector<unsigned char> rec_poison(gdn.recurrent_host_image_bytes(), 0x99);
    gdn.unpack_slot_from_host(0, conv_poison.data(), rec_poison.data(), ctx.stream);
    ctx.synchronize_all();
    q36::detail::RamRestoreTarget target;
    target.text                = &text_dest;
    target.text_pool           = &text_pool;
    target.text_dst_pages      = ninfer::pages_for_tokens(4);
    target.gdn                 = &gdn;
    target.gdn_current_slot    = 2;
    target.gdn_checkpoint_slot = 3;
    target.tail_hidden         = &hidden_out;
    target.reuse               = ninfer::PrefixReusePath::RestoreTurnRollback;
    target.reuse_base          = 4;
    target.stream              = ctx.copy_stream;
    cache.claim(match4->entry_id);
    (void)cache.unpack_device(match4->entry_id, target);
    ctx.synchronize_all();
    auto [conv_now, rec_now] = packed_slot(2);
    auto [conv_ckpt, rec_ckpt] = packed_slot(3);
    auto [conv_left, rec_left] = packed_slot(0);
    if (conv_now != conv_rb || rec_now != rec_rb) {
        std::cerr << "rollback unpack installed current/rewrite/ladder GDN into current\n";
        ++failures;
    }
    if (conv_left != conv_poison || rec_left != rec_poison) {
        std::cerr << "rollback unpack clobbered leftover current/2C GDN\n";
        ++failures;
    }
    if (conv_ckpt != conv_rw || rec_ckpt != rec_rw) {
        std::cerr << "rollback unpack dropped rewrite GDN that sits at F<=E\n";
        ++failures;
    }
    std::vector<unsigned char> hidden_host(128);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != hid_rb) {
        std::cerr << "rollback unpack installed eviction/ladder hidden\n";
        ++failures;
    }

    wipe_dest();
    target.reuse      = ninfer::PrefixReusePath::RestoreContextCheckpoint;
    target.reuse_base = 6;
    target.text_dst_pages = ninfer::pages_for_tokens(6);
    (void)cache.unpack_device(match4->entry_id, target);
    ctx.synchronize_all();
    auto [conv_ld_got, rec_ld_got] = packed_slot(2);
    if (conv_ld_got != conv_ld || rec_ld_got != rec_ld) {
        std::cerr << "ladder unpack installed current/rollback GDN instead of the F=6 head\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != hid_ld) {
        std::cerr << "ladder unpack installed the wrong hidden\n";
        ++failures;
    }

    bool kind_mismatch = false;
    try {
        target.reuse      = ninfer::PrefixReusePath::RestoreTurnRollback;
        target.reuse_base = 6;
        (void)cache.unpack_device(match4->entry_id, target);
    } catch (const std::logic_error&) { kind_mismatch = true; }
    if (!kind_mismatch) {
        std::cerr << "unpack accepted RestoreTurnRollback at a ladder frontier\n";
        ++failures;
    }

    wipe_dest();
    gdn.unpack_slot_from_host(0, conv_poison.data(), rec_poison.data(), ctx.stream);
    ctx.synchronize_all();
    target.reuse          = ninfer::PrefixReusePath::AppendAtFrontier;
    target.reuse_base     = 8;
    target.text_dst_pages = ninfer::pages_for_tokens(8);
    (void)cache.unpack_device(match4->entry_id, target);
    ctx.synchronize_all();
    auto [conv_e, rec_e]           = packed_slot(2);
    auto [conv_left_e, rec_left_e] = packed_slot(0);
    if (conv_e != conv_cur || rec_e != rec_cur) {
        std::cerr << "AppendAtFrontier unpacked a checkpoint GDN instead of eviction current\n";
        ++failures;
    }
    if (conv_left_e != conv_poison || rec_left_e != rec_poison) {
        std::cerr << "AppendAtFrontier clobbered leftover current/2C GDN\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    std::vector<unsigned char> hid_cur(128, 0xa1);
    if (hidden_host != hid_cur) {
        std::cerr << "AppendAtFrontier unpacked a checkpoint hidden instead of eviction hidden\n";
        ++failures;
    }

    cache.consume(match4->entry_id);
    text.release();
    text_dest.release();
    return failures;
}

int test_context_checkpoint_dflash_cyclic_isolation(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    ninfer::LayoutBuilder cyclic_builder;
    const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 16, 2, 8, 2);
    ninfer::DeviceArena cyclic_arena(cyclic_builder.finish(256));
    ninfer::CyclicKVCache dflash_local({cyclic_arena.base(), cyclic_arena.capacity()},
                                       cyclic_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 7);

    std::vector<unsigned char> conv_cur(gdn.conv_host_image_bytes(), 0xaa);
    std::vector<unsigned char> rec_cur(gdn.recurrent_host_image_bytes(), 0xab);
    std::vector<unsigned char> conv_ld(gdn.conv_host_image_bytes(), 0x51);
    std::vector<unsigned char> rec_ld(gdn.recurrent_host_image_bytes(), 0x52);
    std::vector<unsigned char> hid_ld(128, 0x53);
    gdn.unpack_slot_from_host(0, conv_cur.data(), rec_cur.data(), ctx.stream);

    ninfer::CyclicKVCacheLayerView local_view = dflash_local.layer_view(0);
    std::vector<unsigned char> k_evict(local_view.k.slice(3, 0, 1).bytes(), 0x3c);
    std::vector<unsigned char> v_evict(local_view.v.slice(3, 0, 1).bytes(), 0x3d);
    CUDA_CHECK(cudaMemcpy(local_view.k.slice(3, 0, 1).data, k_evict.data(), k_evict.size(),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(local_view.v.slice(3, 0, 1).data, v_evict.data(), v_evict.size(),
                           cudaMemcpyHostToDevice));
    std::vector<unsigned char> cyclic_head(dflash_local.lane_host_bytes(), 0xa5);

    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    q36::detail::RamLadderHead ladder;
    ladder.frontier        = 4;
    ladder.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    ladder.kind            = q36::detail::ContextCheckpointKind::Ladder;
    ladder.conv            = conv_ld.data();
    ladder.recurrent       = rec_ld.data();
    ladder.hidden          = hid_ld.data();
    ladder.dflash          = cyclic_head.data();
    ladder.conv_bytes      = conv_ld.size();
    ladder.recurrent_bytes = rec_ld.size();
    ladder.hidden_bytes    = hid_ld.size();
    ladder.dflash_bytes    = cyclic_head.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier     = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier        = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid          = source.execution_frontier;
    source.dflash_context_frontier = source.execution_frontier;
    source.tail_hidden_valid      = true;
    source.ledger                 = retained.token_ids;
    source.identity               = &identity;
    source.hash_f                 = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                               source.execution_frontier);
    source.text                   = &text;
    source.text_pool              = &text_pool;
    source.gdn                    = &gdn;
    source.gdn_current_slot       = 0;
    source.tail_hidden            = &hidden;
    source.ladder_heads           = {ladder};
    source.dflash_local           = &dflash_local;
    source.dflash_lane            = 0;
    source.stream                 = ctx.copy_stream;
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("DFlash cyclic checkpoint capture failed");
    }

    const auto prefix4 = text_prompt({1, 2, 3, 4});
    const auto match   = cache.plan_match(prefix4, q36::detail::prefix_hash_chain(prefix4));
    if (!match || match->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        match->reuse_base != 4) {
        text.release();
        return fail("DFlash cyclic checkpoint did not select ladder F=4");
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    std::vector<unsigned char> k_poison(local_view.k.slice(3, 1, 1).bytes(), 0x11);
    std::vector<unsigned char> v_poison(local_view.v.slice(3, 1, 1).bytes(), 0x12);
    CUDA_CHECK(cudaMemcpy(dflash_local.layer_view(0).k.slice(3, 1, 1).data, k_poison.data(),
                           k_poison.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dflash_local.layer_view(0).v.slice(3, 1, 1).data, v_poison.data(),
                           v_poison.size(), cudaMemcpyHostToDevice));
    ctx.synchronize_all();

    q36::detail::RamRestoreTarget target;
    target.text             = &text_dest;
    target.text_pool        = &text_pool;
    target.text_dst_pages   = ninfer::pages_for_tokens(4);
    target.gdn              = &gdn;
    target.gdn_current_slot = 2;
    target.tail_hidden      = &hidden_out;
    target.reuse            = ninfer::PrefixReusePath::RestoreContextCheckpoint;
    target.reuse_base       = 4;
    target.dflash_local     = &dflash_local;
    target.dflash_lane      = 1;
    target.stream           = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    ctx.synchronize_all();

    int failures = 0;
    std::vector<unsigned char> restored(dflash_local.lane_host_bytes(), 0);
    dflash_local.copy_lane_to_host(1, restored.data(), ctx.stream);
    ctx.synchronize_all();
    if (restored != cyclic_head) {
        std::cerr << "staged DFlash restore unpacked eviction cyclic instead of the head\n";
        ++failures;
    }
    std::vector<unsigned char> evict_k(k_evict.size(), 0);
    CUDA_CHECK(cudaMemcpy(evict_k.data(), dflash_local.layer_view(0).k.slice(3, 0, 1).data,
                           evict_k.size(), cudaMemcpyDeviceToHost));
    if (evict_k != k_evict) {
        std::cerr << "staged DFlash restore clobbered the eviction source lane\n";
        ++failures;
    }

    cache.consume(match->entry_id);
    text.release();
    text_dest.release();
    return failures;
}

int test_same_f_rewrite_beats_ram_rollback(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 5);

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    std::vector<unsigned char> conv(64, 0x41);
    std::vector<unsigned char> rec(64, 0x42);
    std::vector<unsigned char> hid(32, 0x43);
    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 4;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv.data();
    rollback.recurrent       = rec.data();
    rollback.hidden          = hid.data();
    rollback.conv_bytes      = conv.size();
    rollback.recurrent_bytes = rec.size();
    rollback.hidden_bytes    = hid.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.rewrite_valid      = true;
    source.rewrite_kind       = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier   = 4;
    source.hash_c_valid       = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.hash_c             = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.ladder_heads       = {rollback};
    source.stream             = ctx.copy_stream;
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("same-F rewrite vs rollback capture failed");
    }

    const auto prefix = text_prompt({1, 2, 3, 4});
    const auto match  = cache.plan_match(prefix, q36::detail::prefix_hash_chain(prefix));
    text.release();
    if (!match || match->reuse != ninfer::PrefixReusePath::RestoreTurnCheckpoint ||
        match->reuse_base != 4) {
        std::cerr << "same-F rewrite vs rollback selected "
                  << (match ? static_cast<int>(match->reuse) : -1) << " base="
                  << (match ? match->reuse_base : 0)
                  << ", expected RestoreTurnCheckpoint at 4\n";
        return 1;
    }
    return 0;
}

int test_rollback_skips_ahead_rewrite_gdn(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 3,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 7);

    std::vector<unsigned char> conv_cur(gdn.conv_host_image_bytes(), 0xaa);
    std::vector<unsigned char> rec_cur(gdn.recurrent_host_image_bytes(), 0xab);
    std::vector<unsigned char> conv_rw(gdn.conv_host_image_bytes(), 0xbb);
    std::vector<unsigned char> rec_rw(gdn.recurrent_host_image_bytes(), 0xbc);
    std::vector<unsigned char> conv_rb(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec_rb(gdn.recurrent_host_image_bytes(), 0x42);
    std::vector<unsigned char> hid_rb(128, 0x43);
    std::vector<unsigned char> conv_poison(gdn.conv_host_image_bytes(), 0x77);
    std::vector<unsigned char> rec_poison(gdn.recurrent_host_image_bytes(), 0x77);
    gdn.unpack_slot_from_host(0, conv_cur.data(), rec_cur.data(), ctx.stream);
    gdn.unpack_slot_from_host(1, conv_rw.data(), rec_rw.data(), ctx.stream);
    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_buf(128);
    rewrite_buf.fill(0xa2);
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 4;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv_rb.data();
    rollback.recurrent       = rec_rb.data();
    rollback.hidden          = hid_rb.data();
    rollback.conv_bytes      = conv_rb.size();
    rollback.recurrent_bytes = rec_rb.size();
    rollback.hidden_bytes    = hid_rb.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier        = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier           = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid             = source.execution_frontier;
    source.mtp_kv_valid              = source.execution_frontier;
    source.tail_hidden_valid         = true;
    source.rewrite_valid             = true;
    source.rewrite_kind              = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier          = 8;
    source.hash_c_valid              = true;
    source.ledger                    = retained.token_ids;
    source.identity                  = &identity;
    source.hash_f                    = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                                   source.execution_frontier);
    source.hash_c                    = q36::detail::prefix_hash_at(retained.token_ids, identity, 8);
    source.text                      = &text;
    source.text_pool                 = &text_pool;
    source.gdn                       = &gdn;
    source.gdn_current_slot          = 0;
    source.gdn_checkpoint_slot       = 1;
    source.tail_hidden               = &hidden;
    source.rewrite_checkpoint_hidden = &rewrite;
    source.ladder_heads              = {rollback};
    source.stream                    = ctx.copy_stream;
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("rewrite-ahead capture failed");
    }

    const auto prefix4 = text_prompt({1, 2, 3, 4});
    const auto match4  = cache.plan_match(prefix4, q36::detail::prefix_hash_chain(prefix4));
    if (!match4 || match4->reuse != ninfer::PrefixReusePath::RestoreTurnRollback ||
        match4->reuse_base != 4) {
        text.release();
        std::cerr << "rewrite-ahead prefix4 did not select RestoreTurnRollback\n";
        return 1;
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    gdn.unpack_slot_from_host(2, conv_poison.data(), rec_poison.data(), ctx.stream);
    gdn.unpack_slot_from_host(3, conv_poison.data(), rec_poison.data(), ctx.stream);
    ctx.synchronize_all();
    q36::detail::RamRestoreTarget target;
    target.text                = &text_dest;
    target.text_pool           = &text_pool;
    target.text_dst_pages      = ninfer::pages_for_tokens(4);
    target.gdn                 = &gdn;
    target.gdn_current_slot    = 2;
    target.gdn_checkpoint_slot = 3;
    target.tail_hidden         = &hidden_out;
    target.reuse               = ninfer::PrefixReusePath::RestoreTurnRollback;
    target.reuse_base          = 4;
    target.stream              = ctx.copy_stream;
    cache.claim(match4->entry_id);
    (void)cache.unpack_device(match4->entry_id, target);
    ctx.synchronize_all();

    auto packed_slot = [&](std::int32_t slot) {
        std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0);
        std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0);
        gdn.pack_slot_to_host(slot, conv.data(), rec.data(), ctx.copy_stream);
        ctx.synchronize_all();
        return std::pair(std::move(conv), std::move(rec));
    };
    int failures = 0;
    auto [conv_now, rec_now]   = packed_slot(2);
    auto [conv_ckpt, rec_ckpt] = packed_slot(3);
    if (conv_now != conv_rb || rec_now != rec_rb) {
        std::cerr << "rewrite-ahead rollback unpack missed the head GDN\n";
        ++failures;
    }
    if (conv_ckpt != conv_poison || rec_ckpt != rec_poison) {
        std::cerr << "rollback unpack installed rewrite GDN that sits ahead of E\n";
        ++failures;
    }
    cache.consume(match4->entry_id);
    text.release();
    text_dest.release();
    return failures;
}

int test_c2_lane1_rollback_slot_isolation(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 3,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 5,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 7);

    std::vector<unsigned char> conv_cur(gdn.conv_host_image_bytes(), 0xaa);
    std::vector<unsigned char> rec_cur(gdn.recurrent_host_image_bytes(), 0xab);
    std::vector<unsigned char> conv_rw(gdn.conv_host_image_bytes(), 0xbb);
    std::vector<unsigned char> rec_rw(gdn.recurrent_host_image_bytes(), 0xbc);
    std::vector<unsigned char> conv_rb(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec_rb(gdn.recurrent_host_image_bytes(), 0x42);
    std::vector<unsigned char> hid_rb(128, 0x43);
    std::vector<unsigned char> conv_p0(gdn.conv_host_image_bytes(), 0x70);
    std::vector<unsigned char> rec_p0(gdn.recurrent_host_image_bytes(), 0x70);
    std::vector<unsigned char> conv_p2(gdn.conv_host_image_bytes(), 0x72);
    std::vector<unsigned char> rec_p2(gdn.recurrent_host_image_bytes(), 0x72);
    std::vector<unsigned char> conv_p4(gdn.conv_host_image_bytes(), 0x74);
    std::vector<unsigned char> rec_p4(gdn.recurrent_host_image_bytes(), 0x74);
    gdn.unpack_slot_from_host(0, conv_cur.data(), rec_cur.data(), ctx.stream);
    gdn.unpack_slot_from_host(1, conv_rw.data(), rec_rw.data(), ctx.stream);
    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_buf(128);
    rewrite_buf.fill(0xa2);
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 4;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv_rb.data();
    rollback.recurrent       = rec_rb.data();
    rollback.hidden          = hid_rb.data();
    rollback.conv_bytes      = conv_rb.size();
    rollback.recurrent_bytes = rec_rb.size();
    rollback.hidden_bytes    = hid_rb.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier        = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier           = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid             = source.execution_frontier;
    source.mtp_kv_valid              = source.execution_frontier;
    source.tail_hidden_valid         = true;
    source.rewrite_valid             = true;
    source.rewrite_kind              = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier          = 2;
    source.hash_c_valid              = true;
    source.ledger                    = retained.token_ids;
    source.identity                  = &identity;
    source.hash_f                    = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                                   source.execution_frontier);
    source.hash_c                    = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    source.text                      = &text;
    source.text_pool                 = &text_pool;
    source.gdn                       = &gdn;
    source.gdn_current_slot          = 0;
    source.gdn_checkpoint_slot       = 1;
    source.tail_hidden               = &hidden;
    source.rewrite_checkpoint_hidden = &rewrite;
    source.ladder_heads              = {rollback};
    source.stream                    = ctx.copy_stream;
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("C=2 lane-1 capture failed");
    }

    const auto prefix4 = text_prompt({1, 2, 3, 4});
    const auto match4  = cache.plan_match(prefix4, q36::detail::prefix_hash_chain(prefix4));
    if (!match4 || match4->reuse != ninfer::PrefixReusePath::RestoreTurnRollback) {
        text.release();
        return fail("C=2 lane-1 capture did not match RestoreTurnRollback");
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    gdn.unpack_slot_from_host(0, conv_p0.data(), rec_p0.data(), ctx.stream);
    gdn.unpack_slot_from_host(1, conv_p0.data(), rec_p0.data(), ctx.stream);
    gdn.unpack_slot_from_host(2, conv_p2.data(), rec_p2.data(), ctx.stream);
    gdn.unpack_slot_from_host(3, conv_p2.data(), rec_p2.data(), ctx.stream);
    gdn.unpack_slot_from_host(4, conv_p4.data(), rec_p4.data(), ctx.stream);
    ctx.synchronize_all();
    q36::detail::RamRestoreTarget target;
    target.text                = &text_dest;
    target.text_pool           = &text_pool;
    target.text_dst_pages      = ninfer::pages_for_tokens(4);
    target.gdn                 = &gdn;
    target.gdn_current_slot    = 1;
    target.gdn_checkpoint_slot = 3;
    target.tail_hidden         = &hidden_out;
    target.reuse               = ninfer::PrefixReusePath::RestoreTurnRollback;
    target.reuse_base          = 4;
    target.stream              = ctx.copy_stream;
    cache.claim(match4->entry_id);
    (void)cache.unpack_device(match4->entry_id, target);
    ctx.synchronize_all();

    auto packed_slot = [&](std::int32_t slot) {
        std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0);
        std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0);
        gdn.pack_slot_to_host(slot, conv.data(), rec.data(), ctx.copy_stream);
        ctx.synchronize_all();
        return std::pair(std::move(conv), std::move(rec));
    };
    int failures = 0;
    auto [conv_l0, rec_l0] = packed_slot(0);
    auto [conv_l1, rec_l1] = packed_slot(1);
    auto [conv_r0, rec_r0] = packed_slot(2);
    auto [conv_r1, rec_r1] = packed_slot(3);
    auto [conv_st, rec_st] = packed_slot(4);
    if (conv_l1 != conv_rb || rec_l1 != rec_rb) {
        std::cerr << "C=2 lane-1 rollback did not land in slot 1\n";
        ++failures;
    }
    if (conv_r1 != conv_rw || rec_r1 != rec_rw) {
        std::cerr << "C=2 lane-1 rollback dropped rewrite GDN at F<=E from slot C+1\n";
        ++failures;
    }
    if (conv_l0 != conv_p0 || rec_l0 != rec_p0) {
        std::cerr << "C=2 lane-1 rollback clobbered lane 0 current GDN\n";
        ++failures;
    }
    if (conv_r0 != conv_p2 || rec_r0 != rec_p2) {
        std::cerr << "C=2 lane-1 rollback clobbered lane 0 rewrite GDN\n";
        ++failures;
    }
    if (conv_st != conv_p4 || rec_st != rec_p4) {
        std::cerr << "C=2 lane-1 rollback clobbered staging slot 2C\n";
        ++failures;
    }
    cache.consume(match4->entry_id);
    text.release();
    text_dest.release();
    return failures;
}

int test_rollback_head_gdn_geometry_mismatch(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 2,
                      .value_head_dim = 2,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 5);
    std::vector<unsigned char> conv_cur(gdn.conv_host_image_bytes(), 0xaa);
    std::vector<unsigned char> rec_cur(gdn.recurrent_host_image_bytes(), 0xab);
    gdn.unpack_slot_from_host(0, conv_cur.data(), rec_cur.data(), ctx.stream);
    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);

    std::vector<unsigned char> conv_short(1, 0x41);
    std::vector<unsigned char> rec_ok(gdn.recurrent_host_image_bytes(), 0x42);
    std::vector<unsigned char> hid(128, 0x43);
    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 4;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv_short.data();
    rollback.recurrent       = rec_ok.data();
    rollback.hidden          = hid.data();
    rollback.conv_bytes      = conv_short.size();
    rollback.recurrent_bytes = rec_ok.size();
    rollback.hidden_bytes    = hid.size();

    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.gdn                = &gdn;
    source.gdn_current_slot   = 0;
    source.tail_hidden        = &hidden;
    source.ladder_heads       = {rollback};
    source.stream             = ctx.copy_stream;
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::KVRamCache cache(16ULL << 20);
    if (!capture_or_evict(cache, source)) {
        text.release();
        return fail("short-head GDN capture failed");
    }

    const auto prefix4 = text_prompt({1, 2, 3, 4});
    const auto match4  = cache.plan_match(prefix4, q36::detail::prefix_hash_chain(prefix4));
    if (!match4) {
        text.release();
        return fail("short-head GDN capture did not match");
    }
    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    q36::detail::RamRestoreTarget target;
    target.text             = &text_dest;
    target.text_pool        = &text_pool;
    target.text_dst_pages   = ninfer::pages_for_tokens(4);
    target.gdn              = &gdn;
    target.gdn_current_slot = 2;
    target.tail_hidden      = &hidden_out;
    target.reuse            = ninfer::PrefixReusePath::RestoreTurnRollback;
    target.reuse_base       = 4;
    target.stream           = ctx.copy_stream;
    cache.claim(match4->entry_id);
    bool threw = false;
    try {
        (void)cache.unpack_device(match4->entry_id, target);
    } catch (const std::logic_error&) { threw = true; }
    cache.consume(match4->entry_id);
    text.release();
    text_dest.release();
    if (!threw) {
        std::cerr << "undersize rollback GDN image did not throw on unpack\n";
        return 1;
    }
    return 0;
}

int test_context_checkpoint_same_f_fifo_first_wins(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(6, 6, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 2,
                      .value_head_dim = 2,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    auto text_a = text_pool.reserve(2);
    auto text_b = text_pool.reserve(2);
    text_a.materialize_pages(2, ctx.stream);
    text_b.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text_a, 1);
    fill_logical_pages(text_pool, text_b, 2);

    auto make_retained = []() {
        auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
        q36::PreparedPromptData retained = prompt;
        retained.token_ids.push_back(0);
        retained.token_types.push_back(0);
        const std::size_t n = retained.token_ids.size();
        retained.positions.resize(3 * n);
        for (int axis = 0; axis < 3; ++axis) {
            for (std::size_t i = 0; i < n; ++i) {
                retained.positions[static_cast<std::size_t>(axis) * n + i] =
                    static_cast<std::int32_t>(i);
            }
        }
        return retained;
    };
    q36::PreparedPromptData retained_a = make_retained();
    q36::PreparedPromptData retained_b = make_retained();
    q36::detail::ResidentPrefixIdentity identity_a;
    q36::detail::ResidentPrefixIdentity identity_b;
    identity_a.assign(retained_a);
    identity_b.assign(retained_b);

    std::vector<unsigned char> conv_a(gdn.conv_host_image_bytes(), 0xa1);
    std::vector<unsigned char> rec_a(gdn.recurrent_host_image_bytes(), 0xa2);
    std::vector<unsigned char> hid_a(32, 0xa3);
    std::vector<unsigned char> conv_b(gdn.conv_host_image_bytes(), 0xb1);
    std::vector<unsigned char> rec_b(gdn.recurrent_host_image_bytes(), 0xb2);
    std::vector<unsigned char> hid_b(32, 0xb3);

    ninfer::DeviceBuffer hidden_a_buf(32);
    hidden_a_buf.fill(0xa3);
    ninfer::Tensor hidden_a(hidden_a_buf.p, ninfer::DType::U8, {32});
    ninfer::DeviceBuffer hidden_b_buf(32);
    hidden_b_buf.fill(0xb3);
    ninfer::Tensor hidden_b(hidden_b_buf.p, ninfer::DType::U8, {32});

    const auto capture = [&](q36::PreparedPromptData& retained,
                             q36::detail::ResidentPrefixIdentity& identity,
                             ninfer::PagedKVAllocation& text, ninfer::Tensor& hidden,
                             const std::vector<unsigned char>& conv,
                             const std::vector<unsigned char>& rec,
                             const std::vector<unsigned char>& hid) {
        q36::detail::RamLadderHead head;
        head.frontier        = 4;
        head.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
        head.conv            = conv.data();
        head.recurrent       = rec.data();
        head.hidden          = hid.data();
        head.conv_bytes      = conv.size();
        head.recurrent_bytes = rec.size();
        head.hidden_bytes    = hid.size();
        q36::detail::RamCaptureSource source;
        source.execution_frontier = 8;
        source.ledger_frontier    = static_cast<std::uint32_t>(retained.token_ids.size());
        source.text_kv_valid      = 8;
        source.tail_hidden_valid  = true;
        source.ledger             = retained.token_ids;
        source.identity           = &identity;
        source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity, 8);
        source.text               = &text;
        source.text_pool          = &text_pool;
        source.gdn                = &gdn;
        source.gdn_current_slot   = 0;
        source.tail_hidden        = &hidden;
        source.ladder_heads       = {head};
        source.stream             = ctx.copy_stream;
        return source;
    };

    q36::detail::KVRamCache cache(16ULL << 20);
    auto source_a = capture(retained_a, identity_a, text_a, hidden_a, conv_a, rec_a, hid_a);
    if (!capture_or_evict(cache, source_a)) {
        text_a.release();
        text_b.release();
        return fail("same-F FIFO first capture failed");
    }
    auto source_b = capture(retained_b, identity_b, text_b, hidden_b, conv_b, rec_b, hid_b);
    if (!capture_or_evict(cache, source_b)) {
        text_a.release();
        text_b.release();
        return fail("same-F FIFO second capture failed");
    }
    ctx.synchronize_all();

    const auto prefix = text_prompt({1, 2, 3, 4});
    const auto match  = cache.plan_match(prefix, q36::detail::prefix_hash_chain(prefix));
    if (!match || match->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        match->reuse_base != 4) {
        text_a.release();
        text_b.release();
        std::cerr << "same-F FIFO did not match a ladder head at F=4\n";
        return 1;
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(32);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {32});
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemset(gdn.conv_slot(layer, 2).data, 0, gdn.conv_slot(layer, 2).bytes()));
        CUDA_CHECK(
            cudaMemset(gdn.recurrent_slot(layer, 2).data, 0, gdn.recurrent_slot(layer, 2).bytes()));
    }
    q36::detail::RamRestoreTarget target;
    target.text             = &text_dest;
    target.text_pool        = &text_pool;
    target.text_dst_pages   = ninfer::pages_for_tokens(match->reuse_base);
    target.gdn              = &gdn;
    target.gdn_current_slot = 2;
    target.tail_hidden      = &hidden_out;
    target.reuse            = match->reuse;
    target.reuse_base       = match->reuse_base;
    target.stream           = ctx.copy_stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    ctx.synchronize_all();

    std::vector<unsigned char> conv_packed(gdn.conv_host_image_bytes(), 0);
    std::vector<unsigned char> rec_packed(gdn.recurrent_host_image_bytes(), 0);
    gdn.pack_slot_to_host(2, conv_packed.data(), rec_packed.data(), ctx.copy_stream);
    ctx.synchronize_all();
    int failures = 0;
    if (conv_packed != conv_a || rec_packed != rec_a) {
        std::cerr << "same-F FIFO unpacked the later entry instead of the first\n";
        ++failures;
    }
    std::vector<unsigned char> hidden_host(32);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != hid_a) {
        std::cerr << "same-F FIFO unpacked the later hidden instead of the first\n";
        ++failures;
    }
    cache.consume(match->entry_id);
    text_a.release();
    text_b.release();
    text_dest.release();
    return failures;
}

int test_failed_second_capture_discards_first(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    const auto prompt_a = text_prompt({11, 12, 13, 14});
    const auto prompt_b = text_prompt({21, 22, 23, 24});
    q36::detail::KVRamCache cache(4ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        alloc.release();
        std::cerr << "abandoned-capture first capture failed\n";
        return 1;
    }
    ctx.synchronize_all();
    cache.wait_pending_copies();
    const auto before = cache.snapshot();
    if (before.entry_count != 1 || before.used_bytes == 0) {
        alloc.release();
        std::cerr << "abandoned-capture first occupancy is wrong\n";
        return 1;
    }
    const auto first = cache.peek_oldest_unpinned();
    if (!first) {
        alloc.release();
        std::cerr << "abandoned-capture missing first id\n";
        return 1;
    }
    q36::PreparedPromptData retained = prompt_b;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt_b.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f =
        q36::detail::prefix_hash_at(retained.token_ids, identity, source.execution_frontier);
    source.text      = &alloc;
    source.text_pool = &pool;
    source.stream    = ctx.copy_stream;
    cache.test_fail_next_capture();
    if (cache.capture(source)) {
        alloc.release();
        std::cerr << "abandoned-capture second capture should have failed\n";
        return 1;
    }
    const auto leaked = cache.snapshot();
    if (leaked.entry_count != 1 || leaked.used_bytes != before.used_bytes) {
        alloc.release();
        std::cerr << "failed second capture changed RAM occupancy before discard\n";
        return 1;
    }
    if (!cache.evict_one_unpinned(*first)) {
        alloc.release();
        std::cerr << "abandoned-capture discard of first id failed\n";
        return 1;
    }
    const auto after = cache.snapshot();
    if (after.entry_count != 0 || after.used_bytes != 0) {
        alloc.release();
        std::cerr << "discard after failed second capture left RAM occupied\n";
        return 1;
    }
    alloc.release();
    return 0;
}

int test_discard_first_of_two_captures_keeps_second(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    const auto prompt_a = text_prompt({31, 32, 33, 34});
    const auto prompt_b = text_prompt({41, 42, 43, 44});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt_a, ctx.copy_stream) != 0) {
        alloc.release();
        std::cerr << "two-capture first capture failed\n";
        return 1;
    }
    ctx.synchronize_all();
    cache.wait_pending_copies();
    const auto first = cache.peek_oldest_unpinned();
    if (!first) {
        alloc.release();
        std::cerr << "two-capture missing first id\n";
        return 1;
    }
    if (capture_text_entry(cache, pool, alloc, prompt_b, ctx.copy_stream) != 0) {
        alloc.release();
        std::cerr << "two-capture second capture failed\n";
        return 1;
    }
    ctx.synchronize_all();
    cache.wait_pending_copies();
    if (cache.snapshot().entry_count != 2) {
        alloc.release();
        std::cerr << "two-capture occupancy is not 2\n";
        return 1;
    }
    if (!cache.evict_one_unpinned(*first)) {
        alloc.release();
        std::cerr << "two-capture discard of first id failed\n";
        return 1;
    }
    const auto after = cache.snapshot();
    if (after.entry_count != 1) {
        alloc.release();
        std::cerr << "discard of first capture did not leave exactly one entry\n";
        return 1;
    }
    if (cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a))) {
        alloc.release();
        std::cerr << "discarded first capture is still hittable\n";
        return 1;
    }
    const auto keep = cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!keep || keep->entry_id == *first) {
        alloc.release();
        std::cerr << "discard of first capture lost the second entry\n";
        return 1;
    }
    alloc.release();
    return 0;
}

int test_consume_retire_failure_keeps_record(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source    = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 19);
    ctx.synchronize_all();
    const auto prompt = text_prompt({50, 51, 52, 53});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, source, prompt, ctx.copy_stream) != 0) {
        source.release();
        return fail("retire-fail capture failed");
    }
    source.release();
    ctx.synchronize_all();
    cache.wait_pending_copies();
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) { return fail("retire-fail capture did not index"); }
    cache.claim(match->entry_id);
    const auto restores_before = cache.snapshot().restores;
    const auto pending_before  = cache.test_pending_copy_count();
    const auto entries_before = cache.snapshot().entry_count;
    const void* block          = cache.host_block(match->entry_id);
    if (block == nullptr) { return fail("retire-fail record had no host block"); }
    cache.test_fail_next_retire();
    bool threw = false;
    try {
        cache.consume(match->entry_id);
    } catch (const std::bad_alloc&) { threw = true; }
    if (!threw) { return fail("retire-fail consume did not throw"); }
    if (!cache.is_claimed(match->entry_id)) {
        return fail("retire-fail dropped the claim before commit");
    }
    if (cache.host_block(match->entry_id) != block) {
        return fail("retire-fail cleared the host block before commit");
    }
    if (cache.snapshot().restores != restores_before) {
        cache.release(match->entry_id);
        return fail("retire-fail counted a restore before commit");
    }
    if (cache.test_pending_copy_count() != pending_before) {
        cache.release(match->entry_id);
        return fail("retire-fail dropped pending copy ids before commit");
    }
    if (cache.snapshot().entry_count != entries_before) {
        cache.release(match->entry_id);
        return fail("retire-fail removed the FIFO record before commit");
    }
    cache.consume(match->entry_id);
    return 0;
}

int test_ram_copy_sync_does_not_hold_io_mutex(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    const auto prompt = text_prompt({10, 11, 12, 13});
    const auto chain  = q36::detail::prefix_hash_chain(prompt);
    auto wait_entered = [](q36::detail::KVRamCache& cache) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!cache.test_copy_sync_entered() &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return cache.test_copy_sync_entered();
    };
    auto probe = [](q36::detail::KVRamCache& cache) {
        const auto t0 = std::chrono::steady_clock::now();
        (void)cache.snapshot();
        return std::chrono::steady_clock::now() - t0;
    };

    {
        q36::detail::KVRamCache cache(8ULL << 20);
        if (capture_text_entry(cache, pool, alloc, prompt, ctx.copy_stream) != 0) {
            alloc.release();
            return fail("copy-sync harvest capture failed");
        }
        ctx.synchronize_all();
        cache.wait_pending_copies();
        cache.test_set_copy_sync_stall_ms(250);
        std::thread worker([&] { (void)cache.harvest_copy_seconds(); });
        if (!wait_entered(cache)) {
            cache.test_set_copy_sync_stall_ms(0);
            worker.join();
            alloc.release();
            return fail("copy-sync harvest stall was not reached");
        }
        const auto elapsed = probe(cache);
        cache.test_set_copy_sync_stall_ms(0);
        worker.join();
        if (elapsed > std::chrono::milliseconds(100)) {
            alloc.release();
            return fail("snapshot waited on harvest_copy_seconds CUDA sync");
        }
    }
    {
        q36::detail::KVRamCache cache(8ULL << 20);
        if (capture_text_entry(cache, pool, alloc, prompt, ctx.copy_stream) != 0) {
            alloc.release();
            return fail("copy-sync consume capture failed");
        }
        ctx.synchronize_all();
        cache.wait_pending_copies();
        const auto match = cache.plan_match(prompt, chain);
        if (!match) {
            alloc.release();
            return fail("copy-sync consume match failed");
        }
        cache.claim(match->entry_id);
        cache.test_set_copy_sync_stall_ms(250);
        std::thread worker([&] { cache.consume(match->entry_id); });
        if (!wait_entered(cache)) {
            cache.test_set_copy_sync_stall_ms(0);
            worker.join();
            alloc.release();
            return fail("copy-sync consume stall was not reached");
        }
        const auto elapsed = probe(cache);
        cache.test_set_copy_sync_stall_ms(0);
        worker.join();
        if (elapsed > std::chrono::milliseconds(100)) {
            alloc.release();
            return fail("snapshot waited on consume CUDA sync");
        }
    }
    {
        q36::detail::KVRamCache cache(8ULL << 20);
        if (capture_text_entry(cache, pool, alloc, prompt, ctx.copy_stream) != 0) {
            alloc.release();
            return fail("copy-sync evict capture failed");
        }
        ctx.synchronize_all();
        cache.wait_pending_copies();
        const auto ids = cache.fifo_ids();
        if (ids.empty()) {
            alloc.release();
            return fail("copy-sync evict had no fifo id");
        }
        cache.test_set_copy_sync_stall_ms(250);
        std::thread worker([&] { (void)cache.evict_one_unpinned(ids.front()); });
        if (!wait_entered(cache)) {
            cache.test_set_copy_sync_stall_ms(0);
            worker.join();
            alloc.release();
            return fail("copy-sync evict stall was not reached");
        }
        const auto elapsed = probe(cache);
        cache.test_set_copy_sync_stall_ms(0);
        worker.join();
        if (elapsed > std::chrono::milliseconds(100)) {
            alloc.release();
            return fail("snapshot waited on evict CUDA sync");
        }
    }
    alloc.release();
    return 0;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    auto paged_plan = plan_paged_cache(10, 10, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena paged_arena(paged_plan.bytes);
    ninfer::PagedKVPool paged_pool({paged_arena.base(), paged_arena.capacity()}, paged_plan.layout);

    auto keep = paged_pool.reserve(3);
    keep.materialize_pages(3);
    failures += round_trip_pool(ctx, paged_pool, 6, 6, 3, "INT8 fragmented PageMajor");
    auto scrambled = paged_pool.reserve(3);
    scrambled.materialize_pages(3);
    // Physical IDs are assigned from the free list; force a logical order that is not physical
    // order by packing the live mapping after a fragmented take. The round-trip above already
    // used non-contiguous IDs {0,1,2,6,7,8}-style after the reserved 3-page keep.
    keep.release();
    scrambled.release();

    auto consecutive_physical = paged_pool.reserve(4);
    consecutive_physical.materialize_pages(4);
    fill_logical_pages(paged_pool, consecutive_physical, 9);
    {
        auto ids = consecutive_physical.page_ids();
        if (ids.size() >= 3 && ids[1] + 1 == ids[2] && ids[0] + 1 != ids[1]) {
            // logical 1..2 is a physical run that is not a prefix of the logical sequence
        }
        const std::size_t image_bytes =
            ninfer::paged_kv_host_image_bytes(paged_pool, consecutive_physical.mapped_page_count());
        ninfer::HostPinnedArena host(std::max<std::size_t>(image_bytes, 256));
        void* image = host.try_alloc(image_bytes, 256);
        ninfer::pack_paged_kv_allocation_to_host(consecutive_physical, paged_pool, image,
                                                 ctx.stream);
        ctx.synchronize_all();
        consecutive_physical.release();
        auto restored = paged_pool.reserve(4);
        restored.materialize_pages(4);
        ninfer::unpack_paged_kv_allocation_from_host(restored, paged_pool, image, 4, 4, ctx.stream);
        ctx.synchronize_all();
        failures += expect_logical_pages(paged_pool, restored, 9, "no zero_pages sort");
        restored.release();
    }

    auto partial = paged_pool.reserve(3);
    partial.materialize_tokens(70);
    failures += expect_size(partial.mapped_page_count(), 2, "partial tail page count");
    fill_logical_pages(paged_pool, partial, 11);
    {
        const std::uint32_t captured = partial.mapped_page_count();
        const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(paged_pool, captured);
        ninfer::HostPinnedArena host(std::max<std::size_t>(image_bytes, 256));
        void* image = host.try_alloc(image_bytes, 256);
        ninfer::pack_paged_kv_allocation_to_host(partial, paged_pool, image, ctx.stream);
        ctx.synchronize_all();
        partial.release();
        auto restored = paged_pool.reserve(3);
        restored.materialize_pages(2);
        ninfer::unpack_paged_kv_allocation_from_host(restored, paged_pool, image, captured, 2,
                                                     ctx.stream);
        ctx.synchronize_all();
        failures += expect_logical_pages(paged_pool, restored, 11, "partial tail page");
        restored.release();
    }

    auto head_major_plan = plan_paged_cache(8, 8, 1, {{ninfer::DType::BF16, 16, 4}},
                                            ninfer::PagedKVPlaneOrder::HeadMajor);
    ninfer::DeviceArena head_major_arena(head_major_plan.bytes);
    ninfer::PagedKVPool head_major_pool({head_major_arena.base(), head_major_arena.capacity()},
                                        head_major_plan.layout);
    failures += round_trip_pool(ctx, head_major_pool, 3, 3, 21, "HeadMajor Hkv>1 count>1");

    auto bf16_plan = plan_paged_cache(8, 8, 2,
                                      {{ninfer::DType::BF16, 32, 2}, {ninfer::DType::BF16, 32, 2}});
    ninfer::DeviceArena bf16_arena(bf16_plan.bytes);
    ninfer::PagedKVPool bf16_pool({bf16_arena.base(), bf16_arena.capacity()}, bf16_plan.layout);
    failures += round_trip_pool(ctx, bf16_pool, 3, 3, 17, "BF16 PageMajor");

    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    {
        const ninfer::Tensor conv = gdn.conv_slot(0, 0);
        std::vector<unsigned char> pattern(conv.bytes());
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            pattern[i] = static_cast<unsigned char>(i * 3 + 1);
        }
        CUDA_CHECK(cudaMemcpy(conv.data, pattern.data(), pattern.size(), cudaMemcpyHostToDevice));
        const ninfer::Tensor rec = gdn.recurrent_slot(1, 0);
        std::vector<unsigned char> rec_pattern(rec.bytes(), 0xab);
        CUDA_CHECK(
            cudaMemcpy(rec.data, rec_pattern.data(), rec_pattern.size(), cudaMemcpyHostToDevice));
        std::vector<unsigned char> conv_host(gdn.conv_host_image_bytes());
        std::vector<unsigned char> rec_host(gdn.recurrent_host_image_bytes());
        gdn.pack_slot_to_host(0, conv_host.data(), rec_host.data(), ctx.stream);
        ctx.synchronize_all();
        gdn.unpack_slot_from_host(2, conv_host.data(), rec_host.data(), ctx.stream);
        ctx.synchronize_all();
        std::vector<unsigned char> conv_out(conv.bytes());
        CUDA_CHECK(cudaMemcpy(conv_out.data(), gdn.conv_slot(0, 2).data, conv_out.size(),
                              cudaMemcpyDeviceToHost));
        if (conv_out != pattern) {
            ++failures;
            std::cerr << "GDN conv host round-trip across slots failed\n";
        }
        std::vector<unsigned char> rec_out(rec.bytes());
        CUDA_CHECK(cudaMemcpy(rec_out.data(), gdn.recurrent_slot(1, 2).data, rec_out.size(),
                              cudaMemcpyDeviceToHost));
        if (rec_out != rec_pattern) {
            ++failures;
            std::cerr << "GDN recurrent host round-trip across slots failed\n";
        }
    }

    ninfer::LayoutBuilder cyclic_builder;
    const auto cyclic_layout =
        ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 32, 2, 8, 3);
    ninfer::DeviceArena cyclic_arena(cyclic_builder.finish(256));
    ninfer::CyclicKVCache cyclic({cyclic_arena.base(), cyclic_arena.capacity()}, cyclic_layout);
    {
        ninfer::CyclicKVCacheLayerView layer = cyclic.layer_view(0);
        std::vector<unsigned char> k_pattern(layer.k.slice(3, 0, 1).bytes(), 0x3c);
        CUDA_CHECK(cudaMemcpy(layer.k.slice(3, 0, 1).data, k_pattern.data(), k_pattern.size(),
                              cudaMemcpyHostToDevice));
        std::vector<unsigned char> v_pattern(layer.v.slice(3, 0, 1).bytes(), 0x4d);
        CUDA_CHECK(cudaMemcpy(layer.v.slice(3, 0, 1).data, v_pattern.data(), v_pattern.size(),
                              cudaMemcpyHostToDevice));
        std::vector<unsigned char> host(cyclic.lane_host_bytes());
        cyclic.copy_lane_to_host(0, host.data(), ctx.stream);
        ctx.synchronize_all();
        cyclic.copy_lane_from_host(host.data(), 2, ctx.stream);
        ctx.synchronize_all();
        std::vector<unsigned char> k_out(k_pattern.size());
        CUDA_CHECK(cudaMemcpy(k_out.data(), cyclic.layer_view(0).k.slice(3, 2, 1).data, k_out.size(),
                              cudaMemcpyDeviceToHost));
        std::vector<unsigned char> v_out(v_pattern.size());
        CUDA_CHECK(cudaMemcpy(v_out.data(), cyclic.layer_view(0).v.slice(3, 2, 1).data, v_out.size(),
                              cudaMemcpyDeviceToHost));
        if (k_out != k_pattern || v_out != v_pattern) {
            ++failures;
            std::cerr << "DFlash cyclic host round-trip across lanes failed\n";
        }
    }

    failures += test_kv_ram_index(ctx, paged_pool);
    failures += test_unpack_consume_and_drop(ctx, paged_pool);
    failures += test_frontier_beats_checkpoint(ctx, paged_pool);
    failures += test_asymmetric_fifo(ctx, paged_pool);
    failures += test_fifo_consume_middle(ctx, paged_pool);
    failures += test_fifo_evict_after_middle_consume(ctx, paged_pool);
    failures += test_prefix_unpack_preserves_tail(ctx, paged_pool);
    failures += test_consume_reaps_for_next_capture(ctx, paged_pool);
    failures += test_copies_ready_query(ctx, paged_pool);
    failures += test_copy_compute_stream_overlap(ctx, paged_pool);
    failures += test_unpack_without_harvest_keeps_save_and_load(ctx, paged_pool);
    failures += test_consume_without_harvest_clears_pending(ctx, paged_pool);
    failures += test_consume_after_unpack_folds_load(ctx, paged_pool);
    failures += test_event_overlap_unpack(ctx, paged_pool);
    failures += test_irregular_page_major_runs(ctx);
    failures += test_restore_throw_then_replay(ctx, paged_pool);
    failures += test_destructor_with_inflight_copies(ctx, paged_pool);
    failures += test_spill_drop_keeps_indexed_source(ctx, paged_pool);
    failures += test_full_state_image(ctx);
    failures += test_context_checkpoint_middle_head(ctx);
    failures += test_context_checkpoint_two_ram_entries(ctx);
    failures += test_context_checkpoint_ladder_beats_rewrite(ctx);
    failures += test_context_checkpoint_equal_execution_is_append(ctx);
    failures += test_context_checkpoint_hash_mismatch(ctx);
    failures += test_context_checkpoint_rollback_recapture(ctx);
    failures += test_context_checkpoint_consume_waits_copies(ctx);
    failures += test_context_checkpoint_catch_up_frontier(ctx);
    failures += test_turn_rollback_kind_roundtrip(ctx);
    failures += test_mixed_checkpoint_gdn_isolation(ctx);
    failures += test_context_checkpoint_dflash_cyclic_isolation(ctx);
    failures += test_same_f_rewrite_beats_ram_rollback(ctx);
    failures += test_rollback_skips_ahead_rewrite_gdn(ctx);
    failures += test_c2_lane1_rollback_slot_isolation(ctx);
    failures += test_rollback_head_gdn_geometry_mismatch(ctx);
    failures += test_context_checkpoint_same_f_fifo_first_wins(ctx);
    failures += test_failed_second_capture_discards_first(ctx, paged_pool);
    failures += test_discard_first_of_two_captures_keeps_second(ctx, paged_pool);
    failures += test_ram_copy_sync_does_not_hold_io_mutex(ctx, paged_pool);
    failures += test_consume_retire_failure_keeps_record(ctx, paged_pool);

    return failures == 0 ? 0 : fail("kv ram cache core test failed");
}
