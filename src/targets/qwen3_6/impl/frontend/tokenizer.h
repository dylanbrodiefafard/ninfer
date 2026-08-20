#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

struct EncodeOptions {
    bool parse_added_tokens = true;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
};

struct EncodedText {
    std::vector<int> ids;
    std::optional<std::uint32_t> prefix_tokens;
};

struct BpePairSlot {
    std::uint64_t key = ~std::uint64_t{0};
    int rank          = 0;
    int result        = -1;
};

struct BpePairTable {
    std::vector<BpePairSlot> slots;
    std::size_t mask = 0;
};

struct AddedTrieNode {
    std::array<int, 256> next{};
    int token_index = -1;

    AddedTrieNode() { next.fill(-1); }
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources);

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    EncodedText encode(std::string_view text, std::optional<std::size_t> prefix_byte_end,
                       EncodeOptions options = {}) const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;
    std::string_view decode_token_bytes(int id, bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] bool is_valid_token(int id) const noexcept;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;

private:
    std::vector<std::string> id_to_token_;
    std::vector<std::string> id_to_decoded_bytes_;
    std::vector<std::uint8_t> valid_token_ids_;
    std::vector<std::uint8_t> special_token_ids_;
    std::vector<std::uint8_t> added_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    BpePairTable bpe_pair_table_;
    std::vector<std::vector<int>> intern_emit_ids_;
    std::array<int, 256> byte_to_intern_id_{};
    bool has_bpe_merges_ = true;
    std::vector<AddedToken> added_tokens_;
    std::vector<AddedTrieNode> added_trie_;
    std::array<std::uint8_t, 256> added_start_bytes_{};
    std::vector<int> default_stop_token_ids_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
