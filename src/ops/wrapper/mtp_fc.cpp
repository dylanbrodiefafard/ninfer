#include "ninfer/ops/mtp_fc.h"

#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_bf16_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t tokens,
                         const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() ||
        tensor.ne[0] != rows || tensor.ne[1] != tokens || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("mtp_fc: invalid ") + name);
    }
    if (!aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("mtp_fc: ") + name +
                                    " must be non-null and 16-byte aligned");
    }
}

} // namespace

void mtp_fc(const Tensor& embedding_norm, const Tensor& hidden_norm, const Weight& weight,
            Tensor& out, cudaStream_t stream) {
    if (weight.qtype != QType::NVFP4) {
        throw std::invalid_argument("mtp_fc: weight must be NVFP4");
    }
    detail::validate_nvfp4_weight(weight, "mtp_fc");
    if (weight.n != 5120 || weight.k != 10240) {
        throw std::invalid_argument("mtp_fc: registered problem is [5120,10240]");
    }
    const std::int32_t tokens = embedding_norm.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("mtp_fc: T must be positive"); }
    require_bf16_matrix(embedding_norm, 5120, tokens, "embedding_norm");
    require_bf16_matrix(hidden_norm, 5120, tokens, "hidden_norm");
    require_bf16_matrix(out, 5120, tokens, "out");
    if (embedding_norm.data == hidden_norm.data || embedding_norm.data == out.data ||
        hidden_norm.data == out.data) {
        throw std::invalid_argument("mtp_fc: inputs and output must not alias");
    }
    detail::launch_nvfp4_mtp_fc_splitk(embedding_norm, hidden_norm, weight, out, stream);
}

} // namespace ninfer::ops
