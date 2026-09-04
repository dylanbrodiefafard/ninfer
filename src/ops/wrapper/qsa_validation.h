#pragma once

#include "core/tensor.h"
#include "ninfer/ops/qsa.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer::ops::detail {

struct QsaAddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
    const char* name;
};

/** Validate every persistent plane, including the alignment required by its typed accesses. */
int qsa_validate_state(const QsaStateView& state, const char* op);

QsaAddressRange qsa_address_range(const Tensor& tensor, const char* op, const char* name);
QsaAddressRange qsa_address_range(const Weight& weight, const char* op, const char* name);

void qsa_require_disjoint(std::initializer_list<QsaAddressRange> ranges, const char* op);

} // namespace ninfer::ops::detail
