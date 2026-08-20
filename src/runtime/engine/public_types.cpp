#include "ninfer/types.h"

#include <utility>

namespace ninfer {

CancellationView::CancellationView(std::function<bool()> requested)
    : requested_(std::move(requested)) {}

bool CancellationView::requested() const { return requested_ && requested_(); }

bool CancellationView::armed() const noexcept { return static_cast<bool>(requested_); }

} // namespace ninfer
