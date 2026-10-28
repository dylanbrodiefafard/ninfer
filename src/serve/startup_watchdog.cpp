#include "serve/startup_watchdog.h"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace ninfer::serve {

StartupWatchdog::StartupWatchdog(std::chrono::milliseconds timeout)
    : timeout_(timeout), deadline_(Clock::now() + timeout) {
    if (timeout <= std::chrono::milliseconds::zero()) {
        throw std::invalid_argument("startup watchdog timeout must be positive");
    }
    worker_ = std::thread([this] { run(); });
}

StartupWatchdog::~StartupWatchdog() { complete(); }

void StartupWatchdog::progress(std::string_view phase, std::uint64_t done) {
    std::lock_guard lock(mutex_);
    if (stopped_ || (phase == phase_ && done <= done_)) { return; }
    phase_    = phase;
    done_     = done;
    deadline_ = Clock::now() + timeout_;
    changed_.notify_one();
}

void StartupWatchdog::begin_warmup() { progress("warmup", 0); }

void StartupWatchdog::complete() {
    {
        std::lock_guard lock(mutex_);
        stopped_ = true;
    }
    changed_.notify_one();
    if (worker_.joinable()) { worker_.join(); }
}

void StartupWatchdog::run() {
    std::unique_lock lock(mutex_);
    while (!stopped_) {
        changed_.wait_until(lock, deadline_);
        if (stopped_ || Clock::now() < deadline_) { continue; }
        char message[512];
        const int count =
            std::snprintf(message, sizeof(message),
                          "ninfer-serve: startup watchdog: no progress for %lld ms during %.200s; "
                          "exiting without HIP cleanup so the supervisor can retry\n",
                          static_cast<long long>(timeout_.count()), phase_.c_str());
        // Bypass iostream/console locks: the stalled thread may hold one. The process owns
        // no usable inference state yet; destructors could block in the same driver.
        const int flags = ::fcntl(STDERR_FILENO, F_GETFL);
        if (count > 0 && flags >= 0 && ::fcntl(STDERR_FILENO, F_SETFL, flags | O_NONBLOCK) == 0) {
            const auto written = ::write(STDERR_FILENO, message, static_cast<std::size_t>(count));
            (void)written;
            (void)::fcntl(STDERR_FILENO, F_SETFL, flags);
        }
        std::_Exit(124);
    }
}

} // namespace ninfer::serve
