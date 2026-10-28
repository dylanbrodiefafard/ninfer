#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::serve {

// A stuck HIP call cannot be unwound safely. Exit the process so its supervisor can retry.
// This guard covers construction and warmup only; it is disarmed before serving requests.
class StartupWatchdog {
public:
    explicit StartupWatchdog(std::chrono::milliseconds timeout = std::chrono::seconds(180));
    ~StartupWatchdog();

    StartupWatchdog(const StartupWatchdog&)            = delete;
    StartupWatchdog& operator=(const StartupWatchdog&) = delete;

    void progress(std::string_view phase, std::uint64_t done);
    void begin_warmup();
    void complete();

private:
    using Clock = std::chrono::steady_clock;
    void run();

    const std::chrono::milliseconds timeout_;
    std::mutex mutex_;
    std::condition_variable changed_;
    Clock::time_point deadline_;
    std::string phase_  = "device initialization";
    std::uint64_t done_ = 0;
    bool stopped_       = false;
    std::thread worker_;
};

} // namespace ninfer::serve
