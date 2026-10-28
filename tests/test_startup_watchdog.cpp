#include "serve/startup_watchdog.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;

namespace {
using namespace std::chrono_literals;

int child(std::string_view mode) {
    if (mode == "progress") {
        ninfer::serve::StartupWatchdog watchdog(500ms);
        for (std::uint64_t done = 0; done < 8; ++done) {
            watchdog.progress("weights", done);
            std::this_thread::sleep_for(100ms);
        }
        watchdog.begin_warmup();
        std::this_thread::sleep_for(100ms);
        watchdog.complete();
        // Serving may be idle indefinitely after warmup; the guard must stay disarmed.
        std::this_thread::sleep_for(600ms);
        return 0;
    }
    if (mode == "unwind") {
        try {
            ninfer::serve::StartupWatchdog watchdog(150ms);
            throw std::runtime_error("ordinary startup failure");
        } catch (const std::runtime_error& error) {
            if (std::string_view(error.what()) != "ordinary startup failure") { return 1; }
        }
        std::this_thread::sleep_for(250ms);
        return 0;
    }
    // Normal cleanup must never run on expiry: it can block in the failed driver.
    std::atexit([] { std::_Exit(77); });
    ninfer::serve::StartupWatchdog watchdog(150ms);
    if (mode == "warmup") { watchdog.begin_warmup(); }
    if (mode == "repeated") {
        watchdog.progress("weights", 5);
        for (;;) {
            watchdog.progress("weights", 3);
            watchdog.progress("weights", 5);
            std::this_thread::sleep_for(5ms);
        }
    }
    if (mode == "full-log") {
        const int flags = ::fcntl(STDERR_FILENO, F_GETFL);
        if (flags < 0 || ::fcntl(STDERR_FILENO, F_SETFL, flags | O_NONBLOCK) != 0) {
            std::_Exit(78);
        }
        const std::array<char, 4096> padding{};
        while (::write(STDERR_FILENO, padding.data(), padding.size()) > 0) {}
        if (::fcntl(STDERR_FILENO, F_SETFL, flags) != 0) { std::_Exit(78); }
    }
    // A driver call may hold an ordinary logging lock. The watchdog must bypass it.
    ::flockfile(stderr);
    for (;;) { ::pause(); }
}

bool run(const char* executable, const char* mode, bool expires) {
    int output[2];
    if (::pipe(output) != 0) { throw std::runtime_error("watchdog test pipe failed"); }
    const int original_flags = ::fcntl(output[1], F_GETFL);
    posix_spawn_file_actions_t actions;
    ::posix_spawn_file_actions_init(&actions);
    ::posix_spawn_file_actions_adddup2(&actions, output[1], STDERR_FILENO);
    ::posix_spawn_file_actions_addclose(&actions, output[0]);
    ::posix_spawn_file_actions_addclose(&actions, output[1]);
    char* args[]      = {const_cast<char*>(executable), const_cast<char*>("--child"),
                         const_cast<char*>(mode), nullptr};
    pid_t pid         = 0;
    const int spawned = ::posix_spawn(&pid, executable, &actions, nullptr, args, environ);
    ::posix_spawn_file_actions_destroy(&actions);
    if (spawned != 0) {
        ::close(output[1]);
        ::close(output[0]);
        throw std::runtime_error("watchdog test spawn failed");
    }
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    int status          = 0;
    bool exited         = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (::waitpid(pid, &status, WNOHANG) == pid) {
            exited = true;
            break;
        }
        std::this_thread::sleep_for(10ms);
    }
    if (!exited) {
        ::kill(pid, SIGKILL);
        (void)::waitpid(pid, &status, 0);
    }
    const bool flags_preserved = ::fcntl(output[1], F_GETFL) == original_flags;
    ::close(output[1]);
    std::string diagnostic;
    std::array<char, 4096> buffer;
    for (;;) {
        const auto bytes = ::read(output[0], buffer.data(), buffer.size());
        if (bytes <= 0) { break; }
        diagnostic.append(buffer.data(), static_cast<std::size_t>(bytes));
    }
    ::close(output[0]);
    const std::string_view expected_phase = std::string_view(mode) == "warmup" ? "warmup"
                                            : std::string_view(mode) == "repeated"
                                                ? "weights"
                                                : "device initialization";
    const bool correct = exited && flags_preserved && WIFEXITED(status) &&
                         WEXITSTATUS(status) == (expires ? 124 : 0) &&
                         (!expires || std::string_view(mode) == "full-log" ||
                          (diagnostic.find("startup watchdog") != std::string::npos &&
                           diagnostic.find(expected_phase) != std::string::npos));
    if (!correct) {
        std::cerr << "startup watchdog case " << mode << " failed: exited=" << exited
                  << " status=" << status << '\n';
    }
    return correct;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "--child") { return child(argv[2]); }
    int failures = 0;
    for (const char* mode : {"stalled", "repeated", "warmup", "full-log"}) {
        failures += !run(argv[0], mode, true);
    }
    for (const char* mode : {"progress", "unwind"}) { failures += !run(argv[0], mode, false); }
    return failures == 0 ? 0 : 1;
}
