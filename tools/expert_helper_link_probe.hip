// Host <-> R9700 link probe for the Flash-Next expert helper's mailbox transport: hipMemcpyAsync
// between registered host memory and device buffers, through the mailbox's own HIP objects
// (HipHostRegistration, HipStream, hip_copy_to_device, hip_copy_to_host in core/hip_device.h).
//
// The mailbox's host memory is CUDA-owned (cudaHostAlloc, mapped and portable) and registered
// with HIP by hipHostRegister(hipHostRegisterMapped). This probe uses no CUDA: its host memory
// is anonymous page-aligned memory, populated before registration as cudaHostAlloc returns it,
// then registered by the same HipHostRegistration. `--host hip-pinned` uses hipHostMalloc
// staging (HipPinnedBuffer) instead.
//
// Per token count T, both directions move T x 2560 x 2 bytes (BF16 activations to the helper, a
// BF16 partial back), each on its own non-blocking stream:
//   h2d, d2h     one direction alone, timed by events around the copy on its stream;
//   concurrent   both in flight: the D2H stream waits on the H2D stream's start event, and each
//                direction is timed from that common start to its own completion event.
// Every repetition starts with both streams idle. Reported: the median over the repetitions, in
// GB/s (1e9 bytes per second) per direction, and for the concurrent case also the aggregate
// 2 x bytes over the later completion.
//
//   ninfer_expert_helper_link_probe [--pci DDDD:BB:DD.F] [--host registered|hip-pinned]
//       [--repeats R] [--tokens T ...]
#include "core/hip_device.h"

#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::HipBuffer;
using ninfer::HipPinnedBuffer;
using ninfer::HipStream;

constexpr std::size_t kRowBytes = 2560 * sizeof(std::uint16_t); // one BF16 [2560] row
constexpr int kWarmups          = 3;

void hip_check(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

// Anonymous private pages, written once so every page is populated before registration.
class AnonymousPages {
public:
    explicit AnonymousPages(std::size_t bytes) : bytes_(bytes) {
        void* data =
            mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (data == MAP_FAILED) { throw std::runtime_error("mmap host buffer"); }
        data_ = data;
        std::memset(data_, 0x5A, bytes_);
    }

    ~AnonymousPages() { (void)munmap(data_, bytes_); }

    AnonymousPages(const AnonymousPages&)            = delete;
    AnonymousPages& operator=(const AnonymousPages&) = delete;

    [[nodiscard]] void* data() const noexcept { return data_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_;
};

// One host buffer of either kind; data() is the CPU address the mailbox enqueues DMA with.
class HostBuffer {
public:
    HostBuffer(std::size_t bytes, bool registered) {
        if (registered) {
            pages_        = std::make_unique<AnonymousPages>(bytes);
            registration_ = std::make_unique<ninfer::HipHostRegistration>(pages_->data(), bytes);
            data_         = pages_->data();
        } else {
            pinned_ = std::make_unique<HipPinnedBuffer>(bytes);
            data_   = pinned_->data();
            std::memset(data_, 0x5A, bytes);
        }
    }

    [[nodiscard]] void* data() const noexcept { return data_; }

private:
    // Declaration order: the registration dies before the pages it registers.
    std::unique_ptr<AnonymousPages> pages_;
    std::unique_ptr<ninfer::HipHostRegistration> registration_;
    std::unique_ptr<HipPinnedBuffer> pinned_;
    void* data_ = nullptr;
};

class TimingEvent {
public:
    TimingEvent() { hip_check(hipEventCreate(&event_), "create event"); }

    ~TimingEvent() { (void)hipEventDestroy(event_); }

    TimingEvent(const TimingEvent&)            = delete;
    TimingEvent& operator=(const TimingEvent&) = delete;

    [[nodiscard]] hipEvent_t get() const noexcept { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

hipStream_t raw(const HipStream& stream) { return static_cast<hipStream_t>(stream.get()); }

float elapsed_ms(const TimingEvent& from, const TimingEvent& to) {
    float ms = 0.0F;
    hip_check(hipEventElapsedTime(&ms, from.get(), to.get()), "elapsed time");
    return ms;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 1 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

std::string first_gfx1201_pci() {
    int count = 0;
    hip_check(hipGetDeviceCount(&count), "device count");
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, device), "device properties");
        if (std::strncmp(properties.gcnArchName, "gfx1201", 7) == 0) {
            std::array<char, 32> pci{};
            hip_check(hipDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), device),
                      "PCI bus id");
            return pci.data();
        }
    }
    throw std::runtime_error("no gfx1201 device");
}

int main_impl(int argc, char** argv) {
    std::string pci;
    std::string host_mode = "registered";
    int repeats           = 20;
    std::vector<int> extents;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--pci" && i + 1 < argc) {
            pci = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            host_mode = argv[++i];
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if (arg == "--tokens" && i + 1 < argc) {
            extents.push_back(std::stoi(argv[++i]));
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [--pci DDDD:BB:DD.F] [--host registered|hip-pinned] [--repeats R]"
                         " [--tokens T ...]\n";
            return 2;
        }
    }
    if (host_mode != "registered" && host_mode != "hip-pinned") {
        throw std::invalid_argument("--host must be registered or hip-pinned");
    }
    if (repeats < 1) { throw std::invalid_argument("--repeats must be positive"); }
    if (extents.empty()) { extents = {1024, 4096, 16384}; }
    for (const int tokens : extents) {
        if (tokens < 1) { throw std::invalid_argument("--tokens must be positive"); }
    }

    const ninfer::HipDevice device(pci.empty() ? first_gfx1201_pci() : pci);
    device.bind();
    const ninfer::HipDeviceInfo info = device.info();
    const std::size_t max_bytes =
        static_cast<std::size_t>(*std::max_element(extents.begin(), extents.end())) * kRowBytes;

    const bool registered = host_mode == "registered";
    const HostBuffer upload_source(max_bytes, registered);
    const HostBuffer download_target(max_bytes, registered);
    const HipBuffer upload_target(max_bytes);
    const HipBuffer download_source(max_bytes);
    const HipStream h2d;
    const HipStream d2h;
    hip_check(hipMemset(download_source.data(), 0x3C, max_bytes), "fill device source");
    hip_check(hipDeviceSynchronize(), "fill device source");
    const TimingEvent start;
    const TimingEvent done;
    const TimingEvent h2d_done;
    const TimingEvent d2h_done;

    const auto upload = [&](std::size_t bytes) {
        ninfer::hip_copy_to_device(upload_target.data(), upload_source.data(), bytes, h2d);
    };
    const auto download = [&](std::size_t bytes) {
        ninfer::hip_copy_to_host(download_target.data(), download_source.data(), bytes, d2h);
    };
    // One timed direction alone: milliseconds from start to done on `stream`.
    const auto alone = [&](const HipStream& stream, const auto& copy, std::size_t bytes) {
        hip_check(hipEventRecord(start.get(), raw(stream)), "record start");
        copy(bytes);
        hip_check(hipEventRecord(done.get(), raw(stream)), "record done");
        hip_check(hipEventSynchronize(done.get()), "wait done");
        return static_cast<double>(elapsed_ms(start, done));
    };
    // Both directions from a common start: milliseconds to each completion.
    const auto both = [&](std::size_t bytes) {
        hip_check(hipEventRecord(start.get(), raw(h2d)), "record start");
        hip_check(hipStreamWaitEvent(raw(d2h), start.get(), 0), "join start");
        upload(bytes);
        download(bytes);
        hip_check(hipEventRecord(h2d_done.get(), raw(h2d)), "record h2d done");
        hip_check(hipEventRecord(d2h_done.get(), raw(d2h)), "record d2h done");
        hip_check(hipEventSynchronize(h2d_done.get()), "wait h2d");
        hip_check(hipEventSynchronize(d2h_done.get()), "wait d2h");
        return std::array<double, 2>{static_cast<double>(elapsed_ms(start, h2d_done)),
                                     static_cast<double>(elapsed_ms(start, d2h_done))};
    };

    std::printf("device %s (%s, %s), host %s, %d repetitions, median GB/s (1e9 B/s)\n",
                info.pci.c_str(), info.name.c_str(), info.architecture.c_str(), host_mode.c_str(),
                repeats);
    std::printf("%6s %9s %9s %9s %12s %12s %12s %10s\n", "T", "MB/dir", "h2d", "d2h", "conc_h2d",
                "conc_d2h", "conc_total", "conc_ms");
    for (const int tokens : extents) {
        const std::size_t bytes = static_cast<std::size_t>(tokens) * kRowBytes;
        std::vector<double> h2d_ms;
        std::vector<double> d2h_ms;
        std::vector<double> both_h2d_ms;
        std::vector<double> both_d2h_ms;
        std::vector<double> both_ms;
        for (int r = -kWarmups; r < repeats; ++r) {
            const double up                     = alone(h2d, upload, bytes);
            const double down                   = alone(d2h, download, bytes);
            const std::array<double, 2> overlap = both(bytes);
            if (r < 0) { continue; }
            h2d_ms.push_back(up);
            d2h_ms.push_back(down);
            both_h2d_ms.push_back(overlap[0]);
            both_d2h_ms.push_back(overlap[1]);
            both_ms.push_back(std::max(overlap[0], overlap[1]));
        }
        const auto rate = [&](double ms, double multiple) {
            return multiple * static_cast<double>(bytes) / (ms * 1e6);
        };
        const double total_ms = median(both_ms);
        std::printf("%6d %9.2f %9.2f %9.2f %12.2f %12.2f %12.2f %10.3f\n", tokens,
                    static_cast<double>(bytes) / 1e6, rate(median(h2d_ms), 1.0),
                    rate(median(d2h_ms), 1.0), rate(median(both_h2d_ms), 1.0),
                    rate(median(both_d2h_ms), 1.0), rate(total_ms, 2.0), total_ms);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "expert_helper_link_probe: " << error.what() << '\n';
        return 1;
    }
}
