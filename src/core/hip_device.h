#pragma once

// Optional expert-helper runtime boundary. No HIP headers escape this interface. All objects
// belong to the one startup-selected device; callers bind it before using their handles.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer {

class HipError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Strict domain:bus:device.function, normalized to lowercase. Throws invalid_argument.
[[nodiscard]] std::string normalize_helper_pci(std::string_view pci);

struct HipDeviceInfo {
    std::string pci, name, architecture;
    int runtime_version    = 0;
    std::size_t free_bytes = 0, total_bytes = 0;
};

class HipDevice {
public:
    // Selects exactly pci; requires gfx1201 and mapped host memory. The mailbox's startup
    // nonce must separately qualify system-scope U64 atomics before execution is licensed.
    explicit HipDevice(std::string_view pci);
    void bind() const;
    [[nodiscard]] HipDeviceInfo info() const;
private:
    int ordinal_ = -1;
    std::string pci_;
};

class HipStream {
public:
    HipStream();
    ~HipStream();
    HipStream(const HipStream&)            = delete;
    HipStream& operator=(const HipStream&) = delete;

    [[nodiscard]] void* get() const noexcept { return stream_; }

    void synchronize() const;
    [[nodiscard]] bool ready() const;
private:
    void* stream_ = nullptr;
};

class HipEvent {
public:
    HipEvent();
    ~HipEvent();
    HipEvent(const HipEvent&)            = delete;
    HipEvent& operator=(const HipEvent&) = delete;
    void record(const HipStream& stream);
    void synchronize() const;
    [[nodiscard]] bool ready() const;
private:
    void* event_ = nullptr;
};

class HipBuffer {
public:
    explicit HipBuffer(std::size_t bytes = 0);
    ~HipBuffer();
    HipBuffer(const HipBuffer&)            = delete;
    HipBuffer& operator=(const HipBuffer&) = delete;

    [[nodiscard]] void* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }
private:
    void* data_        = nullptr;
    std::size_t bytes_ = 0;
};

class HipPinnedBuffer {
public:
    explicit HipPinnedBuffer(std::size_t bytes);
    ~HipPinnedBuffer();
    HipPinnedBuffer(const HipPinnedBuffer&)            = delete;
    HipPinnedBuffer& operator=(const HipPinnedBuffer&) = delete;

    [[nodiscard]] void* data() const noexcept { return data_; }
private:
    void* data_ = nullptr;
};

// Explicit HIP registration of CUDA-owned mapped memory. Must die before its CUDA owner and
// only after all HIP users have completed. Does not register the expert host store.
class HipHostRegistration {
public:
    HipHostRegistration(void* host, std::size_t bytes);
    ~HipHostRegistration();
    HipHostRegistration(const HipHostRegistration&)            = delete;
    HipHostRegistration& operator=(const HipHostRegistration&) = delete;

    [[nodiscard]] void* alias() const noexcept { return alias_; }
private:
    void* host_  = nullptr;
    void* alias_ = nullptr;
};

// Explicit stream-capture transaction: destruction aborts an unfinished capture. Instantiate
// only once, outside CUDA capture. Executables must be idle before destruction.
class HipGraph {
public:
    explicit HipGraph(const HipStream& stream);
    ~HipGraph();
    HipGraph(const HipGraph&)            = delete;
    HipGraph& operator=(const HipGraph&) = delete;
    void instantiate();
    void launch(const HipStream& stream) const;
private:
    void* stream_     = nullptr;
    void* graph_      = nullptr;
    void* executable_ = nullptr;
    bool capturing_   = true;
};

void hip_copy_to_device(void* dst, const void* src, std::size_t bytes, const HipStream& stream);
void hip_copy_to_host(void* dst, const void* src, std::size_t bytes, const HipStream& stream);
void hip_zero(void* dst, std::size_t bytes, const HipStream& stream);

} // namespace ninfer
