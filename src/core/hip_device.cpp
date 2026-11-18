#include "core/hip_device.h"

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstdio>
#include <exception>

namespace ninfer {
namespace {
void check(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw HipError(std::string("expert helper HIP: ") + operation + ": " +
                       hipGetErrorString(status));
    }
}

void destroy_check(hipError_t status, const char* operation) noexcept {
    if (status != hipSuccess) {
        std::fprintf(stderr, "expert helper HIP teardown: %s: %s\n", operation,
                     hipGetErrorString(status));
        std::terminate();
    }
}

bool complete(hipError_t status) {
    if (status == hipErrorNotReady) { return false; }
    check(status, "completion query");
    return true;
}
} // namespace

std::string normalize_helper_pci(std::string_view pci) {
    if (pci.size() != 12 || pci[4] != ':' || pci[7] != ':' || pci[10] != '.') {
        throw std::invalid_argument("expert helper PCI must be domain:bus:device.function");
    }
    std::string result(pci);
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (i == 4 || i == 7 || i == 10) { continue; }
        char& c = result[i];
        if (c >= 'A' && c <= 'F') { c = static_cast<char>(c - 'A' + 'a'); }
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            throw std::invalid_argument("expert helper PCI contains a non-hex digit");
        }
    }
    if (result[8] > '1' || result[11] > '7') {
        throw std::invalid_argument("expert helper PCI device/function is out of range");
    }
    return result;
}

HipDevice::HipDevice(std::string_view pci) : pci_(normalize_helper_pci(pci)) {
    check(hipDeviceGetByPCIBusId(&ordinal_, pci_.c_str()), "select PCI device");
    bind();
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, ordinal_), "device properties");
    const std::string_view arch(properties.gcnArchName);
    if (arch.substr(0, arch.find(':')) != "gfx1201" || properties.canMapHostMemory == 0) {
        throw HipError("expert helper requires gfx1201 with mapped host memory");
    }
    // HostNativeAtomicSupported describes native CPU/GPU mixed RMW, not this protocol's
    // mapped system-scope GPU atomics. Both devices' exact U64 exchange is qualified with the
    // startup nonce before service readiness; the attribute alone cannot license execution.
}

void HipDevice::bind() const { check(hipSetDevice(ordinal_), "bind device"); }

HipDeviceInfo HipDevice::info() const {
    bind();
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, ordinal_), "device properties");
    HipDeviceInfo result{
        .pci = pci_, .name = properties.name, .architecture = properties.gcnArchName};
    check(hipRuntimeGetVersion(&result.runtime_version), "runtime version");
    check(hipMemGetInfo(&result.free_bytes, &result.total_bytes), "memory info");
    return result;
}

HipStream::HipStream() {
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
    stream_ = stream;
}

HipStream::~HipStream() {
    if (stream_ != nullptr) {
        destroy_check(hipStreamSynchronize(static_cast<hipStream_t>(stream_)), "drain stream");
        destroy_check(hipStreamDestroy(static_cast<hipStream_t>(stream_)), "destroy stream");
    }
}

void HipStream::synchronize() const {
    check(hipStreamSynchronize(static_cast<hipStream_t>(stream_)), "synchronize stream");
}

bool HipStream::ready() const {
    return complete(hipStreamQuery(static_cast<hipStream_t>(stream_)));
}

HipEvent::HipEvent() {
    hipEvent_t event = nullptr;
    check(hipEventCreateWithFlags(&event, hipEventDisableTiming | hipEventBlockingSync),
          "create event");
    event_ = event;
}

HipEvent::~HipEvent() {
    destroy_check(hipEventDestroy(static_cast<hipEvent_t>(event_)), "destroy event");
}

void HipEvent::record(const HipStream& stream) {
    check(hipEventRecord(static_cast<hipEvent_t>(event_), static_cast<hipStream_t>(stream.get())),
          "record event");
}

void HipEvent::synchronize() const {
    check(hipEventSynchronize(static_cast<hipEvent_t>(event_)), "synchronize event");
}

bool HipEvent::ready() const { return complete(hipEventQuery(static_cast<hipEvent_t>(event_))); }

HipBuffer::HipBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes != 0) { check(hipMalloc(&data_, bytes), "allocate device buffer"); }
}

HipBuffer::~HipBuffer() {
    if (data_ != nullptr) { destroy_check(hipFree(data_), "free device buffer"); }
}

HipPinnedBuffer::HipPinnedBuffer(std::size_t bytes) {
    if (bytes == 0) { throw std::invalid_argument("HIP staging buffer cannot be empty"); }
    check(hipHostMalloc(&data_, bytes, hipHostMallocDefault), "allocate pinned staging");
}

HipPinnedBuffer::~HipPinnedBuffer() { destroy_check(hipHostFree(data_), "free pinned staging"); }

HipHostRegistration::HipHostRegistration(void* host, std::size_t bytes) {
    check(hipHostRegister(host, bytes, hipHostRegisterMapped), "register CUDA-owned mailbox");
    const hipError_t status = hipHostGetDevicePointer(&alias_, host, 0);
    if (status != hipSuccess) {
        destroy_check(hipHostUnregister(host), "rollback mailbox registration");
        check(status, "HIP mailbox alias");
    }
    host_ = host;
}

HipHostRegistration::~HipHostRegistration() {
    destroy_check(hipHostUnregister(host_), "unregister mailbox");
}

HipGraph::HipGraph(const HipStream& stream) : stream_(stream.get()) {
    check(hipStreamBeginCapture(static_cast<hipStream_t>(stream_), hipStreamCaptureModeThreadLocal),
          "begin capture");
}

HipGraph::~HipGraph() {
    if (capturing_) {
        hipGraph_t abandoned = nullptr;
        const hipError_t status =
            hipStreamEndCapture(static_cast<hipStream_t>(stream_), &abandoned);
        if (status != hipSuccess && status != hipErrorStreamCaptureInvalidated) {
            destroy_check(status, "abort capture");
        }
        if (abandoned != nullptr) {
            destroy_check(hipGraphDestroy(abandoned), "destroy abandoned graph");
        }
    }
    if (executable_ != nullptr) {
        destroy_check(hipGraphExecDestroy(static_cast<hipGraphExec_t>(executable_)),
                      "destroy graph executable");
    }
    if (graph_ != nullptr) {
        destroy_check(hipGraphDestroy(static_cast<hipGraph_t>(graph_)), "destroy graph");
    }
}

void HipGraph::instantiate() {
    if (!capturing_) { throw std::logic_error("HIP graph is already instantiated"); }
    hipGraph_t graph        = nullptr;
    const hipError_t status = hipStreamEndCapture(static_cast<hipStream_t>(stream_), &graph);
    capturing_              = false;
    check(status, "end capture");
    graph_                    = graph;
    hipGraphExec_t executable = nullptr;
    check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate graph");
    executable_ = executable;
}

void HipGraph::launch(const HipStream& stream) const {
    if (executable_ == nullptr) { throw std::logic_error("HIP graph has no executable"); }
    check(hipGraphLaunch(static_cast<hipGraphExec_t>(executable_),
                         static_cast<hipStream_t>(stream.get())),
          "launch graph");
}

void hip_copy_to_device(void* dst, const void* src, std::size_t bytes, const HipStream& stream) {
    check(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice,
                         static_cast<hipStream_t>(stream.get())),
          "H2D copy");
}

void hip_copy_to_host(void* dst, const void* src, std::size_t bytes, const HipStream& stream) {
    check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost,
                         static_cast<hipStream_t>(stream.get())),
          "D2H copy");
}

void hip_zero(void* dst, std::size_t bytes, const HipStream& stream) {
    check(hipMemsetAsync(dst, 0, bytes, static_cast<hipStream_t>(stream.get())),
          "clear device buffer");
}
} // namespace ninfer
