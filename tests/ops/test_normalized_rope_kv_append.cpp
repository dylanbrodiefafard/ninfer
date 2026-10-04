#include "ops/normalized_rope_kv_append_reference.h"
#include "ops/sanitizer_scope.h"

#include "core/decode_graph.h"
#include "core/device.h"

#include <array>
#include <exception>
#include <iostream>

using namespace ninfer;
namespace ref = ninfer::test::normalized_append;

namespace {

int prefix_case(int tokens, int batch, cudaStream_t stream) {
    ref::Fixture data(tokens, batch, 4093);
    DecodeGraphDefinition definition;
    definition.capture(stream, [&] { data.launch(stream); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    for (const int count : std::array{0, 1, -1, tokens}) {
        for (const bool captured : std::array{false, true}) {
            data.reset(count, captured);
            if (captured) {
                graph.launch(stream);
            } else {
                data.launch(stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            failures += data.verify("normalized_rope_kv_append B=" + std::to_string(batch) +
                                    " W=" + std::to_string(tokens) + " C=" + std::to_string(count) +
                                    (captured ? " graph" : " eager"));
        }
    }
    return failures;
}

int replacement_case(cudaStream_t stream) {
    ref::Fixture data(ref::kCapacity, 1, 4093);
    data.launch(stream, true);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int failures = data.verify("normalized_rope_kv_append full replacement eager");
    DecodeGraphDefinition definition;
    definition.capture(stream, [&] { data.launch(stream, true); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    data.reset(ref::kCapacity, true);
    graph.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    failures += data.verify("normalized_rope_kv_append full replacement graph lane reuse");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (test::require_cuda() != 0) { return 1; }
        DeviceContext device;
        int failures = 0;
        if (test::sanitizer_scope(argc, argv)) {
            failures += prefix_case(3, 1, device.stream);
            failures += prefix_case(8, 6, device.stream);
        } else {
            for (const int batch : std::array{1, 2, 6}) {
                for (const int tokens : std::array{3, 4, 5, 6, 7, 8, 12}) {
                    failures += prefix_case(tokens, batch, device.stream);
                }
            }
            for (const auto profile : std::array{ref::InputProfile::Varied, ref::InputProfile::Tiny,
                                                 ref::InputProfile::Cancellation}) {
                for (const int position : std::array{0, 1, 262143, 1048573}) {
                    ref::Fixture data(8, 2, position, profile);
                    data.launch(device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                    failures += data.verify("normalized_rope_kv_append numeric position=" +
                                            std::to_string(position));
                }
            }
            ref::Fixture scalar_case(8, 2, 93, ref::InputProfile::Varied, 3.0e-5F, 10'000.0F);
            scalar_case.launch(device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            failures += scalar_case.verify("normalized_rope_kv_append explicit epsilon/theta");
        }
        failures += replacement_case(device.stream);
        if (failures != 0) { return 1; }
        std::cout << "normalized_rope_kv_append: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "normalized_rope_kv_append: " << error.what() << '\n';
        return 1;
    }
}
