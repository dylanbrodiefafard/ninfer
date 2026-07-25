# Linear Low-Precision Compute Policy Memo

## Status and scope

This document records the current design decision for extending the existing `linear` Op from its
current BF16-activation implementations to INT8 and NVFP4 Tensor Core implementations. It is an
architecture memo, not an implementation plan, task list, compatibility promise, or statement that
the described kernels have already been implemented or qualified.

The decision applies to the semantic `linear` Op and is also the intended model for fused semantic
Ops that own a linear projection, such as `linear_add` and `linear_swiglu`.

## Decision

NInfer retains one semantic Op named `linear`. It does not introduce parallel entry points such as
`linear_a16`, `linear_a8`, or `linear_a4`.

The public activation and output remain contiguous BF16 tensors. A8 or A4 activation operands are
formed privately by the selected execution leaf, allowing activation quantization to remain fused
or on chip. Quantized activations do not become public `Tensor` values merely because a production
kernel uses an integer or FP4 Tensor Core instruction.

The caller explicitly supplies, or binds into an immutable execution plan, a low-precision
permission policy:

```cpp
enum class LinearPolicy {
    A16Only,
    AllowA8,
    AllowA4,
};
```

The policy has no implicit default. It is selected by the registered target execution profile, not
by an end-user runtime heuristic.

The policy is permission, not a demand:

| Policy | Permitted activation compute |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16 or A8 |
| `AllowA4` | A16 or A4 |

`AllowA8` therefore means that an A8 route may be used when it is numerically qualified and faster
for the exact registered problem. It does not require decode or small-T kernels to quantize their
activations. `AllowA4` has the corresponding meaning for an A4 route.

## Stable mathematical contract

All policies retain one public mathematical oracle:

```text
out[n,t] = BF16(
    sum over k of decode(weight[n,k]) * BF16(x[k,t])
)
```

The persistent weight format defines `decode(weight)`. Internal activation quantization, Tensor
Core operand packing, accumulator representation, split policy, and reduction order belong to the
production execution profile. They do not create a second public activation tensor or a second
mathematical reference.

Each permitted compute profile is qualified directly against the common oracle with its registered
numerical criterion. An A16 fallback used under `AllowA8` or `AllowA4` must also be qualified
directly under that policy; it is not accepted solely from an assumption that a wider internal
format must be more accurate.

## Policy versus resolved plan

The externally selected policy and the internally resolved plan are distinct:

```text
policy       = the low-precision error/performance permission
resolved plan = the exact activation compute and kernel schedule
```

A resolved plan records at least:

```cpp
struct LinearPlan {
    ActivationCompute actual_compute; // A16, A8, or A4
    KernelSchedule schedule;
    std::size_t workspace_bytes;
};
```

Plan resolution selects among complete kernel candidates. It does not first select an activation
width and then attempt to find a suitable kernel. Candidate identities include their actual
activation compute, for example:

```text
Q4DecodeA16
Q4SmallTA16
Q4GemmA16
Q4GemmA8

W8DecodeA16
W8SmallTA16
W8GemmA8

Nvfp4DecodeA16
Nvfp4SmallTA16
Nvfp4GemmA4
```

For decode and small T, the winning candidate may therefore remain A16 even under `AllowA8` or
`AllowA4`. A low-precision candidate is admitted into a production route only where it both passes
its numerical qualification and beats the best registered A16 candidate for the relevant
shape/token regime. This avoids activation quantization when it provides no performance benefit.

Resolution is deterministic for the registered weight format, shape, token regime, target, and
policy. The resulting plan is fixed before CUDA Graph capture. Benchmarks and diagnostics may
report or force a particular private candidate, but forcing a kernel is not part of the product
`linear` contract.

## Closed legal combinations

The initial legal policy matrix is:

| Persistent weight format | Legal policy |
|---|---|
| BF16 | `A16Only` |
| `Q4G64_F16S` | `A16Only`, `AllowA8` |
| `Q5G64_F16S` | `A16Only`, `AllowA8` |
| `Q6G64_F16S` | `A16Only`, `AllowA8` |
| `W8G32_F16S` | `A16Only`, `AllowA8` |
| `NVFP4G16` | `A16Only`, `AllowA4` |

Other combinations are unsupported rather than inferred. In particular, the policy does not
construct a Cartesian product such as W8+A4, NVFP4+A8, or BF16+A8.

NVFP4 activation calibration data remains associated with the NVFP4 weight or its immutable bound
execution view. It is not an additional per-call `linear` argument and does not turn the BF16 input
into a public quantized tensor.

## Consequences

This decision preserves all of the following:

- one stable semantic `linear` name and BF16 input/output boundary;
- explicit caller authorization for the wider A8/A4 numerical envelope;
- private, potentially on-chip activation quantization;
- A16 decode and small-T paths when low-precision Tensor Core execution has no benefit;
- deterministic, inspectable plan selection rather than an undocumented precision switch;
- one mathematical oracle across A16, A8, and A4 production routes.

It deliberately does not promise that a policy name identifies the exact hardware instruction used
for every token extent. A policy controls the permitted numerical implementation set; the resolved
plan identifies the actual compute path and kernel.
