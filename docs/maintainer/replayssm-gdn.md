# ReplaySSM: raw-input replay of GDN speculative state

This document discusses ReplaySSM for Gated DeltaNet (GDN) in short-window speculative decoding:
target verify keeps the original token-by-token recurrence, but does not save the full recurrent
state at every verify position; it records only the raw inputs that drive the state transitions.
Once the final accepted length is determined, the accepted prefix is replayed in order from the
committed checkpoint to obtain the state for the next round.

The key question here is not only "can the state be rebuilt from the records". The GDN recurrence
can be written in several algebraically equivalent forms, but different forms change the
normalization, reduction, multiply-add association, Tensor Core precision, and cast boundaries.
State is a value that persists across rounds; even a small reconstruction error enters the next
round and keeps propagating. The core requirement of this technique is therefore:

> The replay fold must execute the same finite-precision state transition as the verify
> recurrence, not merely evaluate an equivalent formula over the reals.

This document covers, in order, the mathematical definition of the state and the record,
accepted-prefix replay, the sources of floating-point drift, the conditions for a closed-loop
bitwise clone, causal-conv history, and the space and compute characteristics of Qwen3.6 under
short windows.

---

## 1. Problem: speculative verify needs selectable state prefixes

### 1.1 Size of the GDN state

For each GDN layer and value head, the recurrent state is an FP32 matrix

\[
S\in\mathbb{R}^{V\times K}.
\]

Qwen3.6 uses \(K=V=128\). The state of one value head contains 16,384 FP32 elements, that is,
64 KiB. Multiplied by all GDN layers and value heads, one full recurrent state image is:

| Model | GDN layers | value heads | One recurrent state |
|---|---:|---:|---:|
| Qwen3.6-27B | 48 | 48 | 144 MiB |
| Qwen3.6-35B-A3B | 30 | 32 | 60 MiB |

### 1.2 Snapshot baseline

Suppose one target verify processes \(T\) inputs and, starting from the committed state \(S_0\),
sequentially obtains

\[
S_1,S_2,\ldots,S_T.
\]

Only one of these prefixes, \(S_m\), is finally committed. The direct way to support rollback is
to save every state:

~~~text
S0 ── input 1 ──> S1 ── input 2 ──> ... ── input T ──> ST
                    │                    │                  │
                    └──────── save full state trajectory ─┘

final commit length = m: select Sm
~~~

This writes one \(V\times K\) FP32 matrix for every verify position. Each additional window
column adds one more full state of capacity and write traffic.

### 1.3 Raw-input ReplaySSM

Raw-input ReplaySSM keeps only:

\[
\text{committed checkpoint }S_0
\quad+\quad
R_1,R_2,\ldots,R_T,
\]

where \(R_t\) is the compact record of the \(t\)-th state transition. Verify still produces all
outputs, but does not write \(S_1,\ldots,S_T\) to persistent storage. Once the final \(m\) is
known, it computes

\[
S_m=F_{\mathrm{fp}}\left(
F_{\mathrm{fp}}\left(
\cdots F_{\mathrm{fp}}(S_0,R_1),R_2
\right)\cdots,R_m
\right),
\]

and publishes only this one state.

Here \(F_{\mathrm{fp}}\) deliberately denotes the actual finite-precision transition, not an
abstract real-number formula. Later sections explain why this distinction determines whether the
reconstructed state can match the verify trajectory.

---

## 2. GDN recurrence and the raw transition record

### 2.1 Logical order corresponding to the finite-precision path

The layer and value-head indices are omitted below. Let

\[
q_t^{raw},k_t^{raw}\in\mathbb{R}^{K},
\qquad
v_t\in\mathbb{R}^{V}.
\]

Query and key first undergo L2 normalization:

\[
\bar q_t=
\frac{q_t^{raw}}
{\sqrt{\sum_i(q_{t,i}^{raw})^2+\epsilon}},
\qquad
\bar k_t=
\frac{k_t^{raw}}
{\sqrt{\sum_i(k_{t,i}^{raw})^2+\epsilon}}.
\]

Let

\[
\alpha_t=\exp(g_t).
\]

One GDN transition can be written in the following order:

\[
S_t^{decay}=\alpha_tS_{t-1},
\]

\[
r_t=S_t^{decay}\bar k_t,
\]

\[
u_t=\beta_t(v_t-r_t),
\]

\[
S_t=S_t^{decay}+u_t\bar k_t^{\mathsf T},
\]

\[
y_t=c\,S_t\bar q_t,
\qquad
c=\frac{1}{\sqrt{128}}.
\]

Merging these gives the common compact form:

\[
u_t=\beta_t
\left(v_t-\alpha_tS_{t-1}\bar k_t\right),
\]

\[
S_t=\alpha_tS_{t-1}+u_t\bar k_t^{\mathsf T}.
\]

The two sets of formulas are identical over the reals; the first set also shows the operation
boundaries that must stay consistent in the actual transition: first decay the state, then read
the correction from the decayed state, and finally apply the rank-one update.

### 2.2 Grouped q/k heads

If \(H_v\) value heads share \(H_{qk}\) q/k heads, let

\[
G=H_v/H_{qk},
\qquad
h_q=\lfloor h/G\rfloor.
\]

Value head \(h\) uses q/k head \(h_q\). Each value head has its own \(S\), \(v\), \(g\), and
\(\beta\), while the same group shares the raw \(q/k\).

### 2.3 Sufficient inputs for a state transition

Given \(S_{t-1}\), computing \(S_t\) requires only

\[
R_t^{raw}=
\left(k_t^{raw},v_t,g_t,\beta_t\right).
\]

The role of each field is:

- \(k_t^{raw}\): re-executes the same normalization, state read, and rank-one write as verify;
- \(v_t\): recomputes the corrected value;
- \(g_t\): forms the decay through the same \(\exp\) path;
- \(\beta_t\): forms the correction;
- \(q_t\): used only for this round's output readout and does not change the state, so it does not
  enter the replay record.

At the usual Qwen3.6 numerical boundaries, raw \(k/v\) are BF16 represented values, \(g/\beta\)
are FP32 represented values, and the checkpoint is FP32. The raw record is a lossless side copy of
these actual transition inputs, not something derived again from the hidden state or the upstream
projections.

---

## 3. Accepted-prefix replay

### 3.1 Verify input, output, and commit length

Suppose the speculative drafter provides \(D\) draft tokens. Target verify processes

\[
T=D+1
\]

inputs:

~~~text
input 1       = the anchor not yet processed at the start of the round
input i + 1   = draft i, 1 <= i <= D
~~~

If the first \(A\) drafts match the target, the target licenses

\[
p=A+1
\]

output tokens: the \(A\) accepted drafts plus one correction/bonus token.

The inputs that must advance the state are exactly the first \(p\) verify inputs, that is, the old
anchor plus the first \(A\) accepted drafts. The final correction/bonus token has not yet been
executed as an input; it becomes the next round's anchor.

The final output boundary may keep only the first \(m\) of the licensed outputs, so the commit
length of the state is

\[
0\le m\le p.
\]

For example, with \(D=5,A=3\), \(p=4\). The four candidate state transitions are "old anchor +
the first three accepted drafts", and the four outputs are "three accepted drafts + correction".
If the final \(m=2\), only "old anchor + the first accepted draft" is replayed; the second output
is the new pending anchor.

### 3.2 Verify: producing outputs and raw records

Verify starts from the committed checkpoint and executes the original recurrence in order:

~~~text
S_verify <- S0

for t = 1 .. T:
    q <- normalize(q_raw[t])
    k <- normalize(k_raw[t])
    alpha <- exp(g[t])
    S_verify <- alpha * S_verify
    u <- beta[t] * (v[t] - S_verify * k)
    S_verify <- S_verify + outer(u, k)
    y[t] <- scale * S_verify * q
    record[t] <- (k_raw[t], v[t], g[t], beta[t])
~~~

Here \(S_{verify}\) is the transient trajectory needed to produce the outputs of the current
window. After verify ends, it is not published as the committed state; the persistent checkpoint
\(S_0\) stays unchanged.

### 3.3 Fold: replay only the accepted prefix

Once the final \(m\) is known, the fold starts from the same \(S_0\):

~~~text
S_fold <- S0

for t = 1 .. m:
    k <- normalize(record[t].k_raw)
    alpha <- exp(record[t].g)
    S_fold <- alpha * S_fold
    u <- record[t].beta * (record[t].v - S_fold * k)
    S_fold <- S_fold + outer(u, k)

publish S_fold
~~~

When \(m=0\), the state is strictly unchanged. The rejected suffix \(R_{m+1:T}\) is never read by
the fold.

Within a single head there are still up to \(m\) sequential transitions, but different layers,
value heads, and batch rows are independent of each other. The parallel width of Qwen3.6 comes
from 30/48 GDN layers, 32/48 value heads, and multiple active rows, while the single-row token
loop has at most 6 or 16 iterations.

### 3.4 Accepted-prefix correctness

Denote the finite-precision transition used by verify as \(F_{\mathrm{fp}}\):

\[
S_t^{verify}=
F_{\mathrm{fp}}(S_{t-1}^{verify},R_t),
\qquad
S_0^{verify}=S_0.
\]

If the fold uses exactly the same \(F_{\mathrm{fp}}\):

\[
S_t^{fold}=
F_{\mathrm{fp}}(S_{t-1}^{fold},R_t),
\qquad
S_0^{fold}=S_0,
\]

then induction on \(t\) directly gives:

\[
S_t^{fold}=S_t^{verify},
\qquad 0\le t\le m.
\]

Therefore

\[
S_m^{fold}=S_m^{verify}
\]

can be a bit-for-bit conclusion in finite precision, not merely an equivalence over the reals. It
holds on the premise that both paths call the same deterministic floating-point transition and
consume the same record bits.

---

## 4. Numerical core: why "algebraically equivalent" still produces state drift

### 4.1 Computation paths in finite precision

For the same set of inputs, one GDN transition can be written over the reals as

\[
S_t=
\alpha_tS_{t-1}
+\beta_t\left(v_t-\alpha_tS_{t-1}\bar k_t\right)
\bar k_t^{\mathsf T},
\]

and can also be written as

\[
S_t=
\alpha_tS_{t-1}
\left(I-\beta_t\bar k_t\bar k_t^{\mathsf T}\right)
+\beta_tv_t\bar k_t^{\mathsf T}.
\]

These two expressions define the same real-valued map, but a floating-point state is the result of
a concrete evaluation program. It depends on:

- the reduction tree, sqrt, and division of the raw key normalization;
- the \(\exp\) path from \(g_t\) to \(\alpha_t\);
- the accumulation order and operand precision of \(S\bar k\);
- the order of state decay, correction, and rank-one update;
- FMA contraction, tile decomposition, and the state-store boundary.

Algebraic equivalence does not automatically imply equal represented states. Once the replay
changes any of the above, it defines a new finite-precision transition; a single-step difference
enters the committed state and becomes an input to the next round's recurrence.

### 4.2 Raw record and replay path

The raw record

\[
R_t=(k_t^{raw},v_t,g_t,\beta_t)
\]

fixes the represented inputs of the transition. The complete replay state is also determined by
the finite-precision map \(F_{\mathrm{fp}}\):

\[
S_t=F_{\mathrm{fp}}(S_{t-1},R_t).
\]

If the fold actually executes a different evaluation program \(\widetilde F_{\mathrm{fp}}\), then
even though the two correspond to the same formula over the reals,

\[
\widetilde F_{\mathrm{fp}}(S,R)
\ne
F_{\mathrm{fp}}(S,R)
\]

can still hold. The raw record fixes the identity of the inputs; verbatim replay fixes the
identity of the transition.

### 4.3 Verbatim closed-loop replay

The record and computation responsibilities of closed-loop replay are:

| Item | Requirement |
|---|---|
| raw \(v\) | Save the represented bits that the verify recurrence actually consumes |
| raw \(k\) | Save the represented bits before normalization |
| \(g\) | Save the FP32 log-decay gate that verify already obtained; do not recompute it from upstream |
| \(\beta\) | Save the FP32 correction gate that verify already obtained |
| normalization | Same epsilon, reduction tree, and sqrt/division form |
| decay | Same \(\exp\) and state-multiply order |
| correction | Recompute \(u\) from the replay state at the current position |
| rank-one update | Same operand precision, FMA contraction, and tile decomposition |
| state store | Same FP32 represented boundary |

"Closed loop" means that every corrected value is computed on the spot from the current replay
state, the raw \(v/k\), and the gate bits. This puts every step of the fold into the same
transition as verify, and makes the represented state obtained in the previous step the direct
input of the next step's correction.

SGLang observed this numerical boundary during the Kimi K3 bring-up: an early fold recomputed the
gates from a different computation path, and although the outputs still looked normal at that
time, the recurrent state had already started to drift. After switching to directly recording the
gate values produced by verify and replicating the recurrence in the fold, the state became
bit-identical to the recurrent baseline. SGLang's GDN exact fold likewise aligns the state tile,
the division-form L2 normalization, and the operation order of the verify branch.

### 4.4 How state drift propagates

First consider two trajectories that use the same represented inputs and the same real-valued GDN
formula, but whose starting states differ:

\[
\Delta S_t=\widetilde S_t-S_t.
\]

The recurrence gives

\[
\Delta S_{t+1}
=\alpha_{t+1}\Delta S_t
\left(I-\beta_{t+1}\bar k_{t+1}\bar k_{t+1}^{\mathsf T}\right).
\]

If the reconstructed transition itself also introduces a local floating-point deviation
\(E_{t+1}\), the error model becomes

\[
\Delta S_{t+1}
=\alpha_{t+1}\Delta S_t
\left(I-\beta_{t+1}\bar k_{t+1}\bar k_{t+1}^{\mathsf T}\right)
+E_{t+1}.
\]

The corresponding state readout difference is

\[
\Delta y_{t+1}=c\,\Delta S_{t+1}\bar q_{t+1}.
\]

Outputs already produced by the verify trajectory in this round are not retroactively modified by
the fold. The real problem appears in the next round: attention/KV, hidden states, and published
outputs come from the verify trajectory, while the GDN checkpoint comes from a different numerical
trajectory. Even if the output cast temporarily masks the difference within a single round, the
persistent state still participates repeatedly in correction and readout in later tokens.

The decay \(\alpha<1\) can attenuate some of the old error, but

\[
I-\beta\bar k\bar k^{\mathsf T}
\]

is a direction-dependent update, and each round may inject a new \(E_t\). Therefore "the gate
attenuates the error" cannot replace direct verification of the committed state.

### 4.5 Two levels of correctness

ReplaySSM state reconstruction should distinguish two criteria:

1. **Mathematical correctness**

   Starting from represented BF16/FP32 inputs and the FP32 \(S_0\), the output and final state
   satisfy the specified error relative to an independent FP32/FP64 GDN oracle.

2. **Finite-precision clone**

   For the same \(S_0\), the same raw record bits, and the same accepted prefix, the FP32 state
   after the fold is bit-identical, element by element, to the sequential verify/baseline.

Final text or BF16 output parity is not enough to verify a bitwise clone. Direct evidence should
cover:

- exact bit copies of the raw \(k/v\) and \(g/\beta\) records;
- the committed checkpoint being completely unchanged during verify;
- exact comparison of the final state for \(m=0,1,T\) and intermediate \(m\);
- long-chain replay composed of sequences of different accepted lengths;
- very small key norms, extreme gate values, and multiple q/k-to-value-head groups;
- the committed state remaining unchanged after the rejected suffix is overwritten.

If the fold and verify use different arithmetic paths, a numerical tolerance should be declared
against an independent oracle, and the result must not be called a bitwise clone.

---

## 5. Finite-window state of the causal convolution

A GDN block usually has a causal depthwise convolution before the recurrence. If the convolution
width is \(W\), the persistent history contains only the most recent \(W-1\) projection columns:

\[
H_0=[p_{-(W-2)},\ldots,p_0].
\]

Verify produces

\[
p_1,p_2,\ldots,p_T.
\]

When the final commit length is \(m\), the correct history is

\[
H_m=
\operatorname{tail}_{W-1}
\left(H_0\mathbin\Vert[p_1,\ldots,p_m]\right).
\]

Each verify position therefore needs to record only one column of represented projection history,
rather than saving the entire \(W-1\)-column window:

- \(m=0\): the history is unchanged;
- \(0<m<W-1\): keep some of the old columns and append the accepted columns;
- \(m\ge W-1\): use the last \(W-1\) columns of the accepted prefix.

As long as the record is the same BF16 represented column that the baseline would write into the
history, this commit is an exact gather, with no recurrent reduction or floating-point
re-association. Qwen3.6 uses \(W=4\), so one column of record is \(1/3\) of a three-column
snapshot.

---

## 6. Space and compute characteristics

### 6.1 General formulas

Let:

- \(L_g\): number of GDN layers;
- \(H_q\): number of q/k heads;
- \(H_v\): number of value heads;
- \(K,V\): key/value dimension;
- \(C_p\): causal-conv projection channels;
- \(W-1\): conv history columns;
- \(T\): verify window.

The byte count of one FP32 recurrent state is

\[
R=4L_gH_vVK.
\]

The byte count of one BF16 conv history is

\[
Q=2L_gC_p(W-1).
\]

If raw \(k/v\) are BF16 and \(g/\beta\) are FP32, the GDN record per token is

\[
P_{gdn}=
2L_g(H_qK+H_vV)+8L_gH_v.
\]

The conv column record per token is

\[
P_{conv}=2L_gC_p.
\]

The full replay log of length \(T\) is

\[
P_{record}(T)=T(P_{gdn}+P_{conv}).
\]

The window-dependent capacity of the snapshot trajectory is

\[
T(R+Q),
\]

while that of raw ReplaySSM is

\[
T(P_{gdn}+P_{conv}).
\]

### 6.2 Qwen3.6 sizes

| Model | \(L_g\) | \(H_q\) | \(H_v\) | \(K/V\) | \(C_p\) | \(W\) |
|---|---:|---:|---:|---:|---:|---:|
| 27B | 48 | 16 | 48 | 128/128 | 10,240 | 4 |
| 35B-A3B | 30 | 16 | 32 | 128/128 | 8,192 | 4 |

The corresponding per-token state/record sizes are:

| Model | recurrent image | conv history | raw GDN record | conv record | record total |
|---|---:|---:|---:|---:|---:|
| 27B | 144.000 MiB | 2.8125 MiB | 0.767578 MiB | 0.9375 MiB | 1.705078 MiB |
| 35B-A3B | 60.000 MiB | 1.40625 MiB | 0.358887 MiB | 0.46875 MiB | 0.827637 MiB |

The ratio of one raw record to one recurrent+conv snapshot is:

| Model | snapshot/position | raw record/position | Size ratio |
|---|---:|---:|---:|
| 27B | 146.8125 MiB | 1.705078 MiB | about 86.1x |
| 35B-A3B | 61.40625 MiB | 0.827637 MiB | about 74.2x |

The record capacity for typical verify windows is:

| Model and window | raw GDN records | conv records | Total |
|---|---:|---:|---:|
| 27B, \(T=6\) | 4.605469 MiB | 5.625000 MiB | 10.230469 MiB |
| 35B-A3B, \(T=6\) | 2.153320 MiB | 2.812500 MiB | 4.965820 MiB |
| 35B-A3B, \(T=16\) | 5.742188 MiB | 7.500000 MiB | 13.242188 MiB |

### 6.3 Compute shape

Raw-input replay keeps the serial recurrence of verify and adds up to \(m\) transitions at commit:

| Item | Snapshot baseline | Raw-input replay |
|---|---|---|
| verify state read | One checkpoint | One checkpoint |
| verify full-state writes | \(T\) copies | 0 |
| verify record writes | 0 | \(T\) small records |
| commit work | Select a snapshot | Replay \(m\) transitions, write one state |
| rollback | Select the corresponding snapshot | Read only the accepted record prefix |
| persistent numerical path | verify recurrence | closed-loop clone of the verify recurrence |

The token dimension in this scenario is very short: MTP has at most \(T=6\), and DFlash has at most
\(T=16\). Along a single layer, value head, and batch row, the fold has \(m\) sequential
transitions; different layers, heads, and batch rows are independent of each other. The overall
compute shape is therefore a large number of independent short recurrences. The fold compute grows
linearly with the accepted length \(m\), the record traffic grows linearly with the verify length
\(T\), and only one committed state is written in the end.

This technique is first of all a capacity and state-traffic optimization. Verify writes \(T\)
fewer large states, at the cost of writing small records and doing one extra short prefix fold
after acceptance. End-to-end latency depends on state traffic, accept length, and the available
layer/head/batch parallelism, and cannot be inferred directly from the space compression ratio.

---

## 7. Core conclusions

The state representation of GDN speculative ReplaySSM is

\[
\text{one committed checkpoint}
+
\text{one short raw transition log}.
\]

Its correctness depends on the following invariants:

1. Verify produces outputs and raw records from the committed checkpoint, but does not modify the checkpoint;
2. The record stores the raw pre-normalization \(k\), the raw \(v\), and the \(g/\beta\) bits that verify itself produced;
3. The final commit length also defines the record prefix that needs to be replayed;
4. The rejected suffix is not read by the fold;
5. The fold recomputes every corrected value in closed loop;
6. The fold and verify use the same normalization, gates, reduction, operation order, and state-store boundary;
7. The committed state is compared directly against the sequential recurrent baseline, rather than substituting output plausibility.

The raw inputs determine "what can be replayed", and the verbatim recurrence determines "whether
the replay yields the same finite-precision state". The former solves the snapshot capacity
problem and the latter prevents cross-round state drift; together they make up short-window GDN
ReplaySSM.

---

## References

- [SGLang and Miles Add Day-0 Support for Kimi K3](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support): raw-input replay, stored gates, and bit-identical state fold.
- [SGLang GDN exact fold](https://github.com/sgl-project/sglang/blob/bc285b2064c0373227cfb6ada77a37e7b8c43510/python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py): closed-loop fold aligned with the verify recurrence.
- [ReplaySSM: Cache SSM Inputs, Not State](https://tridao.me/blog/2026/replayssm/): the basic idea of caching SSM transition inputs and folding the state when needed.
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464): the mathematical source of GDN and the gated delta rule.
