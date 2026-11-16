"""Replays a lane-batched expert-cache trace through admission policies.

Input: a directory written by `ninfer_qwen3_8_flash_next_parity --mode lanes-trace` (one
concurrency C): routes.npy int32 [rounds, 48, C, W, 10], the post-prefill pools (owner_p<c>.npy,
referenced_p<c>.npy, clock_hands.npy), frequency.npy/frequency_lookups.npy,
codecs.npy [48, 512] and trace.json with the device
counters per bucket. Each round is 48 layer calls in ascending layer; a call's routed set U is the
union of its C lanes' experts, as moe_expert_cache_admit sees it.

Policies (two pools, NVFP4 and F3, as the engine; a hit always sets the CLOCK reference bit):
  clock            moe_expert_cache_admit exactly: every miss is admitted in ascending expert id,
                   CLOCK picks the victim, skipping slots of U and clearing reference bits. Its
                   counters can be checked bucket for bucket with --check-policy clock.
  second:W         admit a miss only when the same expert also missed within the previous W
                   rounds (a ghost record of its last unadmitted miss); others are read through
                   mapped host memory and leave the pools unchanged.
  gclock:K[+P]     GCLOCK: reference counters capped at K instead of bits (a hit adds one, the
                   hand subtracts one), with admission P (one of the above; default every miss).
  tinylfu:A[:ge]   a frequency filter in front of CLOCK: every lookup increments the expert's
                   counter, and all counters halve after A x (total slots) lookups. A miss takes
                   the CLOCK victim only when its counter exceeds the victim owner's (with ge, is
                   at least it); an empty victim slot always admits. A rejected miss leaves the
                   victim in place (the hand and the cleared reference bits stay advanced).

A miss costs its record bytes over PCIe whether admitted (moe_expert_cache_fill) or read through
mapped memory, so miss bytes per round are policy-comparable. --extra-bytes adds cache bytes,
re-planned with the engine's 11% NVFP4 rule, from the calibration warm start (prefill admits
nothing, so the post-prefill pools equal the warm start). The implied tokens per second are
published outputs / (other + miss bytes / bandwidth), with `other` the measured non-PCIe
round time and bandwidth defaulting to PCIe Gen5's measured 53 GB/s. Policy decisions use miss
bytes, not this implied rate; a changed admission policy can also change overlap and kernel time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPERTS = 512
LAYERS = 48
NVFP4_BYTES, F3_BYTES = 2_764_800, 1_996_800
RECORD = (NVFP4_BYTES, F3_BYTES)
NVFP4_PERCENT = 11


def plan_slots(budget: int, counts: tuple[int, int]) -> list[int]:
    """plan_expert_slots: NVFP4 takes 11% of the budget, F3 the rest, NVFP4 grows into leftovers."""

    def fit(nbytes: int, codec: int) -> int:
        return min(nbytes // RECORD[codec], counts[codec])

    nv = fit(budget // 100 * NVFP4_PERCENT, 0)
    f3 = fit(budget - nv * NVFP4_BYTES, 1)
    nv = fit(budget - f3 * F3_BYTES, 0)
    return [nv, f3]


def warm_owners(calib: np.ndarray, codecs: np.ndarray, slots: list[int]) -> list[np.ndarray]:
    """ExpertResidency's warm start: per pool the codec's nonzero-count experts in descending count
    (stable, so ties go to the lower layer-major index); the hottest takes the last slot."""
    flat, kind = calib.reshape(-1), codecs.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    pools = []
    for c in (0, 1):
        cand = [int(e) for e in order if flat[e] > 0 and kind[e] == c][: slots[c]]
        owner = np.full(slots[c], -1, np.int64)
        for rank, code in enumerate(cand):
            owner[slots[c] - 1 - rank] = code
        pools.append(owner)
    return pools


class Cache:
    """Two CLOCK pools with an optional admission filter."""

    def __init__(
        self, owners, referenced, hands, codecs, policy: str, frequency=None, lookups: int = 0
    ):
        self.owner = [[int(x) for x in o] for o in owners]
        self.ref = [[int(x) for x in r] for r in referenced]
        self.hand = [int(h) for h in hands]
        self.available = [sum(owner != -2 for owner in pool) for pool in self.owner]
        self.kind = [int(k) for k in codecs.reshape(-1)]
        self.slot_of = {}
        for c in (0, 1):
            for s, o in enumerate(self.owner[c]):
                if o >= 0:
                    self.slot_of[o] = (c, s)
        # "gclock:K+<admission>": reference counters capped at K (a hit adds one, the hand takes
        # one) in place of CLOCK bits; the admission part is clock, second:W or tinylfu:A[:ge].
        self.cap = 1
        if policy.startswith("gclock:"):
            head, _, rest = policy.partition("+")
            self.cap = int(head.split(":")[1])
            policy = rest or "clock"
        name, *params = policy.split(":")
        self.name = name
        self.round = 0
        if name == "second":
            self.window = int(params[0])
            self.ghost: dict[int, int] = {}
        elif name == "tinylfu":
            total = sum(len(o) for o in self.owner)
            self.period = int(float(params[0]) * total)
            self.ge = len(params) > 1 and params[1] == "ge"
            self.freq = (
                [0] * (LAYERS * EXPERTS)
                if frequency is None
                else [int(f) for f in frequency.reshape(-1)]
            )
            self.seen = lookups
        elif name != "clock":
            raise SystemExit(f"unknown policy {policy}")

    def victim(self, c: int, prot: set[int]) -> int:
        o, r, n = self.owner[c], self.ref[c], len(self.owner[c])
        while True:
            s = self.hand[c]
            self.hand[c] = (s + 1) % n
            if o[s] == -2 or o[s] in prot:
                continue
            if r[s]:
                r[s] -= 1
                continue
            return s

    def call(self, layer: int, routed, admitted=None) -> tuple[list[int], list[int]]:
        base = layer * EXPERTS
        codes = sorted({base + int(e) for e in routed})
        eligible = None if admitted is None else {base + int(e) for e in admitted}
        prot = set(codes)
        look, miss = [0, 0], [0, 0]
        missing = []
        for code in codes:
            c = self.kind[code]
            look[c] += 1
            hit = self.slot_of.get(code)
            if hit is not None:
                ref = self.ref[hit[0]]
                ref[hit[1]] = min(ref[hit[1]] + 1, self.cap) if self.cap > 1 else 1
            else:
                miss[c] += 1
                missing.append((c, code))
        if self.name == "tinylfu":
            for code in codes:
                self.freq[code] += 1
            self.seen += len(codes)
            if self.seen >= self.period:
                self.freq = [f >> 1 for f in self.freq]
                self.seen = 0
        for c, code in missing:
            if eligible is not None and code not in eligible:
                continue
            o, n = self.owner[c], len(self.owner[c])
            if n == 0 or (len(prot) >= self.available[c] and all(x == -2 or x in prot for x in o)):
                continue
            if self.name == "second":
                last = self.ghost.get(code)
                if last is None or self.round - last > self.window:
                    self.ghost[code] = self.round
                    continue
                del self.ghost[code]
            s = self.victim(c, prot)
            old = o[s]
            if self.name == "tinylfu" and old >= 0:
                incoming_frequency, owner_frequency = self.freq[code], self.freq[old]
                if not (
                    incoming_frequency > owner_frequency
                    or (self.ge and incoming_frequency == owner_frequency)
                ):
                    continue
            if old >= 0:
                del self.slot_of[old]
            o[s] = code
            self.ref[c][s] = 1
            self.slot_of[code] = (c, s)
            prot.add(code)
        return look, miss

    def end_round(self) -> None:
        self.round += 1


def replay(cache: Cache, routes: np.ndarray, bucket: int, root_only: bool = False):
    rounds, layers = routes.shape[0], routes.shape[1]
    flat = routes.reshape(rounds, layers, -1)
    buckets, cur = [], np.zeros((2, 2), np.int64)
    for r in range(rounds):
        for layer in range(layers):
            roots = routes[r, layer, :, 0, :].reshape(-1) if root_only else None
            look, miss = cache.call(layer, flat[r, layer], roots)
            cur[0] += look
            cur[1] += miss
        cache.end_round()
        if (r + 1) % bucket == 0 or r + 1 == rounds:
            buckets.append(cur.copy())
            cur[:] = 0
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", type=Path)
    ap.add_argument("--policies", nargs="+", default=["clock"])
    ap.add_argument(
        "--other-ms",
        type=float,
        required=True,
        help="measured non-PCIe round time at this concurrency",
    )
    ap.add_argument(
        "--extra-bytes",
        type=int,
        nargs="+",
        default=[0],
        help="cache bytes added to the traced budget (0 keeps the traced pools)",
    )
    ap.add_argument(
        "--counts", type=Path, default=Path("/home/battlefront/ai/study/calib/routing_counts.npy")
    )
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--root-only",
        action="store_true",
        help="admit only experts used by column zero; all columns still incur misses",
    )
    ap.add_argument(
        "--check-policy", help="policy whose bucket counters must reproduce the recorded engine"
    )
    ap.add_argument("--pcie-gb-s", type=float, default=53.0)
    args = ap.parse_args()
    if args.check_policy is not None and (
        args.check_policy not in args.policies or 0 not in args.extra_bytes
    ):
        ap.error("--check-policy must be selected in --policies with 0 in --extra-bytes")
    t = args.trace
    meta = json.loads((t / "trace.json").read_text())
    routes = np.load(t / "routes.npy")
    if args.root_only and routes.ndim != 5:
        raise SystemExit("--root-only requires a [rounds,layers,lanes,width,10] trace")
    frequency = np.load(t / "frequency.npy")
    frequency_lookups = int(np.load(t / "frequency_lookups.npy")[0])
    codecs = np.load(t / "codecs.npy")
    calib = np.load(args.counts)
    lanes, rounds, bucket = meta.get("lanes", 1), routes.shape[0], meta["bucket_steps"]
    slots = [meta["nvfp4_slots"], meta["f3_slots"]]
    budget = slots[0] * NVFP4_BYTES + slots[1] * F3_BYTES
    expert_counts = (int((codecs == 0).sum()), int((codecs == 1).sum()))
    engine = [np.array([b["lookups"], b["misses"]]) for b in meta["buckets"]]
    rows = []

    def report(label, buckets):
        tot = sum(buckets)
        hit = 1 - tot[1].sum() / tot[0].sum()
        mb = float((tot[1] * np.array(RECORD)).sum()) / rounds / 1e6
        outputs = meta.get("published_tokens", lanes * rounds) / rounds
        tps = outputs / ((args.other_ms + mb / args.pcie_gb_s) * 1e-3)
        print(f"C={lanes} {label:34s} hit {hit:.4f}  miss MB/round {mb:7.1f}  tok/s {tps:6.1f}")
        rows.append(
            dict(
                lanes=lanes,
                policy=label,
                hit_rate=float(hit),
                miss_mb_per_round=mb,
                implied_tokens_per_second=tps,
            )
        )
        return mb

    report("engine counters", engine)
    for extra in args.extra_bytes:
        if extra == 0:
            owners = [np.load(t / f"owner_p{c}.npy") for c in (0, 1)]
            refs = [np.load(t / f"referenced_p{c}.npy") for c in (0, 1)]
            hands = np.load(t / "clock_hands.npy")
            plan = slots
        else:
            plan = plan_slots(budget + extra, expert_counts)
            owners = warm_owners(calib, codecs, plan)
            refs = [np.zeros(s, np.uint8) for s in plan]
            hands = [0, 0]
        for policy in args.policies:
            cache = Cache(
                owners,
                refs,
                hands,
                codecs,
                policy,
                frequency if extra == 0 else None,
                frequency_lookups if extra == 0 else 0,
            )
            buckets = replay(cache, routes, bucket, args.root_only)
            if (
                extra == 0
                and policy == args.check_policy
                and (
                    len(buckets) != len(engine)
                    or not all((b == e).all() for b, e in zip(buckets, engine, strict=True))
                )
            ):
                raise SystemExit(f"{policy} does not reproduce the engine counters")
            report(f"{policy} +{extra / 1e9:.2f} GB ({plan[0]}+{plan[1]})", buckets)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
