# GLM-5.3-Flash on 2× RTX PRO 6000: where decode and prefill time actually goes

Written 2026-10-30 against recipe commit a88386c and engine tree `.src/tf` at 9246a6743
(f350cd3e + 66 patch commits).
This is an analysis document. No engine code was changed to produce it.
Source citations are relative to `.src/tf/zig/` (CUDA kernels under `kernels/cuda/`); patches and
recipe scripts are at the repository root.

## Baseline (from `docs/handoff.md`, production, same-slot measurements)

| Measure | Now |
| --- | --- |
| Prose decode | ~170 tok/s |
| Code decode | ~277 tok/s |
| File-edit decode | ~514 tok/s |
| Plain decode step | 7.59 ms |
| Prefill 16K–64K prompt | 5,470–5,553 tok/s |

The goal of this document is to say which of those numbers has headroom, how much, and what it costs
to get it — with a file and line for every claim, so the next agent does not have to re-derive it.

## Method, and what not to expect

I read the two hot paths as they are in the tree: the decode MoE experts
(`kernels/cuda/exl3_grouped.cuh`, `kernels/cuda/exl3_decode.cuh`, `kernels/cuda/exl3_experts.cu`,
`src/cuda/exl3_experts.zig`), the prompt experts and their dispatch
(`kernels/cuda/exl3_prompt.cuh`, `src/cuda/exl3_prompt.zig`), the chunk driver
(`src/families/glm/cuda_engine.zig`), and the GLM attention (`kernels/cuda/glm_glue_attn.cuh`).

Two things are already true and are worth saying before any recommendation, because they close off
the obvious ideas:

* **The prompt expert GEMM is already a pipelined tensor-core kernel.** It stages B through shared
  memory with 16-byte `cp.async` in a 3-stage ring and feeds the MMA with `ldmatrix`
  (`exl3_prompt.cuh:19`, `:28` `TBK = 32, TSTAGES = 3`, pipeline at `:88–95`, `ldmatrix4` at `:114`).
* **GLM attention's projections are tensor-core; its chunk body is not.** Patch 0041 moved the
  absorb/expand projections to MMA (+6.6% prefill), but the chunk kernels' QK^T / online softmax / PV
  are still SIMT fp32 `__fmaf_rn` (`glm_glue_attn.cuh:52` "8 warps: online softmax by 32 keys",
  `:266`, `:352–357`, `:439–441`). That is a real lever, not an obvious one already taken — see P6.

So in the experts, "add cp.async / use tensor cores" is not available as advice here. The remaining
headroom is in *occupancy*, *weight traffic per row*, *split-K traffic*, and *rank balance* — four
different levers, listed below with the arithmetic that says how much each is worth.

---

## Decode

The decode MoE projection is `tf_exl3_grouped_{0,1,2}` (`exl3_experts.cu:19–43`), all instantiating
`grouped_body` with `GROUPED_LOAD = tf_exl3::AHEAD` (`exl3_experts.cu:14`), i.e. the one-tile
load-ahead variant of `enum Load` (`exl3_grouped.cuh:26–27`). Weight words are read with
`ld.global.nc.L1::no_allocate.u32` (`exl3_decode.cuh:54–57`) through `ldg_lane_words`
(`exl3_decode.cuh:63`).

The real shape makes one suspicion checkable, and rules it out. With `dims 4096` and `width 2048` —
the engine's names for `text_config.hidden_size` and `text_config.moe_intermediate_size` —
`n_routed_experts 288`, `num_experts_per_tok 8` and `num_hidden_layers 45` (`top_k = 8` in the engine
at `cuda_exl3.zig:222`), and `quantization_config.avg_bits 3.5039` (`.state/model-config.json`), one token
reads 8 × 3 × 4096 × 2048 × 0.438 B ≈ **88 MB per layer, 3.97 GB
across 45 layers**. Split over two ranks and spent over a 7.59 ms step, that is **≈260 GB/s per GPU —
15% of the 1.79 TB/s GDDR7 peak**. Decode is not short of bandwidth; it is short of *outstanding work
per SM*. That is what makes occupancy (D1) and load depth (D3) the levers here, and it means a change
that only cuts bytes cannot collect the gain its arithmetic promises.

### D1 — Occupancy is the ceiling, and it is set by two things we control

The contract's own measurements file records the compiled kernels at **128–158 registers and 32,832 B
of static shared memory per block** (`src/core/contract/measurements/pending_133.zig:4–6`, entries for
`tf_exl3_grouped_0/1/2`).

At 256 threads a block, 128 registers × 256 = 32,768 registers is exactly half the 65,536 register
file: **2 blocks per SM**. That is 512 threads, 16 warps of a 64-warp budget, on a kernel whose whole
job is to wait on weight loads. The static shared figure (32,832 B) is a second, independent cap once
registers come down.

Where the shared memory goes is visible: `__shared__ __align__(16) float red[W][RR][NT * 16]`
(`exl3_grouped.cuh:205`) — a per-warp float staging buffer the size of the whole tile output, and a
second, `16`-row-tall one in the other body at `:114` (`red[W][16][NT * 16]`). It exists so that W
warps' partial sums can be folded by a strided serial loop over `w = 1..W`
(`exl3_grouped.cuh:157–158` and `:257–258`).

**Opportunity:** fold the split-K partials with warp shuffles and atomics on the output instead of a
`[W][RR][NT*16]` shared array. It is worth stating the tension plainly, because it is easy to get
wrong: **shared is not the binding cap today, registers are.** At 256 threads, 4 blocks/SM needs ≤64
registers per thread (`65,536 / (4 × 256)`), half of today's 128 — at 96 registers the cap is still 2
blocks/SM. So the budget freed by removing `red[][]` buys one of two things, not both: a deeper
load-ahead at 2 blocks/SM (D3), or a deliberate 64-register/4-block variant. They should be built and
measured as alternatives.

**Expected gain:** +10–30% on the decode MoE stage, the size depending on which of the two wins. This
is the single largest decode lever in the kernel itself. **Risk:** medium — real kernel surgery, and
the reduction order changes. **Gate:** the project's own rule (handoff, *Rules*) says a numerics change
is judged by NLL against the null spread, not by hashes; a pure occupancy change with an unchanged
summation order should be hashed exactly, so aim for that first.

### D2 — Split-K partials are written to global memory and folded by the epilogue

`plan()` (`src/cuda/exl3_experts.zig:46–50`) picks the *first* candidate that divides the k extent,
and the candidate list starts at **4 for gate/up** (`const cands = [_]usize{ if (gateup) 4 else 1, 2,
1 }`, `:47`) — so for any k that is a multiple of 256 the gate/up path runs four splits by default.
Every split block writes its own partial to global `z`
(`exl3_grouped.cuh:158`, `:258`); the scratch is sized `2 * g_sk * width` floats for gate+up and
`d_sk * dims` for down (`src/cuda/exl3_experts.zig:75`, `:100`), and the epilogue is handed `g_sk` so
it can fold the slices (`src/cuda/exl3_experts.zig:236`).

At `sk ≥ 2` each of the `sk` partials is written to global and read back by the epilogue, so that
output's DRAM traffic is `2 × sk ×` its size — pure overhead on a 7.59 ms step. Two options: a
cooperative in-kernel fold (device-wide barrier or an
atomic-add epilogue) so one partial per element is written, or a split policy that prefers `sk = 1`
when the tile already has enough parallelism from `items` × `nblocks`.

**Expected gain:** 3–8% of the MoE stage at the split counts actually chosen. **Risk:** low-medium.
**Gate:** summation order changes → NLL per the *Rules* section, not hashes.

### D3 — Load-ahead is one tile deep, against the prompt kernel's three

`AHEAD` in the decode path means the next *one* tile's words are in flight (`exl3_grouped.cuh:26–27`),
2–3 32-bit words per lane at a time (`exl3_decode.cuh:63` and its `lane_words<K2>()`). The prompt
kernel runs a 3-stage ring (`exl3_prompt.cuh:28`, `:88–95`) because at three stages the load latency
is covered and at one it is not.

Deepening this in registers is the right move *only after* D1 frees registers — the two compete for
the same budget, which is why they belong in one change and not two.

**Expected gain:** +5–10% decode combined with D1. **Risk:** low once D1 lands. **Gate:** exact.

### D4 — Rank 0's slower GPU is ~16% of the step, and it is a scheduling choice

Handoff records that GPU0 drives the monitors, clocks about 10% lower, and that **rank 0 waits about
1.2 ms per token** for it. Against a 7.59 ms plain step, that is **~16% of the whole step spent
waiting on an imbalance we created** by sharding experts evenly across two unequal GPUs.

The fix is not hardware and not numerics: shard by measured throughput — give rank 0 fewer experts,
e.g. 46/54 rather than 50/50, with the count taken from config instead of an even division. The
constraint the user has already set is *do not move the monitors*; this works within it.

**Expected gain:** up to ~10–15% decode, and it also lifts prefill for the same reason.
**Risk:** medium — per-GPU memory must still fit, and 4 streams share the pool, so the imbalance has
to hold under contention. **Gate:** exact hashes (it changes which rank owns which expert, not any
arithmetic).

### D5 — The drafter is already carrying prose

514 tok/s on file edits against 277 on code and 170 on prose says the DFlash2 accept rate, not the
target step, is what separates those rows. That is an acceptance-rate problem (drafter quality and
tap freshness, patches 0013/0036–0040), not a kernel problem, and it is out of scope for this pass —
noted here so nobody re-opens the kernels to chase prose tok/s.

---

## Prefill

`prefillChunk` (`src/families/glm/cuda_engine.zig:447–470`) is the driver: indexer split at
`:447–448`, `prompt_pair.prefill` at `:451`, the profiler banded by row count at `:459–460`. The chunk
size comes from `fw.prefill_rows` (`:449–451`), and the profiler bands by row count so a chunk size
change is measurable from the existing log without new instrumentation.

`rotate` is dispatched per chunk in two forms: the shared-input branch launches `k.rotate` with
`grid = {rows, dims/128, 1}` at 32 threads (`src/cuda/exl3_prompt.zig:100`), and the general branch
launches `e.rot_in` with `grid = {pairs, dims/128, 2}` (`:104`), where `pairs = rows × slots` (`:96`).
Either way it writes the rotated activation `sc.xg`, which `gateup` then reads as `X0`. So each chunk
writes and re-reads `rows × dims × 2 B` per layer through global memory before any expert work starts.

At 5,470–5,553 tok/s a 2048-row chunk — the engine default (`cuda_state.zig:19`) — is 0.372 s. Where does
it go?

### P1 — Rows per block sets weight traffic, and weight traffic is what prefill costs

`exl3_prompt.cuh` streams B once per *item* against a **64-row** tile of A (`TPM = 64`,
`exl3_prompt.cuh:27`), staged through shared at `:72–76`, and the router splits each expert's rows into
items of at most `tile = 64` rows (`exl3_prompt.zig:10`, `:106`; `grouped.zig:15–17`,
`maxItems = min(pairs, experts) + pairs / tile`). So the weight traffic of a chunk is
`Σ_experts ceil(rows_e / 64)` full reads of that expert, not a fixed one — and **whether `TPM` matters
therefore depends on the chunk size**, which is the part that has to be computed rather than assumed.

At this model's shape one clean read of every expert in a layer is 288 × 3 × 4096 × 2048 × 0.438 B ≈
**3.2 GB**, and a 45-layer sweep is **≈143 GB**. The engine's chunk is 2048 rows (`cuda_state.zig:19`),
i.e. 2048 × 8 = 16,384 pairs over 288 experts — **57 rows per expert on average**, and the mean sits
just under the 64-row tile. That does not make the floor free: routing is uneven (sd ≈ 7.4 at uniform
routing, so the counts straddle 64), and under that model the plan costs **~332 expert reads, 3.7 GB per
layer, ~165 GB per sweep — about 15% above the floor**, with every spill a full extra read of that
expert's weights. `TPM` 128 removes the spill entirely at this chunk size (`ceil(57/128) = 1`), taking
the sweep from ~165 to ~143 GB.

That also corrects the obvious next step: at 4096 rows `rows_e ≈ 114`, so the item count becomes **2**
for nearly every expert, the sweep rises to ~286 GB, and bytes per row stay flat at ~70 MB. Raising
`prefill_rows` alone buys only per-chunk fixed cost, not traffic. The order that pays is **P2 first,
then `TPM` 128 at 4096 rows**, where `ceil(114/128) = 1` brings bytes per row back to ~35 MB — half.

Note also that the comment above `chunk_rows` still says "4,096 with PR #1576" (`cuda_exl3.zig:223`)
while the constant it aliases is 2048; the comment is stale, and anyone sizing work off it will be
sizing against a chunk the engine does not run.

And the ceiling on all of this: 143 GB over the ~0.372 s that a 2048-row chunk takes at the measured
5,470–5,553 tok/s is **≈384 GB/s aggregate, ~11% of the 2 × 1.79 TB/s peak** (~12% on the ~165 GB
plan). As with decode, the stage is not bandwidth-bound at the rate it achieves, so the byte arithmetic
is an upper bound on the gain, not a promise. A `dram__bytes_read.sum` counter would separate "traffic is
larger than the floor" from "the kernel cannot issue enough loads", and that single number decides
whether P1/P2 or P6/latency work is the better place to spend the next week.

**Opportunity:** `TPM` 128, worth most paired with a 4096-row chunk, so a B tile is amortised over 128
rows instead of 64. **Expected gain:** a few % at today's 2048 rows from the 15% spill it removes,
rising to +25% at 4096 if the stage turns out to be traffic-bound. **Risk:** medium — the extra A staging
competes for the same shared budget as the 3-stage B ring. **Gate:** exact if only tiling/chunking
changes.

### P2 — Chunk rows is a knob, and the cheapest experiment on the list

`fw.prefill_rows` flows straight into the chunk loop (`cuda_engine.zig:449–451`) and the profiler
already bands by row count (`:459–460`), so an A/B at 2048/4096 needs no kernel work at all. Batching
only → **hashes must match exactly**, which makes this the one prefill idea that can be landed and
trusted on a hash alone. **Do this first.** Expected gain is small on its own — the P1 arithmetic has
weight bytes per row flat across 2048 → 4096 — so what it buys is per-chunk fixed cost (allocation,
launch, rank exchange), and its real value is as the control for the `TPM` 128 experiment, which only
means anything at 4096 rows.

### P3 — W8A8 is already built, already measured, and parked

Handoff patch 0064 added the W8A8 attention matmuls and the ABSORB variant: **+10% alone, +19% with
ABSORB** on prefill, then parked on numerics risk. That is the largest fully-written prefill gain in
the tree and it needs no new code — it needs a `glmeval` gate and a decision. Under the *Rules*
section that means NLL against the null spread, not hashes. **This is the user's call, not a kernel
task.**

### P4 — Chunk overlap

`prompt_pair.prefill` (`cuda_engine.zig:451`) plus the paired-chunk machinery of patches 0049–0063
already overlap the two ranks. The next step is double-buffering chunks so the rank exchange for
chunk *n+1* overlaps the attention of chunk *n*. **Expected +3–6%; risk medium** (stream and pool
accounting, 4 streams share one pool).

### P5 — Fold `rotate` into the gateup A-load

`gateup` already stages A into shared through `cp_async16` (`exl3_prompt.cuh:72–76`, `XR` rows from
`X0`/`X1`). Applying the Hadamard rotation during that staging deletes a full global write+read of
`rows × dims × 2 B` per layer plus a launch of `pairs × dims/128 × 2` one-warp blocks
(`src/cuda/exl3_prompt.zig:104`, `block = 32`). **Expected +2–5% prefill; risk medium** (the rotation
has to match the existing `rotate` bit-for-bit or the result changes — check against `rotate`'s own
test).

### P6 — The chunk attention body is still SIMT fp32

Confirmed above: `glm_glue_attn.cuh` runs QK^T, the online softmax (`:266`) and PV (`:352–357`,
`:439–441`) as scalar `__fmaf_rn` in 8 warps of 32 keys per block (`:52`), at 11–13 TFLOPS and 1 block
per SM — where the same 0041 treatment that moved the projections to MMA bought **+6.6% prefill**
(`glm.md:86`). This is the remaining attention lever and it is mechanical, not research: bf16
`mma.m16n8k16` with the same fp32 sum-order discipline 0041 already established.
**Expected +5–10% prefill at 16K, more at 64K** where the KV stream grows linearly. **Risk:** medium —
bf16 operands change rounding, so it gates on NLL, not hashes. Doing it after P2/P4 means measuring
against a chunk size already known to be best.

---

## Ranked

| # | Change | Est. gain | Risk | Effort | Gate |
| --- | --- | --- | --- | --- | --- |
| 1 | **P2** `prefill_rows` A/B | 0–3% prefill; enables P1 | none | config | exact hash |
| 2 | **D4** expert shard 46/54 by clock | 10–15% both | medium | config+sched | exact hash |
| 3 | **D1+D3** occupancy: kill `red[][]`, deeper ahead | 15–30% decode | medium | kernel | exact, then NLL |
| 4 | **P1** `TPM` 128 | a few % today (removes a ~15% spill); up to 25% at 4096 | medium | kernel | exact |
| 5 | **P3** W8A8+ABSORB (already written) | 10–19% prefill | numerics | `glmeval` | NLL |
| 6 | **D2** split-K fold without global partials | 3–8% decode | low-med | kernel | NLL |
| 7 | **P5** rotate folded into A staging | 2–5% prefill | medium | kernel | exact |
| 8 | **P4** chunk double-buffering | 3–6% prefill | medium | kernel | exact |
| 9 | **P6** chunk attention body to MMA | 5–10% prefill | medium | kernel | NLL |

Rows 1 and 2 cost no kernel work and are worth ~10–18% between them; do those before touching a `.cuh`,
and land row 1 before row 4, which is only meaningful at 4096 rows. Row 3 is the biggest single kernel
change. Row 5 is a decision, not a task. Row 9 is the largest prefill item with a measured precedent
behind it.

## How these estimates were made, and how much to trust them

D1's occupancy arithmetic is exact — registers and shared bytes are recorded in the tree
(`pending_133.zig:4–6`) and the block/SM consequence follows from the 65,536-register file. D4's 16%
is `1.2 ms / 7.59 ms` from two numbers in handoff. P1's range is the widest and the least certain: the
byte arithmetic gives a **floor of ~143 GB per chunk sweep, and ~165 GB once the uneven routing at the
64-row tile is modelled**, which the ~0.372 s that a chunk takes at the measured prefill rate already
burns through at only ~11–12% of peak — so time is going somewhere other than DRAM, and the traffic
numbers are an upper bound on what removing them can buy.

That is testable in one run, and it decides the order: profile a single chunk with
`ncu --metrics dram__bytes_read.sum` (or the existing profiler's memory counters). Near 143 GB → the
spill is absent too, `TPM` 128 buys nothing, and if the stage still runs at ~11% of peak it is
latency-bound like decode, so P6 and the prompt-side occupancy work lead instead. Near the modelled
~165 GB → the spill is real and P1 pays its few %, but no more. Materially above both → traffic is being
wasted somewhere the model does not see, and that is where the next week goes.

## What is already done, and should not be proposed again

* `cp.async` + `ldmatrix` tensor-core pipelines in prompt experts, and in GLM attention's *projections*
  (0041) — but not yet in the chunk body (P6).
* Paired prompt chunks and the rank exchange (0049–0063); the indexer split at
  `cuda_engine.zig:447–448`.
* DFlash2 speculative decode, at 514 tok/s on file edits (0013, 0036–0040).
* Per-kernel profile marks — `.mtp`, `.commit`, `.rotate` are already spanned in the decode step and
  the chunk driver, so a re-measure needs no new instrumentation.
