# GLM-5.3-Flash on 2× RTX PRO 6000: where decode and prefill speed is left (OPUS v2)

**Date:** 2026-10-31. **Analysis only:** no engine code, patch, setting, container or GPU state was changed, and no
GPU slot was taken. The only requests sent to production were a one-token health probe and, later, the review
rounds with GLM that the user asked for (§10).

**Labels:** **[M]** measured (by me from logs and captures, or by the overnight agents, cited); **[C]** read in the
exact production source; **[E]** estimate. Gains are per the critical rank (rank 1 = GPU0) unless stated.

**Source read:** the exact production engine, rebuilt as upstream `f350cd3e` + `patches/engine/0001-0091` (all 91
apply cleanly with `git apply`). Line references are into that tree (`zig/src/families/glm/` = `fg/`,
`zig/kernels/cuda/` = `kc/`).

**Constraint from the user (2026-10-31):** one session must be able to reach ~985K tokens in the shared pool, and 4
streams must be available. Every item below either costs no KV memory or says how much it costs.

**Companion files:** the four analysis passes behind this report are in `opt-sm120-notes/opt/opus2/`:
`decode.md`, `prefill.md`, `spec.md` and `upstream.md`. Their claims were re-checked before use, and §9 lists where
I overruled them.

---

## 0. Summary

| # | What | Gain [E unless M] | Exact? | Memory | Effort |
|---|---|---|---|---|---|
| 1 | mHC site rework: immutable loads above the PDL waits, `hc_post` folded into the P2P gather (the Sinkhorn is already overlapped in production's `finish_split`) | −0.1 to −0.2 ms a step | yes | 0 | days |
| 2 | PDL on the ~350 decode launches still without it (KDA chain, `router_tile`, the DSA chain, `down_combine`) | −0.06 to −0.15 ms a step | yes | 0 | days |
| 3 | Indexer scores split across the two ranks (decode, long context) | −0.25 to −0.3 ms a step at ~1M, ~−0.05 at 200K | yes (exact top-k merge) | **frees** 352 B/token/GPU (+5.8% pool) | 1-2 weeks |
| 4 | Embedding table (1.18 GiB bf16) from each GPU to mapped pinned host memory | +~199K pool tokens of headroom; a few µs a step | yes | **frees** 1.18 GiB/GPU | 1-2 days |
| 5 | Keep a finished session's KV resident until another stream needs the pages (no host restore on the next turn) | −70 ms a turn at 220K, ~−0.3 s at ~985K | yes | 0 | days |
| 6 | Prefill fill pieces: A/B `TF_GLM_OVERLAP_PIECES` 1/2/3 (zero code), then a whole-chunk shared expert and single-launch fills | +1.5-4% prefill | yes | 0 | hours → days |
| 7 | Prompt sparse attention: stop at the last live 32-key tile (~19% of tile iterations are dead), vectorised fp8→bf16 stores | +2-2.5% prefill | yes | 0 | low |
| 8 | glm_mm W8A16 prompt GEMM: a 128×128 block tile (today 128×64, L2/smem-bound at ~145-160 TF) | +3-7% prefill | yes (same MMA chains) | 0 | medium-high; NCU first |
| 9 | Prompt scratch: smaller indexer score blocks, fused dense-prompt attention, unused buffers (M1-M5); later one aliased region (M6) | **frees** ~753 MB/GPU (~118K tokens); M6 ~700 MB more | yes | frees | low → medium |
| 10 | Coupled drafting (already in the code) with a fitted selector temperature; size it by offline replay first | +0-6% sampled decode | outputs identical | 0 | low |
| 11 | Memory clock P1 → P0 (`nvidia-smi -cc 1` to test; **not** `-lmc`, see §7) | up to +4.5% decode | yes | 0 | user, root |
| 12 | Asymmetric power caps (GPU0 440 W / GPU1 360 W) | the largest prefill lever, ≤ ~+4.7% | yes | 0 | user, root |
| 13 | Refined W8A8, chunked KDA prefill, fp8 index keys (q4 dense removed by the user for quality) | +5-6% / +4-7% prefill | **no**: the quality gate decides | frees memory (fp8 keys) | gated |

Together the exact engine items are worth roughly:
- **decode:** 0.4-0.65 ms of the 7.05 ms plain step (to ~6.4-6.7 ms, conditional; corrected after the review
  rounds), plus 0.25-0.3 ms a step at ~1M context;
- **prefill:** +8-15% (not additive under the 400 W cap);
- **follow-up turns:** up to 0.3 s a turn at 985K;
- **memory:** ~1.9 GiB a GPU of freed headroom (embedding + scratch), all with today's ~985K window and 4 streams
  intact.

These sit within the independent decode pass's 0.5-0.8 ms and GLM's 0.3-0.8 ms.

**There is no single large exact decode lever left.** The step is 2.3-2.9 ms above its byte floor, and that gap is
spread over ~1,300 small latency-bound launches. Bigger decode moves would need faster memory (the user's clock) or
fewer bytes; q4 dense is ruled out by the user for quality. Prefill has more exact room: tile shapes, dead work and piece quantisation.

---

## 1. What production does today

### 1.1 Benchmarks (the 91-patch image, overnight combined slot) [M]

| Measure | Value |
|---|---|
| Plain decode step (no drafter), short / 200K / ~690K context | 7.05 / 7.72 / 8.56 ms |
| Prose / code decode, greedy | 184.6 / 301.6 tok/s |
| Prose / code decode, sampled (default settings) | 174.0 / 287.6 tok/s |
| File edits (copy drafts) | 585 tok/s |
| 4 streams, **aggregate** output | **394.6 tok/s** (the 129.4 in REPORT.md is a per-request median: Codex's correction, re-derived from `prof5/res/ab.jsonl`) |
| Prefill 4K / 16K / 64K / 128K / 200K | 5,174 / 5,883 / 5,984 / 5,863 / 5,677 tok/s |
| KV pool | 1,928 pages × 512 = 987,136 tokens; one request up to 985,088 |

### 1.2 A real agentic session (production log, 12:15-12:32 today) [M]

The traffic since the 12:15 restart is one agentic run (thinking on, effort max): another AI agent doing this same
task, growing from 5.6K to 222K tokens over 47 turns. It is a fair sample of agentic use, but one session.

| | Value |
|---|---|
| Decode, all turns | 71,472 tokens, **201 tok/s**, 13.66 ms a round, 2.75 tokens a round, 64% of offered drafts land |
| Decode, DFlash2-dominated turns | 2.46 tokens a round, 13.2 ms a round, **186 tok/s**; landing 0.60 |
| Decode, copy-dominated turns (7 file rewrites at ≥179K) | 6.8-10.3 tokens a round, 19-24 ms, 300-440 tok/s |
| Follow-up prefill (1.3-14K new tokens on a cached prefix) | 2,900-5,600 tok/s; ttft ≈ 85 ms + 0.19 ms a new token |
| Host-tier restore each turn | 1.54-1.58 GB in 69-72 ms (23 GB/s), plus a 63 ms keep after the prompt |
| Re-prefilled per turn | ~5.7K tokens: ~16% is the previous prompt's tail past the last 2,048 boundary (`snap_grid`), the rest new tool output |

**What a real round costs:** about 7.1-7.4 ms of plain step, plus ~1.2 ms for each of ~2.5 extra verify rows, plus
~1.1 ms of drafter, plus ~1 ms of sampling and host work. Two consequences:
- 1 ms off the plain step is ~+8% real decode;
- 0.1 ms off each verify row is only ~+2%.

So the per-step fixed cost (the per-layer-site glue) matters more than per-row costs for this workload.

### 1.3 Hardware facts checked today [M]

- **GPU0 (rank 1) is the tighter GPU in memory as well as clocks.**
  - Free with the engine loaded: GPU0 3,559 MiB, GPU1 5,287 MiB.
  - The desktop holds ~2.6 GB on GPU0: gnome-shell 1.5 GB; Brave's GPU process 719 MiB, a compute+graphics context that time-slices with CUDA; Xwayland; the terminal; an editor.
  - Rank 0 admits the pool by the smallest room of the ranks (`fg/cuda_native.zig:284`), so **GPU0 sets the pool**.
- **Memory clock:** both cards sit at 13,365 MHz (P1) under CUDA load, against 14,001 MHz maximum. `nvidia-smi -lmci`
  reports "Memory Clock Switching Type: Runtime", so `-lmc` locking is at least nominally supported. ECC is off.
- **Clocks in decode** are held by the "Reliability" (voltage) limit, not power. **In prefill** both cards are
  power-capped at 400 W, and GPU0 runs ~20% lower SM clocks (it is hotter).

---

## 2. Decode: where the 7.05 ms goes

### 2.1 Byte floor [C]

Bytes read a step per rank (computed from the loader's shapes by the decode pass, cross-checked against the
checkpoint headers):

| Context | Bytes a step | Floor at 1.71 TB/s | Floor at 87% (what the int8 GEMV reaches) |
|---|---|---|---|
| Short | 7.10 GB | 4.15 ms | 4.77 ms |
| 150K | 7.29 GB | 4.26 ms | 4.89 ms |
| ~1M | 7.92 GB | 4.63 ms | 5.32 ms |

- **Where the bytes go:** KDA projections 2.57 GB, MoE 2.77 GB (routed experts 2.1 GB), DSA 0.91 GB, LM head
  0.63 GB, dense FFN 0.23 GB, mHC weights 71 MB, and the indexer's pooled keys at 64 B × context × 11 layers.
- **Note:** the routed experts are TP-split by their intermediate dimension, so each rank streams half of each of
  the 8 experts. Expert parallelism would move the same bytes per rank on average and add imbalance (v1's +15-30%
  was wrong; both GLM and I re-derived this).

### 2.2 The gap above the floor (rank 1, a plain step, [E] from the 66-patch nsys scaled to today)

| Latency class | Time | Count |
|---|---|---|
| mHC sites (hc_post, hc_partial, hc_finish) | ~0.59 ms | 90 sites |
| P2P all-gathers incl. waits for the slower rank | ~0.41 ms | 90 |
| MoE router (`router_tile` + `topk_group_sum`) | ~0.44 ms | 42 |
| EXL3 small kernels around the grouped GEMVs (rot_in, epilogue, combine) | ~0.42 ms | 42 × 3 |
| EXL3 grouped kernels below the int8 GEMV's efficiency | ~0.25 ms | 84 |
| DSA attention chain (norms, absorb, cache writes, chunks at 28 µs/layer, merge, expand) | ~0.5 ms | 11 |
| KDA chain, split-K reduces, step overhead (incl. a 93 µs token broadcast) | ~0.4 ms | |

About 1,309 kernels a plain step at short context (~1,353 on the sparse path). The int8 GEMVs are at the bandwidth
roof; **the gap is glue latency**. CUDA graphs are not the answer here:
- the P2P gather's epoch is host-incremented, which blocks capture;
- PDL already covers most of the chain;
- measured GPU idle was only 4.8% of the step.

---

## 3. Decode opportunities (exact)

**D1. Rework the mHC site** [C+E] (−0.1 to −0.2 ms a step; corrected after Codex's review).
- **The Sinkhorn is mostly off the critical path already.** In the generic `finish` (`kc/glm_glue_hc.cuh`), warp 0
  runs the 20-iteration Sinkhorn before the block barrier. But production runs `finish_split` (`hc_split = true`,
  `fg/cuda_forward.zig:56`; `kc/glm_glue_hc.cuh:367-417`). There warp 0 publishes `pre` and then runs the Sinkhorn
  alongside warps 1-7's collapse; glue5 c6 measured that split at ~0.
- **What is left:**
  - a probe that computes the 4×4 iteration in registers;
  - a probe that runs it in an extra CTA of the following GEMV.
  Neither helps unless the Sinkhorn outlasts the collapse. Not inside the P2P gather's wait either: on the critical
  rank 1 the gather does not wait, because rank 0 has already posted its flag (`kc/glm_glue_p2p.cuh:58-82`).
- **Fold `hc_post` into the gather.** The peer's partial lands in local memory inside `tf_glm_p2p_gather`
  (`kc/glm_glue_p2p.cuh:57,85`). In decode only `hc_post` reads the gathered buffer. Doing the post's
  `comb·x + post·Σpartials` there removes 90 launches (~1.4 µs each plus a boundary).
- **Hoist immutable loads above the PDL wait.** `hc_partial8_o2` and `hc_finish_split` load immutable weights after
  the wait (`kc/glm_glue_hc.cuh:123` vs `:157`, and `:384`). Moving them up is 30-70 µs.
- **Caution:** glue5's earlier post+partial fusion measured ~0, and 0086/0089 needed the 0090 wait fix. Re-run
  `pdl_sites.py`, and measure each sub-item separately in the same slot.

**D2. PDL on the rest of the decode chain** [C] (−0.06 to −0.15 ms).
- About 350 launches a step still go out without the attribute: `kda_prep/step/out`, `router_tile` (it triggers but
  isn't launched as a dependent), the whole DSA chain, and `down_combine`.
- `down_combine` follows the side-stream join event, so it needs a device-flag join instead.
- This is the pdl5 method, with its audit tool as the gate.
- Caution: pdl5 measured that a trigger in the wrong place regresses (+0.9% on `kda_out`/`expand_v`), so measure
  every site.

**D3. Split the decode indexer across the ranks** [C+M] (−0.25 to −0.3 ms a step at ~1M; frees 5.8% of the pool).
- One-row scores read all pooled keys on both ranks: 64 MiB a layer and 704 MiB a step at 1M. attn5 measured this
  at the DRAM floor (55.3 MB in 43 µs a layer at 864K).
- Each rank could score half the pools and keep its local top-512 with (score, id). One small exchange (~4 KB a
  layer) and an exact merge in today's tie order then give the same 512 pools.
- Every agent that looked agrees this is the only way under the floor: attn5 named it and left it, and GLM's A4 and
  the decode pass reach the same conclusion.
- It is also the only exact item that **grows with context**, which is this recipe's headline feature. And it frees
  the replicated half of the pooled keys.

**D4. The router's single-CTA top-k** [C] (−40 to −60 µs).
- `topk_group_sum` is still one CTA of 1,024 threads (`fg/cuda_glue.zig:271`): top-8 of 288 as serial argmax rounds.
- A one-pass rank-based select in one or two warps is exact if the NaN and tie order are reproduced.

**D5. Pipeline the latent-attention tile loop** [C+E] (−90 to −130 µs).
- At 1 row the DSA chunk kernel runs 10 CTAs at ~28 µs a layer.
- Overlap the QKᵀ of tile t+1 with the softmax/PV of tile t, keeping each tile's operation order, to reach ~16-20 µs.
- This also helps the 4-stream rounds.

**D6. Smaller exact items** (each ≤ 50-100 µs, each needs its own A/B):
- **Sibling int8 GEMVs as two-problem launches** (GLM A2, the 0073 pattern). It saves launches, not bytes.
- **`f_b`/`g_b` inside `kda_prep`.** Exact only if glm_mm8's arithmetic is reproduced.
- **A last-arriving-CTA split-K reduce.**
- **Fast q8 unpack in decode GEMVs** (Codex 4.1). Verified: the fast path only runs for `P2 && BITS == 8`, which is
  prompt only (`kc/glm_mm.cu:249`). My expectation is little gain at 1 row, where one byte feeds one FMA and the
  GEMV is DRAM-bound, and some at 16-32-row windows. Cheap to test.
- **The integer-placement fp8 → bf16 widening in attention** (Codex 4.1, CPU-verified on all finite codes). Gain
  unknown.

**D7. What does not pay here** (measured or re-derived):
- CUDA graphs for the step;
- a GPU drafter selector or drafter graphs (graph5: 0 of 22 metrics);
- more EXL3 prefetch or rings (−14 to −24%);
- the staged router;
- expert parallelism;
- rejection-sampling verification (the drafts are deterministic, so acceptance = p(draft), the same as exact match);
- draft trees (an alternative branch row lands ~10% of the time against the ~23% it needs on this MoE);
- the q8 LM head and q6 dense (closed by measurement).

---

## 4. Speculative decoding

- **How windows are sized** [C]:
  - Past the first draft, drafts are offered while their chance product stays ≥ 0.7 (`fg/cuda_lanes.zig:592-598`).
  - Windows are priced from GB10 tables (`:24-25`), but the cut rule makes the prices almost irrelevant.
- **The drafter is overconfident.** Its mean raw chance is 0.88 against 0.75 landed on the traced prompts, and 0.60
  landed on the agentic run.
- **Perfect sizing is the ceiling.** It would gain ~+10% on the agentic run (replays: +16% lone, +23% shared).
  Realistic calibrated rules gain +0-2%.
- **Sampled decode, the one cheap lever: coupled drafting** (`TF_GLM_DFLASH2_COUPLED`, `fg/dflash_select.zig:20-73`).
  - It is already implemented: the drafter's choice runs the same keyed Gumbel race as the target's sampler, so a
    well-calibrated selector lands its draft more often at T > 0.
  - At selector temperature 1 it measured nothing, because the selector's scores are far too peaked.
  - A fitted τ > 1 could add +0-6% to sampled decode, with outputs unchanged.
  - It can be sized **offline, exactly**: the selector's inputs (`dr.cand`, `dr.gh`) do not depend on which draft is
    picked, so dumping them with the replies lets any τ be replayed against the true tokens.
- **Keep `CLEAR_THINKING=1`.** Keeping old reasoning would re-prefill ~27% more and grow the context ~32% on this
  session.

---

## 5. Prefill

### 5.1 What binds [M]

GPU0 (rank 1) is the critical rank. I re-measured the per-call kernel medians from the 64K capture (same grids on
both ranks, so the same work):

| Kernel (2,048-row pass) | Rank 0 (GPU1) | Rank 1 (GPU0) | Ratio |
|---|---|---|---|
| tf_exl3_prompt_gu (8,544) | 1,912 µs | 2,636 µs | 1.38 |
| tf_exl3_prompt_down (16,544) | 1,100 µs | 1,282 µs | 1.17 |
| glm_mm8_128s2 KDA in-proj 2048×12576×4096 (197,16) | 1,096 µs | 1,471 µs | 1.34 |
| tf_glm_chunks_32_fp8_paged (2048,1,5) | 2,971 µs | 3,844 µs | 1.29 |
| tf_glm_kda_step | 875 µs | 971 µs | 1.11 |

- **The skew is ~1.3×, and it is clock.** The "46-49% of a long prefill waiting" quoted in profile5, glm-v2 and
  codex comes from exclusive-time accounting of overlapped SendRecv kernels. By per-call time rank 0 idles ~20-25%.
  The conclusion (rank 1 sets the pace) stands; the size of the imbalance is ~1.3×, not ~1.9×.
- **Achieved throughput on rank 1** [E from the shapes]:

  | Kernel | rank 1 | rank 0 |
  |---|---|---|
  | prompt gate/up | ~104 TFLOPS | ~144 TFLOPS |
  | prompt down | ~107 TFLOPS | |
  | W8A16 KDA in-proj | ~143 TFLOPS | ~192 TFLOPS |
  | sparse attention | ~72 TFLOPS | ~92 TFLOPS |

  Against sm_120's ~500 TFLOPS dense bf16 at boost (less at the 1.7-2.2 GHz the cap allows), the EXL3 prompt
  experts are dequant-bound: the trellis decode is ALU work paid per weight.

### 5.2 Rank 1's budget today [E]

The 66-patch 64K capture, divided by each speed-up shipped since (gu 1.177×, down 1.161×, glm_mm 1.12×, fused
attention 1.20×), gives ~374 ms a 2,048-row chunk. That matches ~5,980 tok/s.

| Rank 1, per chunk | Share | Rate | Limit |
|---|---:|---|---|
| glm_mm8_128s2 (all W8A16 linears) | 27.8% | KDA in-proj ~161 TF (48% of peak at clock) | 128×64 tile, 64×16 warp tile (`kc/glm_mm.cu:105,351-354`): ~51 FLOP per L2 byte, so L2/smem rather than the int8→bf16 conversion, which is already the PRMT/LOP3/HSUB2 trick (`:92-101`) |
| EXL3 prompt gate/up / down | 23.6% / 12.9% | ~123 / ~124 TF | integer issue of the trellis decode + ~39% fixed cost per 64-member item [M exl3p]; not DRAM-bound |
| MLA sparse attention (fused) | 10.8% | ~75 TF (22%) | 1 CTA/SM (101 KB smem), 5 barriers a 32-key tile, scalar fp8→bf16 stores with bank conflicts, **dead tiles** |
| KDA chain | ~9.8% | kda_step at ~18% of fp32 issue | token-serial SIMT recurrence; grid (32,4) = 128 CTAs leaves 60 SMs idle |
| combine, put_slot, router | 2.5% | | |
| indexer | 1.9% | | the pools all-gather runs on the main stream (`fg/cuda_sparse.zig:184`): ~2.7 ms exposed a chunk |

Measured tensor peak on this card [M, Oct 30 `pf/bench_mma` at 2,535 MHz]:
- bf16→fp32 `mma.sync.m16n8k16`: 493-503 TFLOPS, full-rate fp32 accumulate;
- e4m3/s8 `m16n8k32`: 960-1,023 TOPS;
- at GPU0's ~1.75 GHz prefill clock, ~337 TF bf16.

### 5.3 Exact prefill opportunities

**P1. Fill pieces** [C+M] (+1.5-4%).
- A chunk's o-projection and shared-expert fills run in 3 pieces (`TF_GLM_OVERLAP_PIECES=3`,
  `fg/cuda_ranks.zig:17`). `ranges()` (`fg/cuda_forward.zig:138`) cuts 2,048 rows into 768/768/512: 384 and 256
  tiles on 376 slots.
- Measured from the nsys sequences: KDA o takes 678 µs against ~478 at in-proj efficiency; DSA o 1,449 µs against
  ~955.
- Step 1: an A/B of 1/2/3 pieces needs no code. Pieces were chosen before pairing and the copy-engine exchange
  existed; pairing now hides the exchanges.
- Step 2: run the prompt shared expert over the whole chunk inside `moe.body`, so the fills only combine.
- Step 3: one launch per fill, signalling each piece with device flags.

**P2. Prompt sparse attention** [C] (+2-2.5%).
- The tile loop runs all `CH/KT` = 16 tiles of every live chunk (`kc/glm_glue_attn.cuh:399,412`). About 75% of
  rows have 2,049-2,051 keys, so their 5th chunk runs 16 tiles for 1-3 keys: ~19% of tile iterations are dead.
- Stop at the last live tile. Checked in `kc/glm_glue_attn.cuh:458-505`: a fully masked tile has `tm = -inf`, so
  `active = false`, `alpha` is exactly 1.0 and `p` is 0. Then `l = fmaf(l, 1, 0)` is unchanged, and the PV MMA adds
  exact zeros to `o`. The one difference is that `−0 + 0` gives `+0`. To be bit-exact, apply `o += 0.0f` once
  whenever tiles are skipped.
- Write `store8`'s bf16 pairs as 16-byte stores (`:362-375`): same values, 8× fewer stores, no bank conflicts.

**P3. A 128×128 glm_mm prompt tile** [C+E] (+3-7%).
- 16 warps with a 32×32 warp tile, 1 block/SM, 128 registers, 3 stages × 24 KB: half the A loads per MMA and −40%
  L2-to-SM bytes.
- Exactness holds because each output keeps the same MMA chains (4 MMAs from zero, then fmaf with the group scale,
  `kc/glm_mm.cu:252-281`).
- Confirm the L2/smem diagnosis with one NCU pass first.
- Route N ≤ 256 shapes (the indexer projection, 48 blocks on 188 SMs) to the bit-equal BM 32 instance: +0.25%.

**P4. KDA twin concurrency** [C] (+1-2%).
- In `pairBody` (`fg/cuda_forward_pair.zig:41-45`), run chunk b's KDA in-proj and f_b on a second stream while
  chunk a's `kda_step` runs on its 128 CTAs.
- It needs a's conv shift moved ahead of its step. Same kernels, same bytes.

**P5. Small exact items** (+0.5-1.5% together):
- the pools all-gather off the main stream;
- `TF_GLM_INDEX_CUDA=scores` re-tested inside a bundle (add it to the rank fingerprint first);
- the index-row split skewed toward the faster rank 0 (rows are independent).

**Not recommended**, with numbers from the prefill pass:
- **Merged twin expert calls / joint planning** (Codex 4.5). A simulation with the 64-member item cap, lognormal
  routing over 288 experts and exl3p's measured item costs drops items 665→598 (uniform) to 866→674 (skewed): 1.06-1.15×
  on the experts.
  - The merged call waits for chunk b's glue (~1.5-2% of exchange exposed).
  - It needs +320-353 MB a GPU (50-55K tokens). Net +0.5-2.5%.
  - Codex agrees it must be sized from real route histograms before any kernel work; see the unified plan.
- **4,096-row passes:** ≥ +1.3 GB a GPU even after the frees in §6. Rejected.

### 5.4 Gated prefill numerics

These are opt-in flags; the NLL gate comes first, as the house rules require.
- **Chunked tensor-core KDA prefill:** +4-7%.
  - llama.cpp #29353 did this for GDN on this card class: chunk-16 `mma.sync`, a bf16 hi/lo split over 3 MMAs, fp32
    state, pp2048 +20.6%, KLD 4.6e-4.
  - That fp32-faithful split is the design most likely to pass a gate W8A8 missed by 0.0006 nats.
- **Refined W8A8** (group size 32, MLP-only, via the existing `TF_GLM_W8A8_SKIP_K`): +4-6%.
- **fp8 QKᵀ on the E4M3 pages:** +1-4% at long prompts; likely to fail its gate.

### 5.5 TTFT for follow-up turns (agentic use) [M+C]
- **Keep a finished session's pages resident** until another stream needs them. Today every turn restores the state
  from host RAM (1.58 GB in ~71 ms at 220K; ~0.3 s at ~985K) and keeps a copy (63 ms) synchronously.
- **Make the keep asynchronous.**
- **Snapshot at the exact prompt end:** −14% ttft. But `snap_grid = 2048` exists so a cache hit stays bit-identical
  to a cold run (`fg/cuda_native.zig:24-26`), so the exact version needs chunk-invariant prompt kernels. Not cheap.
- **Re-enable short-prompt batching for stepped serving** (Codex 4.2). Verified: `steps_vt` clears `prefill_many`
  and `many_rows` (`fg/cuda_lanes.zig:106-107`). The machinery exists; the error and rollback paths need review.

---

## 6. Capacity (the ~985K + 4 streams requirement)

**Today's state meets the requirement:** one session up to 985,088 tokens, 4 streams on one pool.

**Facts:**
- KV is **6,380 B per token per GPU**: (512 fp8 latent + 4 B scale + 64 B pooled keys) × 11 MLA layers
  (`fg/cuda_state.zig:79-91`).
- The latent and the pooled keys are identical on both ranks (`kv_a` and the indexer key projection are
  replicated).
- A host-tier state is 6.43 KB/token + ~223 MiB. A ~985K session is ~6.3 GiB a GPU, which fits the 8 GiB host
  tier (one such session).

**Memory that could be freed without changing results:**

| Source | Freed a GPU | Pool tokens | Notes |
|---|---|---|---|
| bf16 embedding table to mapped pinned host memory (`fg/cuda_weights.zig:560`; used only for row gathers at `cuda_forward.zig:464,543`, `dflash_block.zig:101`) | 1.18 GiB | ~+199K | costs 8 KB over PCIe a token: a few µs a step, ~1 ms a 2,048-row chunk |
| Split pooled index keys (D3) | 352 B/token | +5.8% | also speeds long-context decode |
| M1: prompt indexer scores in 128-row blocks instead of 512 (scratch is 512 B per token of capacity, `fg/cuda_state.zig:292`, `select_rows` at `fg/cuda_sparse.zig:15`) | ~378 MB | ~+59K | rows are independent: same bits |
| M2: dense prompt attention through the fused kernel too (`if (sparse && n == 0)`), prefill partials sized 0 | ~303 MB | ~+47K | the fused kernel is byte-identical by 0068's check |
| M3-M5: unused `eact` on the EXL3 path, 8-slot `ey`, MTP rows when no target head is loaded | ~155 MB | ~+24K | verify `w.mtp == null` in production first |
| M6: one aliased region for the KDA/DSA/MoE/MLP prompt scratch | ~700 MB | ~+110K | `pdl_sites.py` must also check stores before the wait first |
| The device prompt cache (0.4 GiB) | 0.4 GiB | ~+67K | leftover space; it only ever held the first 4K state in today's session |
| `VISION=0` (glm-v2 D4) | 1.05 GiB on **rank 0** | **~0 today** | GPU0 is the binding rank, so freeing GPU1 doesn't grow the pool |

M1-M6 byte counts are from the sizing code with assumed config values; log the sizes array once to confirm.

This headroom can pay for speed or a fatter reserve for GPU0's desktop growth, while keeping ~985K. It is not needed
for the current requirement, which today's pool already meets.

---

## 7. User decisions (no engine code)

| # | Decision | Effect | How |
|---|---|---|---|
| U1 | Memory clock P1 → P0 | up to +4.5% decode (bandwidth-bound) | **Don't use `-lmc`:** on an RTX 5090 (same GB202 chip) with our exact driver 615.71.09, `-lmc 14001,14001` cut read bandwidth 1,706 → 1,060 GB/s (NVIDIA open-gpu-kernel-modules #1266; NVIDIA reproduced it, no fix yet). Try `sudo nvidia-smi -i 0,1 -cc 1` (CUDA clock override; present in this driver, untested on Blackwell), check `clocks.mem` and a bandwidth test plus a plain-step A/B under load, and undo with `-cc 0` |
| U2 | Asymmetric power caps 440/360 W | prefill follows GPU0's clock: ≤ ~+4.7% [E] | `sudo nvidia-smi -i 0 -pl 440 && sudo nvidia-smi -i 1 -pl 360`, then persist in `nvidia-power-limit.service` |
| U3 | ~~q4 dense~~ | removed by the user on 2026-10-31 (quality issues) | production stays q8 |
| U4 | Mixed-traffic policy `TENSORFOLD_PREFILL_SHARE` | p=0.84: TTFT +15%, decoders 3.4× during a long prefill | knob exists, exact |
| U5 | Brave's GPU process time-slices with CUDA on GPU0 | unmeasured | optional: `--render-node-override=/dev/dri/renderD130` (the iGPU) keeps the monitors where they are |

---

## 7b. Upstream and ecosystem, what applies (sources in `opt/opus2/upstream.md`)

- **TensorFold main** is 169 commits past our pin (94b90c9d). Nothing there is from the banned accounts.
  - **Relevant to GLM:** #1736 `TF_GLM_HOST_CACHE`, measured on this exact hardware: a 104K/250K resume went
    34/83 s → 0.7/0.8 s, with 57 GB/s pinned transfers against our 23 GB/s restore. Compare it with our 0015 tier
    before porting anything.
  - **Also relevant:**
    - #1739/#1740 (drafts): a shared round's attention and GDN in one launch across 2-32 streams, byte-identical, ~2.2%
      on GB10. Pairs with Codex's 4.3 (segment-batched DSA indexing).
    - #1580 (L2 prefetch): GLM code unchanged since the night's test (R5, skipped).
- **The DeepSeek V3.2 DSA reference uses fp8 (e4m3) indexer keys** with a per-token scale. sm_120 fp8 logits
  kernels exist now (FlashInfer #8099, RTX PRO 6000).
  - Our pooled keys are bf16 (256 B a pool). fp8 halves the scorer's bytes (~0.26 ms a step at 1M) and frees ~5% of
    the pool.
  - It is lossy, so opt-in behind the NLL gate. It also stacks with D3.
- **Chunked GDN on sm_120** (llama.cpp #29353): chunk-16 `mma.sync` with a bf16 hi/lo split, +20.6% pp2048 on an
  RTX PRO 6000, KLD 4.6e-4. This is the best template for the gated chunked-KDA prefill.
- **mHC on this card** (SGLang #42864): 160 K-slices instead of 80 CTAs took the M=1 mHC from 8.45 to 6.23 µs. Our
  `hc_partial8` uses (1,16,13) CTAs; check its fill against 188 SMs as part of D1.
- **PCIe one-shot all-reduce** (rtx6kpro, 4× RTX PRO 6000): 6.4 µs at 8 KB against NCCL's 13.9. Our P2P gather is
  already faster (3.2 µs a 16 KB gather), so nothing to take.
- **Spec decoding for GLM-5.3-Flash:**
  - incoai's card gives DFlash2 acceptance lengths of 4.0-5.8 (T = 1), above MTP.
  - DFlow (+10-13% acceptance length with no extra target work) and TreeSpark (+15-25%) are new drafter-side ideas.
    They would need drafter retraining or tree verification, which is not on sm_120 (FlashInfer's tree KDA verify
    needs tcgen05).
- **TP vs EP or DP-attention:** no 2-GPU evidence that either beats TP at batch 1. Keep TP=2.

---

## 8. Verification plan

- **One slot, interleaved legs.** Same-slot A B B A per item, greedy hashes identical, gain > 2× the A-to-A spread.
  Same harness as the night (`prof5/ab/run_ab5.sh`).
- **Report aggregate 4-stream throughput** as total tokens / makespan (Codex's correction).
- **Add an agentic replay:** the 47-turn session's prompts (from the log) at its real context positions, to measure
  real rounds and follow-up ttft.
- **Refresh the profile first.** No nsys exists of the 91-patch build. Take one capture of a plain step and a 64K
  prefill, so items D1-D5 are sized on today's kernels before anyone builds them.
- **Numerics items:** `opt/num3/qh.py` NLL against the null spread, then the T7 accuracy arm. Never KL or hashes.

---

## 9. Cross-check of the other two reports and my own passes

**Spot-checked in code: right (the main code claims of all three reports):**
- **Codex:**
  - the fast q8 unpack is prompt-only (`kc/glm_mm.cu:249`);
  - `prefill_many` is disabled in stepped serving (`fg/cuda_lanes.zig:106-107`);
  - the DSA indexer runs per segment in shared rounds (`fg/cuda_shared.zig:148`);
  - the 4-stream metric is a per-request median (re-derived: aggregate 341.6 → 394.6 tok/s, +15.5%);
  - the KV formula.
- **GLM:**
  - EP gives no gain at TP2;
  - the chunk attention body is already mma16816;
  - the indexer split is the long-context lever;
  - the night's tested-and-dropped list.

**Corrected:**
- **glm-v2 D4 (`VISION=0` → +315K pool tokens): wrong.** The pool follows the smallest room among ranks
  (`fg/cuda_native.zig:284`), and GPU0 (the desktop card, rank 1) is ~1.7 GB tighter than GPU1. Freeing the vision
  tower on rank 0 adds nothing until rank 0 becomes the tighter one. The vision tower is also 1.05 GiB, not 1.8.
- **The prefill rank skew is ~1.3× per kernel, not "46-49% waiting".** Exclusive-time accounting inflated it in all
  three reports.
- **glm-v2 A1 (EXL3 grouped occupancy restructure) is disputed.** The decode pass found that warp count, split-K and
  K range are fixed by the trellis bit layout (`kc/exl3_grouped.cuh:303`), and that ring, tile and fusion variants
  all measured nothing. It stays open only as a measured experiment.
- **My spec pass's host-loop item (+2-4%) is mostly already tested.** graph5 measured the GPU selector and drafter
  graphs at 0 of 22 metrics. Only the follower rank's two blocking host round trips (`fg/cuda_follow.zig:117-131`)
  remain untested.
- **glm-v2's `q4` freed-memory figure (0.9 GiB) looks low** (now moot: the user removed q4). q8 packing freed 3.42 GiB from bf16, so q4 should free
  ~1.7 GiB a GPU [E].
- **Codex 4.5 (joint expert planning across paired chunks) is sized low.** My prefill pass simulated it: +0.5-2.5%
  net, and it needs 320-353 MB a GPU. It is worth one route-histogram measurement, not a kernel.
- **Codex 4.1 (fast q8 unpack in decode)** is a real gap, but decode GEMVs are DRAM-bound at 1 row, so expect gains
  only at wide windows. My prefill pass found the prompt GEMM's limit is its tile shape, not the conversion.
- **My decode pass's "Sinkhorn inside the P2P gather (−130 µs)" was wrong for the critical rank.** Rank 1 arrives
  last and does not wait in the gather, so the work would stay on its path.
- **Both the decode pass and I read the generic `finish`, not production's `finish_split`** (Codex caught this in
  review). The Sinkhorn already overlaps the collapse there, so D1 shrinks to −0.1 to −0.2 ms.
- **My own U1 was wrong in the first draft** (`-lmc`). The upstream pass found the #1266 bandwidth collapse on this
  chip and driver, so U1 now says `-cc 1` with a bandwidth check.
- **glm-v2's "4,096-row passes: dead" agrees with my prefill pass** (≥ +1.3 GB a GPU even after the frees).

---

## 10. The unified plan

The plan agreed with Codex and GLM is in `docs/unified-plan.md`. After three review rounds both returned **FINAL:
AGREE**; its §9 records each round. Where this report and the plan differ, the plan wins: it carries the reviewers'
corrections, including `finish_split`, the 2.5 dependency and the O11 scope.
