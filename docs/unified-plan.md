# Unified speed plan: decode and prefill for GLM-5.3-Flash on 2× RTX PRO 6000

**Date:** 2026-10-31. **Status:** plan only; no code, setting or GPU state was changed to produce it.

**Agreed by:**
- Claude Opus (`docs/OPUS-v2.md`);
- Codex (gpt-6.1-sol, `GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold/docs/codex.md`);
- GLM-5.3-Flash (`docs/glm-v2.md`).

It took three review rounds, and both reviewers returned **FINAL: AGREE** on this version. The record is under §9.

**Labels:** **[C]** code (`G/` = `zig/src/families/glm/`, `K/` = `zig/kernels/cuda/` in the production source:
upstream f350cd3e + `patches/engine/0001-0091`); **[M]** measured; **[E]** estimate.

## 1. Constraints
- **The user's requirements:**
  - One session must be able to reach ~985K tokens in the shared KV pool, and 4 streams must stay available. Never
    trade context or streams for speed.
  - Monitors stay on GPU0.
  - Power and clock settings are the user's to change.
- **Rank 1 (GPU0) is the critical rank** for decode and prefill; only gains on its path count in an ABBA.
- **Exact items** ship with identical greedy hashes in a same-slot ABBA and a gain above 2× the A-to-A spread.
  - Each also needs its own byte or contract check, with adversarial cases: ties, NaN, ±0, subnormals.
  - `pdl_sites.py` must PASS after any PDL-adjacent change.
- **Numerics items** are opt-in flags gated by NLL against the null spread, then the accuracy arm. They stay off
  unless the user turns them on.
- **Capacity-track items** need identical hashes and no measurable slowdown (within the A-to-A spread), with the pool
  line from rank0.log showing the gain. They ship only if the user wants the headroom, since today's pool already
  meets the requirement.

## 2. Phase 0: measure before building (no engine code except the trace patch in 0.5)
- **0.1 A fresh nsys capture of the 91-patch build** at the production 985K/4-stream config. Preflight the profiling
  memory: CUPTI costs ~0.7 GiB.
  - Shapes: plain step, forced 8-row round, 4-stream round, 64K prefill, 200K follow-up turn.
  - Also: both ranks' critical paths, paired-chunk expert-route histograms, and unprofiled control legs (nsys
    inflates 4-9%).
  - No capture of today's kernels exists; every estimate below is sized from the 66-patch capture.
- **0.2 Zero-code A/Bs:**
  - `TF_GLM_OVERLAP_PIECES` 1/2/3 (prompt fills);
  - attn5 c4 at ~690K (one pair so far);
  - the c9 `TF_GLM_HC_PART` table.
- **0.3 User-run and optional, outside the 400 W baseline:**
  - `sudo nvidia-smi -i 0,1 -cc 1`, with a bandwidth test and a plain-step A/B. Memory at 13,365 of 14,001 MHz caps
    the gain at 4.76%.
  - **Never `-lmc`:** −37.9% read bandwidth on GB202 with driver 615.71.09 (NVIDIA open-gpu-kernel-modules #1266).
  - The 440/360 W power-split A/B.
  - Not required: reinstalling the host CUDA toolkit.
    - Builds, `pdl_sites.py` (run with `cuobjdump` from the cached `nvidia/cuda:13.3.1-devel` image, as integ5 did)
      and profiling all work in containers.
    - A host install only makes quick host-side builds and checks more convenient.
- **0.4 A length-matched synthetic agentic benchmark** built from today's 47-turn log: context positions, follow-up
  sizes, output lengths. Use captured request traces if you allow them.
  - Report 4-stream throughput as total tokens / makespan, not as a per-request median.
  - Report TTFT by follow-up-length bucket.
- **0.5 Selector trace, then the coupled-draft τ** (`TF_GLM_DFLASH2_COUPLED`, `G/dflash_select.zig`):
  - Record `dr.cand`, `dr.gh`, the `pred`/`succ` constants, seed, position, anchor, sampler parameters and codebook
    identity.
  - Capture the emitted target token ids with request identity and absolute positions.
  - Run one replay leg with `TF_GLM_DFLASH2_TRACE=1`.
  - The replay recomputes the predecessor-dependent chain at the recorded anchors.
  - `dr.temp` calibrates probabilities; a race-score temperature needs its own definition.
  - The τ fit, exact replay and ship decision follow once these traces exist: ship τ only if the replay gains.
  - Any τ change requires a live sampled ABBA with identical seeded outputs; offline replay alone cannot establish
    end-to-end gain.
  - Expected +0-6% sampled decode [E].
- **0.6 The T7 accuracy arm** for the broad W8A8, in the first long slot. Broad W8A8 remains failed and off: +0.0074
  nats exceeds the +0.0068 null limit. The arm supplies more evidence; it cannot overturn that NLL failure.

## 3. Phase 1: cheap isolated exact probes (days each, in this order)

| ID | Item | Expected [E] | Gate |
|---|---|---|---|
| 1.3a | Prompt sparse attention: 16-byte `store8` writes (same values) | with 1.3b, +2-2.5% prefill | byte check |
| 1.3b | Stop at the last live 32-key tile: 14.8% of all tile iterations are dead (18.75% within the affected rows); add an `o += 0.0f` canonicalisation when tiles are skipped | | adversarial CUDA byte check, incl. ±0 and subnormal `o` |
| 1.7 | `topk_group_sum`: a small-row specialisation (today 1 CTA × 1,024 threads); keep the large-window path | −40 to −60 µs a step | contract harness: ordered fp32 weight sums, NaN/all-NaN, lower-id ties (`K/glm_glue_moe.cuh:129-186,270-286`) |
| 2.9 | Fast q8 unpack in decode GEMVs (prompt-only today, `K/glm_mm.cu:249`); integer fp8→bf16 widening in attention, keeping the native fp32 scale | unknown; kept only if measured | NaN codes, ±0, the amax/448 boundary |
| 1.4a | Hoist immutable weight/base/scale loads above the PDL waits in `partial_split` and `finish_split` | −30 to −70 µs | `pdl_sites.py` |
| 1.4b | Fold `hc_post` into `tf_glm_p2p_gather`: release the flags before the extra work; keep the abort path, rank-order sums, bf16 rounding and FMA order (`K/glm_glue_rows.cuh:102-114`); the old path stays behind a flag for the NCCL fallback | ~−90 µs | byte check; ABBA |
| 1.4c | Bundled with 1.4b: P2P call-identity hardening (an epoch skew can pair different gathers carrying the same epoch; graph5) | robustness | planted-fault test |
| 1.5 | PDL on the remaining launches, **site by site**: KDA chain, `router_tile`, the DSA chain, and `down_combine` via a device-flag join (stream-order and deadlock checks) | −60 to −150 µs in total | `pdl_sites.py`; per-site ABBA (one trigger site already regressed +0.9%) |
| 1.9 | O8: last-arriving-CTA split-K reduce. O9: f_b/g_b inside `kda_prep`. O11: device token feedback, initially restricted to plain, unmasked greedy steps without drafting; the current path stays for other requests | −0.14 to −0.24 ms in total (O11 only on plain greedy steps) | O8/O9: byte checks. O11: token and state equivalence, plus EOS, stop-string, output-limit, cancellation and slot-reuse tests across both ranks; work computed ahead is discarded or rolled back before it affects committed state |

**Not in the plan, on purpose:** moving the mHC Sinkhorn.
- Production's `finish_split` (`G/cuda_forward.zig:56`, `K/glm_glue_hc.cuh:367-417`) already runs it alongside the
  collapse.
- The critical rank 1 does not wait inside the gather, because rank 0 has already posted its flag.
- It stays only as a measured probe, should 0.1 show the Sinkhorn outlasting the collapse.

## 4. Phase 2: TTFT and concurrency (exact)

| ID | Item | Expected [E] | Notes |
|---|---|---|---|
| 2.7 | Re-enable short-prompt batching for stepped serving: `steps_vt` clears `prefill_many`/`many_rows` (`G/cuda_lanes.zig:106-107`); `tiled` is on in production because the q8 packs require it (`G/cuda_engine.zig:187`) | TTFT under concurrent arrivals | only already-queued eligible requests; rollback, grammar/logprobs, image/resume exclusions |
| 1.6 | Reclaimable resident **prompt-prefix** pages plus their matching saved KDA/index/DFlash state, with an asynchronous keep. The synchronous freeze covers the saved KDA/index/DFlash state and cur/bases; only then do pages become reclaimable and the KV copy go async | up to −70 ms a turn at 220K, ~−0.29 s at 985K (restore only; today 1.58 GB in ~71 ms at 220K [M]) | admission counts the pages as reclaimable only with coordinated, completed eviction; transparent fallback to host restore |
| 2.6 | Segmented DSA indexing across a shared round's streams, and grouping of heterogeneous dense/sparse segments (attention batching already exists) | 4-stream rounds; sized by 0.1 | preserve `row0`, score order, ties |
| 2.10 | Mixed-traffic policy `TENSORFOLD_PREFILL_SHARE` (exists, exact) | measured: p=0.84 → decoders 3.4× during a long prefill, TTFT +15%; p=0.68 → 6.2×, +42% | the user's choice |

## 5. Phase 3: medium exact work, chosen by the Phase 0 profile (1-2 weeks each)

2.3's design work (tie-order proof, snapshot/layout migration, the glm-index-check extension) starts in parallel with
Phase 1, because it has the longest lead and 3.4 stacks on it.

| ID | Item | Expected [E] |
|---|---|---|
| 2.1 | glm_mm W8A16 prompt tile 128×128 (today 128×64, ~145-160 TF, likely L2/smem-bound): four ordered MMAs per 64-input group, the same scale FMA and split-K; generalise every hard-coded 8-column loader/output map (`K/glm_mm.cu:144,160,289`); route N ≤ 256 to BM 32 | +3-7% prefill (speculative until NCU confirms) |
| 2.2 | Prompt fills: the whole-chunk shared expert in `moe.body` (keep the existing prompt arithmetic on short pieces and tails, and include those shapes in the byte check); single-launch fills with all-CTAs-complete flags and progress-safe side-stream waits (if 0.2 shows a piece cost) | +1.5-4% prefill |
| 2.3 | Indexer split across the ranks, decode and prefill. Order by descending `order_key(score)`, then ascending global pool id, emitted in ascending id (`K/glm_index_decode.cuh:24-33,136-188`). Includes snapshot/layout migration and glm-index-check extended to the split path before the build. `TF_GLM_INDEX_CUDA` and any new mode go in the rank fingerprint | −0.25 to −0.3 ms a step at ~1M context; +5.84% capacity |
| 2.4 | Decode latent-attention tile pipelining (chunk geometry and merge order fixed) | −90 to −130 µs a step |
| 2.5 | KDA twin concurrency, for long paired prompts only. **Dependencies:**<br>• a's prep completes before a's convolution shift;<br>• that shift completes before b's input projection overwrites the shared `p`;<br>• b's `g_b`/prep wait until a's out has consumed the shared `kg`;<br>• the remaining scratch and recurrent-state lifetimes are preserved.<br>Require paired-prompt byte checks of the outputs, convolution state and recurrent state, including the following continuation (`G/cuda_kda.zig:125-158`, `G/cuda_kda_block.zig:69`; twins alias, `G/cuda_state.zig:378-383`) | +1-2% prefill |

## 6. Capacity track (optional: the requirement is already met)

The KV pool costs 6,380 B per token per GPU.
- **C1: the embedding table to mapped pinned host memory.** It frees 1.182 GiB/rank, about 199K tokens.
  - **Optional and opt-in (the user's request, 2026-10-31):** build it behind `TF_GLM_EMBED_HOST=1`, default off,
    exposed as a recipe knob `EMBED_HOST=0|1` in `scripts/config.sh` and in the context-fit key. With the flag off,
    the device table and today's behaviour stay exactly as they are.
  - It is row-gathered only: `G/cuda_forward.zig:464,543`, `G/dflash_block.zig:101`, `G/cuda_forward_pair.zig:31`.
  - Design:
    - one copy shared by both rank processes, so 1.18 GiB of host RAM rather than 2.36;
    - the prefill rows prefetched into a small device staging buffer (~16 MB) on a side stream, since the chunk's
      ids are known ahead;
    - decode reads its few rows straight from the mapped memory.
  - Expected cost [E]: +1-3 µs a plain step (~0.03%), +4-8 µs a drafted round, ~0 in prefill once prefetched.
    Accept it under the capacity rule: identical hashes and no slowdown beyond the A-to-A spread.
- **C2: prompt scratch.**
  - M1: 128-row score blocks instead of 512 (`G/cuda_state.zig:292`), 378.3 MB.
  - M3-M5: 155.2 MB; M5 only if `w.mtp == null`.
  - Log the sizes array once at startup to confirm these sizes and the M5 condition.
- **C3: M2, dense prompt attention through the fused kernel (303.2 MB).**
  - It needs new dense wrappers carrying `pos` and explicit fused dispatch: today's wrappers force sparse, and
    `partials.rows = 0` means "unknown size".
  - It needs its own dense byte proof.
- **C4: M6, one aliased prompt-scratch region (~700 MB).**
  - Only after C2/C3, 2.2 and 2.5 settle.
  - Needs a cross-stream lifetime/alias analysis, and `pdl_sites.py` extended to stores before the wait.
- **C5:** check returning the 0.4 GiB device prompt cache to the pool. It held only the first ≤4K state during the
  whole session.
- **Totals:** embedding + M1-M5 ≈ 1.96 GiB/rank; ~2.6 GiB if M6 works out.

## 7. Phase 4: gated numerics

These are opt-in, judged by NLL, then the accuracy arm, plus long-context continuation checks.
- **3.2 Chunked tensor-core KDA prefill:** +4-7% prefill.
  - Keep the per-channel decay and fp32 boundary state, with an fp32-faithful split. llama.cpp #29353's GDN kernel
    on this card class is the design reference.
  - Include workspace accounting.
- **3.3 Refined W8A8 by operation class**, with a prompt-row floor and a fingerprint: +4-6% prefill. It can't be
  built on `SKIP_K`, which filters by width.
- **3.4 fp8 indexer keys:** 341 B/token (+5.65% capacity), only +2.91% after 2.3.
  - Settle GLM's indexer format first (Hadamard/scale).
  - Its scoring gains overlap with 2.3.

## 8. Designs only, deferred, dropped

- **Designs only:**
  - Grow-on-demand admission: pages are promised at admission today (`G/cuda_pages.zig:150-183`).
  - Copy-on-write immutable prefix pages, including the DSA pooled keys, index rings, KDA/conv state and drafter
    context.
  - A suffix-overlap measurement before any cross-request drafting.
- **Deferred:**
  - CUDA graphs: for ROI (idle 4.8%; host epochs need a graph-safe protocol), not impossibility.
  - Draft trees: landing economics, and no native support (`G/cuda_lanes.zig:437`).
  - Merged twin expert calls: until the 0.1 route histograms are in; the 64-member item cap makes routing decisive.
- **Dropped:**
  - Expert parallelism: the same bytes per rank, plus imbalance.
  - Rejection-sampling verification: with deterministic drafts it accepts exactly as often, p(d).
  - The GPU drafter selector: 0 of 22 metrics; the drafter graph arm wasn't run, but its estimated headroom is
    ≤0.02-0.05 ms.
  - 4,096-row passes: ≥ +1.3 GB/rank.
  - `VISION=0` as a pool lever: GPU0 binds the pool (`G/cuda_native.zig:284`).
  - `-lmc`.
  - **q4 dense:** removed by the user on 2026-10-31 for quality. Not to be re-proposed.
  - q8 LM head and q6 dense: your recorded decisions.

**Expected if Phases 0-3 land** [E, conditional, not additive]:
- plain step ~6.4-6.7 ms (from 7.05), so real agentic decode +4-8%;
- ~1M-context plain step −0.25 to −0.3 ms more;
- prefill +8-15% under the 400 W cap;
- follow-up TTFT −0.07 to −0.29 s a turn;
- if you want it, ~1.96-2.6 GiB/rank of capacity headroom.

The biggest single levers stay yours:
- `-cc 1`: up to +4.76% decode bandwidth;
- the power split: prefill follows GPU0's clock.

## 9. Review record

| Round | Codex | GLM | What changed |
|---|---|---|---|
| 1 (plan-v0) | agree with amendments | agree with amendments | Production runs `finish_split`, so 1.4 shrank; the plain-step target went from 6.3-6.5 to 6.4-6.7 ms; exact M1-M5 bytes (836.6 MB); 14.8% dead tiles; 341 B/token for fp8 keys; M2 moved out of low-risk; tracing moved into Phase 0; the capacity track separated; O8/O9/O11, P2P hardening, mixed traffic, grow-on-demand/CoW and the fingerprint guard added |
| 2 (plan-v1) | blockers: 2.5 (aliased `p` race), 1.9/O11 (scope, control-plane gate) | agree, 4 non-blocking fixes | Codex's blocker texts adopted verbatim; the replay qualification, short-piece arithmetic, W8A8 status, both mixed-traffic points, the critical-rank rule and the sizes log restored |
| 3 (plan-v2) | **FINAL: AGREE** | **FINAL: AGREE** | none |
| After agreement (the user) | — | — | C1 (embedding to host) kept, as an opt-in flag that is off by default. q4 dense removed (quality). The host CUDA toolkit reinstall made optional: nothing in the plan needs it, since `pdl_sites.py` already runs with the devel image's `cuobjdump` |

Briefs, every plan version and all six reviews are in `opt-sm120-notes/opt/opus2/review/`. The supporting analyses
are in `opt-sm120-notes/opt/opus2/` (`decode.md`, `prefill.md`, `spec.md`, `upstream.md`).

## Results (2026-10-31 run; full report: `opt-sm120-notes/opt/plan1/REPORT.md`, ship steps: `opt/plan1/SHIP.md`)

A2 and 410/390 W went live at the user's request at 22:04, then were rolled back at 23:19 and A2 discarded (the user: "A2 hurts decode speeds, a lot"). Production and the default: `987b580ba9d2`, `GPUS=0,1`, 400/400 W. At 18:20 the user set small gains aside, so most exact items were never timed; their patches are kept.

- **Candidate, later discarded by the user:** 2.3's A2 stage only (`TF_GLM_INDEX_DEEP=32768`, image `tensorfold-glm53-zig:p1-d5bb88a35504`): +7.57% prefill and -9.4% follow-up TTFT at ~690K, +1.76% prefill at 200K, identical hashes, no memory cost [M]. It helps deep prompts only.
- **Levers refuted:** `-cc 1` no effect (memory stays at 13,365 MHz); 440/360 W -3.80% prefill; 360/440 W -8.84%. Best split 410/390 W, about +2% prefill [M], not persisted.
- **Dropped with numbers:** 1.3a, 1.5a, 1.9 O8, 2.1 (best +0.75%), the 2.2 fills half, 2.9 fp8 widening, the tau knob. C1 fails its rule (plain -0.61%, prose -1.74%) and stays out.
- **Measured but set aside:** 1.4 bundle (plain +2.28%, real work +0.8-1.7% within spread).
- **New:** Brave's GPU process active on GPU0 cost about 4-7% decode and 12% deep prefill in the one leg it ran [M]. With the machine quiet, A-to-A spreads fall to 0.05-0.3%.
