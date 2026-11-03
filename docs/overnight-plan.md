# Overnight plan, 2026-10-31: decode and prefill

Revised at ~00:55 after tonight's lanes review with Codex (three rounds, agreed in rounds 2 and 3). The night starts with a fresh profile, then works the tracks below in parallel and spends the GPU in slots. **There is no restore deadline.** Main starts the night unattended as soon as this plan is written. It runs until every primary track and the reserve queue have passed, failed or stopped. Production then comes back on 8888 with today's image, and main writes the report.

Sources:
- The measured data in `opt-sm120-notes`: PLAN.md §2–3, profile.md, decode.md, draft2.md, num2.md, num3.md, prefill.md, prefill2.md, prefill3.md.
- Codex round 5 (`opt/codex-round5.md`) and tonight's lanes rounds (`opt/overnight/codex/lanes1-answer.md`, `lanes2-answer.md`, `lanes3-answer.md`).
- Tonight's research: `opt/overnight/research-upstream.md` (upstream TensorFold since 10-29) and `research-ecosystem.md` (vLLM, SGLang, FlashInfer, TensorRT-LLM, ExLlamaV3, fla, llama.cpp).
- `docs/glm.md` and `docs/deepseek.md`, both checked against the source.

Labels: **[M]** measured, **[E]** estimated, **[C]** read from the code.

## Decisions

| # | Question | Answer |
| --- | --- | --- |
| 1 | Restore deadline | **None.** You cancelled 07:15 at 00:27. |
| 2 | Who starts the night | Main, unattended, once this plan is written (you, 00:27). |
| 3 | Do you need 8888 overnight? | Only between slots. It goes down for each slot and is restored and health-checked after it. |
| 4 | May exact wins ship to production overnight? | **No.** Production stays on today's image `8d045cb28b66`. Passing patches go into a candidate image for you. |
| 5 | May the night run the W8A8 quality test? | Yes, as evaluation only. W8A8 stays off. |

The power-split A/B needs sudo, so it waits for you (see "With you, afterwards").

## What we start from

- **Production:** image `8d045cb28b66` (f350cd3e + 66 patches), source `opt-integ2` @3f94715d in `tensorfold-wt-integ`. 4 streams share 1,757 pages (899,584 tokens); one request can use 864,256.
- **Power:** both GPUs are back at **400 W**. You had set 300 W at 23:27 and restored 400 W before midnight; `nvidia-power-limit.service` also sets 400 W at boot. Every measurement tonight notes the limit.
- **The card** [M, device query]: 188 SMs, 128 MB L2 (80 MB max persisting), 100 KB shared memory per SM.
- **Measured now** [M]: plain step 7.59 ms; prose ~170, code ~277, file edits ~514 tok/s; prefill 5,470–5,553 tok/s at 16K–64K.
- **Bytes a rank reads per decode token** [C+E]: int8 dense ~3.8 GB, routed experts ~2.1 GB, head 0.63 GB, about 6.6 GB in all. At a practical ~1.55 TB/s the floor is ~4.3 ms, so the step is ~1.8× its floor.
- **The only step breakdown** is from the 9.03 ms build, before wave 3 [M, PLAN.md §2]:

  | Part | Time |
  | --- | --- |
  | Routed experts | 2.3–2.7 ms |
  | int8 attention projections | ~2.3 ms |
  | KDA/MLA attention + indexer | ~1.2 ms |
  | Shared expert + dense | ~0.9 ms |
  | mHC + norms | ~0.9 ms |
  | Router | ~0.6 ms |
  | LM head | ~0.4 ms |
  | Communication | ~0.6 ms |

- **Prefill, one unpaired 2048-row chunk** [M, prefill3.md trace, ~428 ms]: experts ~108 ms, attention cores ~72 (MLA 37, KDA 35), glue ~34, exposed exchange ~77. Paired production chunks take ~370 ms. Both GPUs sit at the 400 W cap.
- **Decode power** [M]: both GPUs at ~399 W, limited mostly by "Reliability" (0x400). GPU0 runs 4.5% slower clocks than GPU1.
- **Production runs the `cut` draft rule** [C]; `tpm` and its seeds are opt-in.

Checked tonight in the source [C]:
- Chunk attention already runs on tensor cores (`mma16816`, `glm_glue_attn.cuh:197,259`).
- `prefill_rows = 2048` is a compile-time constant (`cuda_state.zig:19`); 4,096 rows would cost ~1.9 GiB a rank.
- Decode's DSA select is one 1,024-thread CTA per row (`cuda_sparse.zig:114`). It picks the top 512 pools.
- Only `glm_mm` launches with PDL (`cuda_mm.zig`), and only `glm_mm` and `glm_mm_reduce` trigger their dependents early (`common/qmm_frag.cuh:112`).
- DFlash2's path selector runs on rank 0's host (`dflash_select.zig:1`): a device→host→device hop every drafted round.
- `TF_GLM_INDEX_CUDA` is not in the rank fingerprint (`cuda_index.zig:19`, `cuda_ranks.zig`).
- Kernels build as native sm_120 SASS (`build/cuda_fatbin.zig:42`), so a PTX fallback shouldn't compile a PDL wait out; pdl5 still checks the loaded modules' SASS before any timing.
- With all streams busy the server queues new requests; there are no admission limits by default (`server/request_admission.zig`).

**`docs/glm.md` and `docs/deepseek.md`.** glm.md's top items held up (fresh profile, graphs, the EXL3 re-tile, the W8A8 gate) and are in the plan. Its errors (SIMT attention, 32 heads, ≤512 experts, "experts dominate decode", EP +15–30%, power split +10–15%) are corrected above and in Codex round 5. deepseek.md adds only two things: TPM-128 expert items (folded into exl3p, after its first step) and DRAM-byte counters (in T1). Its other items are wrong or done: chunk rows is not a knob; attention is already MMA; chunks already overlap (pairing); the rotate kernel is ~0.3% of a pass; an uneven expert split is neither exact nor free; its occupancy figures are for kernels production no longer runs; its byte math uses the 3.5 bpw config and 45 MoE layers.

## Tracks

All estimates are potential gains [E], not additive. A failed experiment is done when it has a reproducible negative result and a report.

| Track | Agent | Target | Exact? | Estimate | First step | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| **T1 + L7** Profile | `prof5` | Critical paths, verify widths, shared rounds, long context, mixed traffic | Measurement | Sizes every track | nsys of the production path on both ranks; prefill without `TF_GLM_PROF` | Critical-path and variance tables; pairing state; bounded byte counts; mixed-traffic TTFT, token-gap p50/p95/p99 and chunk time |
| **L6** Index modes | `prof5` | `TF_GLM_INDEX_CUDA` on long prompts | Intended exact | 0–5% long prefill | Bracket `off → scores → scores → off → both → both → off`, cold ≥128K prompts; check both ranks loaded the mode | glm-index-check, continuation hashes and timing all pass; fingerprint any mode we keep |
| **T2** EXL3 decode | `exl3r` | Grouped EXL3 decode loads, staging and tiling; ExLlamaV3's GEMV as reference | Yes | 0.15–0.45 ms a plain step | Today's kernels against load-policy and register-ring variants; cold weights, real widths, 1/8/16/32 members | Bitwise operator and planted-fault checks; a useful component gain; its own engine ABBA passes |
| **exl3p** EXL3 prompt experts | `exl3p` | TPM=64 member specialization and decode/MMA interleave | Yes | +2–4% prefill | Item histogram; specialize live member blocks; alternate decode/MMA fragments; read the SASS | Equality at mixed widths and skewed routing; registers and residency hold; prefill ABBA passes. TPM=128 only after that |
| **T3** Decode glue | `glue5` | The largest eligible glue candidate, then the next | Yes | 0.05–0.25 ms a pick; HC fold 0–0.10 | T1's sizing; keep each projection's SK; audit ownership, taps and scratch | Operator/state checks and the relevant 1- and 4-stream ABBA pass, per candidate |
| **T4** Drafter graph | `graph5` | DFlash2 block-pass graph | Yes | 0.2–0.4 ms a round | Capture-safe scope and graph keys; rank-agreed capture; audit pointer/shape updates and epoch/parity waits | Graph and eager agree on state and output; safe fallback; 1–4-stream legs; graph memory in the fit |
| **pdl5** Dependent launch | `pdl5` | Audited, kernel-by-kernel PDL along the decode chain, plus the P2P gather's early trigger | Yes | Unmeasured here; llama.cpp got +22% decode on an MoE on this card | The SASS gate below, then the first chain | Bitwise operator/state checks, exact continuation hashes, a reproducible per-step ABBA gain at plain, verify and shared widths |
| **T5(a)** DSA select | `attn5` | Faster exact decode pool selection | Yes | 0–0.8 ms near max context with the existing selector; 0.7–1.3 ms with a full replacement | `tf_glm_index_pools` on decode, today's scores kept; R=1/4/16 at 128K/200K/864K with ties and fallbacks | Exact pools, order and tokens; bounded scratch; long-context engine ABBA passes |
| **T6** Draft pricing | `prof5` | Lone and shared-window draft costs; `cut` vs `tpm` | Target outputs exact | 0–5%, workload-dependent | Fit lone costs **and** the shared-window tables (`cuda_lanes.zig:23,604`); replay before engine time | Per-workload ABBA and hashes; keep `cut` where `tpm` loses; reprice after a big combined speed change |
| **T7** W8A8 quality | `gate5` | ABSORB-on baseline vs W8A8 | No: opt-in, off | Quality result; its prior speed gain is ~5–6% | Freeze the policy (SKIP_K agreed, row-threshold contract); a disjoint paired pilot; predeclare N | NLL against the null spread, and a one-sided 95% UCB of the paired loss < 1 point **in every stratum**; else inconclusive or fail |
| **T8** Integration | `integ5` | Stack passing exact candidates; build the candidate image | Passing exact work only; W8A8 off | Measured combined result | Starts after ≥2 individual passes, or at the end with one; review diffs, stack, run host suites | Combined ABBA passes; fit and pool line recorded; candidate image built; production healthy |

### Reserve queue

A worker that finishes its own track takes the highest eligible reserve; nobody waits for another track to finish first.

| # | Item | Owner | Exact? | Estimate | First step | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Upstream #1571: sampled-decode broadcasts batched | first free code worker | Yes | +0.5–2% sampled; ~0 greedy | Apply the pinned patch uncommitted; check gathered rows byte for byte | Sampled same-seed ABBA and hashes pass; greedy, structured and logprob checks; staging memory in the fit |
| R2 | T5(b): sparse attention in-kernel merge | `attn5` after T5(a) | Yes | +3–4% prefill; frees **353.375 MiB a rank** (not 642) | Keep the five independent chunk results and the merge's FMA sequence | Exact operator and engine checks; prefill ABBA; refit and the real pool line |
| R3 | Copy engines for the paired twin | free worker with comm experience | Yes | 0–3% prefill | Only if T1 shows exposed twin NCCL; register both roles; audit flags and events | Paired-path equality; no ordering regression; fit and prefill ABBA pass |
| R4 | KDA step WARPS=8/4/2 (V-split) | first free kernel worker | Yes, if shown locally | 0–0.05 ms a step; maybe prefill | Sweep the existing template at decode and verify widths | Bitwise outputs, saves, recurrent and replay states; whole-chain gain; engine ABBA |
| R5 | Upstream #1580 L2 prefetch | first free code worker | Yes | 0–2% plain; 0–1% drafted | Port only the three L2PF commits; audit code/scale spans; KDA-o site alone at 4/8/16 MiB unpaced, 2 CTAs | Active-path logs; a bounded join; exact hashes; plain, drafted and shared ABBA |
| R6 | L1 mm16: faster exact W8A16 prompt GEMM | first free GEMM worker | Yes | 1.10–1.25× kernel; +1.5–4% prefill | Bounded prototype keeping the four-K16 group resets, ascending scale FMAs and rounding | Bitwise shape coverage and a real component gain; engine ABBA before inclusion |
| R7 | Sparse drafter candidate head (ExLlamaV3) | `graph5` when free | Target outputs exact; drafts change | Not budgeted yet | Measure today's head/top-16 share; adapt the candidate/confidence interface | Target checks, sampled-key checks, acceptance/probe/rebuild accounting, fit and ABBA |

### Not tonight

- **Expert-parallel MoE.** Not disproved, but it moves the same bytes per rank and adds imbalance.
- **The MTP chain.** It would cut context to ~516K.
- **4,096-row passes.** +1.9 GiB a rank, roughly −320K tokens of pool.
- **q4 dense** (you chose q8), **an uneven TP split** (costs pool), **moving the monitors**.
- **FlashKDA / any chunked KDA prefill.** bf16 state and reassociation: a numerics change.
- **T5(c) CH128.** Dropped: numerics change.
- **Tree verification.** No exact KDA tree kernel; KDA would need per-branch states.
- **Decode context parallelism for the latent cache.** About 2× pool, but not exact and far too big for a night.

## Slot sequence (readiness-driven)

Times are a guide, not a deadline. Main moves a slot when its work isn't ready, and inserts an extra slot when candidates are.

| # | When (about) | GPU | Who | What |
| --- | --- | --- | --- | --- |
| 0 | 00:55–01:20 | Production up | main | Setup (below); launch the agents |
| 1 | 01:20–02:40 | **Slot 1**, production down | `prof5` | T1 first, then L6, L7 and T6 sizing. FORCE=16 still clamps to 8: get 16-row data from copy windows. Restore and health-check |
| 2 | 02:40–05:00 | Quiet: production up | code agents | Code, plus short timing runs serialized under `gpu.lock`, ≤1.5 GB a GPU. Its length follows progress |
| 3 | ~04:30 | Production up, then **slot P** | `gate5` | T7's paired pilot: the baseline pilot on production, then the W8A8 pilot in a short slot. Freeze N and the rule |
| 4 | after 3 | Eval: production up and busy | `gate5` | T7 baseline arm against 8888, capped at ~2 h. Others: host tests and builds only, no GPU timing |
| 5 | after 4 | **Slot 2** | `gate5` | T7 W8A8 arm, NLL/null runs and a same-slot speed bracket, capped at ~2 h. Restore |
| 6 | when ready | **Slot 3** | `prof5` | Ready candidates, each with its own ABBA. Default order: T5(a) (if T1 confirms its size), T2, pdl5, exl3p, T3's pick, T4, T6, then reserves |
| 7 | as they come | **Slot 4+** | `prof5` | Later candidates and reserves, in bounded batches. Restore after each |
| 8 | at the end | **Integration slot** | `integ5` | Every individually passing exact candidate together: combined ABBA, mixed traffic, context refit (4 streams, 864,256 cap), the pool line against today's 899,584 |
| 9 | last | Restore | main | Today's image on 8888, health check; candidate image and `opt/overnight/REPORT.md` |

If two or more candidates are ready before the T7 baseline arm would start, main may run slot 3 first. A shared bracket such as `A B B A C C A` gives two ABBA comparisons in one slot.

## Setup (main, before the agents start)

1. **Watchdog (done 00:43).** `opt/overnight/watchdog2.sh` replaced the 07:15 watchdog. It has no deadline and was tested in a sandbox on 12 cases. Every 5 minutes, with no slot flag:
   - two failed health checks in a row → restore with the recipe's `./stop.sh` / `./start.sh`;
   - a busy server that still finishes generations (new `[tensorfold] done` lines in rank0.log) is never restarted;
   - slot containers (`tf.prof=1`) running without a flag are removed after 15 minutes.

   During a slot it stands down, and ends the slot 15 minutes past its end. `touch opt/overnight/STOP_WATCHDOG` stops it.
2. **Slot tools.** Main runs slots only with `slot_begin.sh HOLDER MINUTES "WHAT"` (writes `SLOT_ACTIVE` first, then stops production), `slot_extend.sh MINUTES` and `slot_end.sh` (restores with `./start.sh`, health-checks, then retires the flag). They log to `opt/overnight/slots.log`.
3. **Notes.** Update `STATE.md`, `calendar.md` and `BRIEF-COMMON.md`: 400 W, no deadline, the tracks and slot sequence above, the reserve queue.
4. **Worktrees** from `opt-integ2` @3f94715d, branch `opt-n1-<track>`, numbered patches from 0067: n1-exl3r, n1-exl3p (new), n1-pdl5 (new), n1-glue5, n1-graph5, n1-attn5, n1-gate5, n1-integ5. `prof5` needs none.
5. **Health check:** "What is 17*23? Answer with just the number, then name the capital of Australia." at T=0; the reply must be `391 | Canberra` (`opt/overnight/health.sh`).

## Track briefs

### T1 Profile, L6, L7 and T6 (`prof5`, slot 1; then runs the A/B slots)

- nsys on the production config (`opt/overnight/prof_up.sh`, image 8d045cb2), both ranks, warm, profiler start/stop around exact ranges:
  1. a plain decode step, and forced windows of 1, 2, 4 and 8 rows (`TF_GLM_DFLASH2_FORCE`, `TF_GLM_DFLASH2_TRACE=1`); 16 rows from copy windows;
  2. the DFlash2 block pass, including the host selector hop;
  3. 4 concurrent streams (shared windows);
  4. decode at ~200K context, and the select's share of the step;
  5. prefill at 16K, 64K and 200K **without `TF_GLM_PROF`** (it disables pairing).
- **L7:** bounded NCU on microbenchmark kernels (EXL3 grouped, a large and a small GEMV, KDA step, select, attention): DRAM read and write bytes, duration, clocks, cold vs warm. Then a mixed-traffic run (one stream prefills a long prompt while others decode): aggregate tok/s, per-request TTFT, token-gap p50/p95/p99, chunk duration and pairing state.
- **Check first:**
  - both GPUs' memory clock under load against its maximum, the P-state, SM clocks, temperatures and limit reasons; the physical rank mapping under `GPUS=1,0` (400 W unchanged);
  - the select under crowded or all-equal bins and its other dispatches (today's path rescans the eligible range in four radix rounds, `glm_glue_index.cuh:119`, with no candidate-bin clipping);
  - the P2P gather's epoch is host-incremented (`cuda/p2p_gather.zig:124`): fine eagerly, but device epochs and wrap/parity checks are prerequisites before any graph captures a gather (T4).
- **L6:** `TF_GLM_INDEX_CUDA` bracket `off → scores → scores → off → both → both → off` on cold ≥128K prompts, unprofiled; check each rank's log shows the requested mode loaded (it can fall back); glm-index-check for scores, pools and token lists; compare a generated continuation's hash, not just the first token.
- **T6:** fit `TF_GLM_VERIFY_MS=a,b,d` from the forced-width sweep **and** measured shared-window tables (today's are GB10's `52 + 6.8·(R−1)` and `76 + 4.1·(R−1)`). Replay before engine time, then ABBA `cut` vs `tpm` on prose, code, file-edit and 4-stream sets.
- Deliver `profile5.md`: a per-rank table by category in ms per step and per row; idle gaps and rank skew; which prompt kernels are SIMT; a sizing line for every track and reserve; a recommendation whether T5(a) moves first in slot 3.
- Owns the shared A/B harness (`opt/integ/run_ab.sh` with `leg.sh` and `table.py`, copied to call `opt/overnight/prof_up.sh`). Candidates run as host binaries bind-mounted over `/opt/tensorfold/native/bin/tensorfold-native`.

### T2 EXL3 decode (`exl3r`)

- Re-tile the grouped decode path (`exl3_grouped.cuh` `warp_tiles`): 16 contiguous bytes a lane and 1–4-step staging, as patch 0028 deferred. #1505 and #1534 are already in.
- Reference: ExLlamaV3's GEMV (`ld.global.cs` evict-first stream, a 2–4-deep register prefetch ring, lane shuffles for trellis windows, no block barriers in the main loop). Keep our MMA accumulation, split boundaries and reductions; its fp16 accumulation cadence is not an exact replacement.
- Bit-identical by construction; 0027's planted-fault contract tests.
- Microbenchmark on `decode4_bench/bench_exl3_v2`, cold weights, 1/8/16/32 members, every width in the 4 bpw mix, ≤1.5 GB a GPU under `gpu.lock`.

### exl3p EXL3 prompt experts (`exl3p`, new)

- Target `tf_exl3_prompt_gu` / `_down` (production's 4-bit path).
- First an item histogram at 2048-row passes (pairs per expert, items per expert).
- Then, at TPM=64: specialize the loop for an item's live member blocks; keep two decoded fragment sets and interleave the next tile's decode with the current MMAs; hoist invariant address/mask work the SASS confirms survives. Reference: the old Python recipe's K12 (`patches/aevonix/0073`); it covers 4-bit shared-input only.
- TPM=128 is a separate experiment after the first variant: at three stages it needs 70–74 KB of smem and 128 accumulator floats a thread, halving residency.
- Exact: each output keeps its ascending K16 MMA chain and the same Hadamard, activation, scale and rounding epilogue. Check intermediate and down fp32 outputs, not just tokens.

### T3 Decode glue (`glue5`)

Wait for T1's sizing, then pick the largest eligible candidate, one at a time. Tie-break order:
1. Batched greedy draws across a shared window (one gather/read of all ≤32 rows, then a row-grid top1).
2. SK-preserving multi-problem projection launches. Concatenating `q_b` (SK 2) and index `qb` (SK 4) into one N=12288 GEMV changes SK and is not exact; `[q_a|kv_a]` + index `kw` keeps SK 8.
3. The residual HC fold (ExLlamaV3 `EXL3_HC_FOLD`) with a schedule that fixes decode2's loss (6.3 vs 3.5 µs): at most 89 boundaries, 84 with the drafter taps kept explicit. The in-place read/write hazard must be handled.

Measure with decode.md's `bench_mm.cu` shapes and `bench_moe_glue2`.

### T4 DFlash2 block-pass graph (`graph5`)

- First move DFlash2's path selector walk from rank 0's host onto the GPU (one block per row, the same tie rule as `dflash_select.zig`), as vLLM and ExLlamaV3 do. That removes a host round trip each round and makes the pass capturable. Check the selector's picks byte for byte, including sampled (keyed) races.
- Then capture the block pass (5 layers, 8 rows, a split MLP with per-layer gathers) per stream count, 1 to 4, using the `cuda/graph_cache.zig` pattern. The P2P gather's epoch is host-incremented (`cuda/p2p_gather.zig:124`), so either the gathers stay outside the graph (piecewise capture) or they move to device epochs with wrap/parity checks first. Host-side abort checks stay outside, or are proven capture-safe. Graph memory goes into the fit.
- Bit-identical: the same kernels replayed.

### pdl5 Dependent launch along the decode chain (`pdl5`, new)

- **Why:** only `glm_mm` launches with PDL and almost nothing upstream of it triggers early, so its "weights before the wait" rarely overlaps. llama.cpp #22522 measured +22% decode on Qwen3.5-35B-A3B (MoE) on an RTX PRO 6000 with PDL on every decode kernel, bit-exact. Our setup differs (no graphs, ~1,500 kernels a step), so that number doesn't transfer.
- **SASS gate, before any timing:** with `cuobjdump`/`nvdisasm`, inspect the modules both ranks actually load. Every module that gains a wait must be native sm_120 SASS with the dependency wait present and correctly ordered. Check module identity and dispatch; build flags alone don't exclude stale artifacts.
- **First chain:** after an explicit gather boundary, main-stream `hc_post → hc_partial8 → hc_finish → the next glm_mm*`, including `glm_mm_reduce` wherever SK > 1. Stop before stateful KDA/cache work and before the MoE fork. First give `glm_mm_reduce` the dependency wait it lacks (`glm_mm.cu`), together with its dependent launch.
- **Hazard rule, kernel by kernel** (Codex lanes round 1's table): immutable loads and private setup may run before the wait; every scratch read or write comes after it. Known write-after-read cases: the shared `mm_part`, EXL3 `sc.z`, `normed`/`post`/`comb` and `hcPost`'s in-place `x`, router scratch, KDA saves/state, index-cache rings, the prompt twins' site scratch, and the W8A8 quantizer. Events, P2P/NCCL and host control stay explicit boundaries. The shared expert's side stream is excluded at first.
- **P2P gather:** it runs its own dependency wait, then triggers its dependents before exchanging data; consumers still wait before reading gathered rows. Keep flags, fences, parity, buffer ownership and abort handling.
- Benchmark trigger placement against implicit completion; a late trigger rarely helps. Correctness must hold if overlap never happens.
- Regressing increments stay disabled. Collective and compute gains get one combined measurement.

### T5(a) DSA select (`attn5`)

- First: route decode to the existing exact pool selector `tf_glm_index_pools` (it samples a floor, keeps survivors and falls back to a full-row select when the set is short or overflows), keeping today's decode scores. Its kernel has no minimum-row requirement; only the host's combined dispatch gates it at 64 rows.
- If that doesn't pay: an exact multi-CTA span select plus merge. Reference SGLang TopK-V2 (cluster + DSMEM histogram; 1M 372 → 36.6 µs on GB300-class). Require bounded threshold-bin storage with overflow detection and an exact fallback, lower-index-first ties at every tie count, unchanged fp32 score-key semantics, the visibility boundary and incomplete-pool tail, and ascending output. Check sm_120 cluster support and residency.
- Size at 128K, 200K and 864K, R=1/4/16, including tie- and fallback-heavy cases. Microbenchmarks first; don't spend the night building max-context engine baselines.
- Then R2 (T5(b)) from the reserve queue.

### T7 W8A8 quality gate (`gate5`)

- Build `opt/overnight/gate/`, a paired functional A/B against the engine's API at T=0, thinking off:

  | Stratum | Items |
  | --- | --- |
  | Math | GSM8K test (1,319) and MATH-500 |
  | Code | HumanEval+ (164) and MBPP+ (378) |
  | Knowledge | MMLU-Pro, a 2,000-item sample |
  | Long context | synthetic multi-key retrieval at 16K–128K; W8A8 touches prompt passes only |

- **At most 3 requests in flight against 8888**, so the health check always finds a free stream.
- Generated code runs **only** in `docker run --network none --memory 2g --cpus 2` with a timeout per test.
- Freeze the policy first: `TF_GLM_W8A8_SKIP_K` agreed and fingerprinted, the rows ≥ 32 contract. Run a disjoint paired pilot (baseline on production, W8A8 in a short slot P), then size N from its discordance and freeze the sample before the main arms. 300 retrieval prompts at a 64K average alone take ~58 min of prefill per arm, so size within the ~2 h cap per arm.
- Decision rule per stratum: the one-sided 95% UCB of the paired accuracy loss < 1 point. A stratum that can't conclude is reported inconclusive, not passed.
- Reuse `/home/zurih/glmeval`'s stratum names where they fit.

### T8 Integration (`integ5`)

- Starts after two candidates pass their own ABBA, or at the end of the queue with one.
- Stack every passing candidate onto `opt-integ2`; run the host suites (`opt/integ/host_tests.sh`); resolve conflicts; build host binaries and the image `tensorfold-glm53-zig:n1-<hash>`.
- The integration slot runs the combined ABBA, mixed traffic and a context refit, and records the pool line. Remove or revise any regression and retest the combination. The recipe's `patches/engine` stays untouched.

## Rules for the night

- `WAVE2-RULES.md` applies where these don't override it:
  - one slot at a time, granted by main; main alone stops and restores production;
  - after every slot, restore and health-check (`slot_end.sh`);
  - nobody edits `start.sh` or a running script;
  - builds run at `nice -n 19 -j8` into `toolchain/install/tensorfold-opt-n1-<track>`.
- Microbenchmarks with production up stay within 1.5 GB a GPU and take `flock opt/overnight/gpu.lock` for timing runs. None run during the eval window.
- **Ship rule:** an exact patch needs identical greedy hashes in a same-slot ABBA plus a gain larger than twice the A-to-A spread, plus its operator checks. Numerics-changing work is opt-in and off; T7 judges W8A8 by NLL against the null spread and by the gate.
- **45-minute no-progress rule** for active work; planned waits for a lock or a slot don't count. A stopped track reports its evidence and releases its claim.
- If a restore can't reach health, main stops all further GPU work and reports.
- No commits or pushes. No power-limit changes. spark1 stays read-only, apart from Codex's `~/codex-glm53-sm120`. No secrets in any file or log.
- Everything persistent goes in `opt-sm120-notes/opt/overnight/`; `/tmp` is wiped on reboot.

## With you, afterwards

1. Read `opt/overnight/REPORT.md`. For each candidate: exact or not, hashes, the change in decode, prefill and memory, and a verdict; the T7 result per stratum; anything stopped and why.
2. **Ship decision:** move passing patches into `patches/engine` and rebuild with `./start.sh`, or keep today's image.
3. **W8A8:** enable it only if every stratum passes the gate.
4. **Power-split A/B, ~15 minutes.** You run:

   ```
   sudo nvidia-smi -i 0 -pl 440 && sudo nvidia-smi -i 1 -pl 360
   ```

   Then main measures decode and prefill. Revert with `-pl 400` on both, or make it persistent in `nvidia-power-limit.service`. Expect at most ~4.7%, and only if speed follows the clock.
5. **The watchdog keeps running after the night** until `opt/overnight/STOP_WATCHDOG` exists or 72 hours pass. Main touches the file after the final restore; if you stop the server yourself before then, touch it first, or the watchdog restarts it.
6. Whether night 2 takes the reserves that didn't run, and anything that slipped.
