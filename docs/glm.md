# GLM-5.3-Flash on 2× RTX PRO 6000: decode and prefill performance report

**Date:** 2026-10-30. **Scope:** analysis only. **No code was changed** — this report is the deliverable; production on port 8888 (commit a88386c) was not touched, stopped, or measured against.

## 0. Method and evidence base

Everything below was read from this repository only, per the task's constraint:

| Source | What it gave |
| --- | --- |
| `docs/handoff.md` | production state, measured numbers, user decisions, house rules |
| `README.md` | architecture of the serving stack, patch group map, memory model |
| `scripts/config.sh`, `scripts/local.sh` | every live setting and its measured effect comments |
| `start.sh` | build/launch flow, context-fit logic, rank placement |
| `docker/Dockerfile` | toolchain: CUDA 13.3.1, cuBLAS 13.6, NCCL 2.30.7, Zig 0.17.0 `-Doptimize=fast`, sm_120, no cuDNN |
| `patches/engine/0001–0066` (read in depth: 0005, 0006, 0011, 0015, 0017, 0027, 0028, 0030, 0034, 0035, 0036, 0037, 0038, 0040, 0041, 0048, 0050, 0064; headers/measured-lines of the rest) | the actual implementation and every measured microbenchmark |
| `logs/current/rank0.log` (startup lines, read-only) | the measured production footprint: `model 82.04 GiB; 4 streams … 1757 pages of 512 rows (5.35 GiB) … 395 MiB fixed each, of 6.5 GiB left after a 3.5 GiB reserve; prompts in passes of up to 2048 rows` |
| Web (where cited) | upstream technique precedents |

Two caveats on completeness: the engine source itself lives in worktrees outside this repo and was **not** read (everything engine-side here comes from the patches, which contain large verbatim hunks of it). Live search failed for two upstream queries (FlashInfer sm_120 status, TP clock-skew literature), so those specific claims are marked *unverified* and rest on the repo's own measurements plus first principles.

## 1. The system as built (verified)

GLM-5.3-Flash served by TensorFold's native Zig/CUDA engine (`tensorfold-native`, upstream f350cd3e + 66 patches), TP=2 in one container, one rank per RTX PRO 6000 (sm_120, PCIe Gen5 x8, 400 W caps). Per rank: **82.04 GiB of weights** (4 bpw EXL3), 4 streams, one shared fp8 KV page pool of 1757 × 512-row pages = 899,584 tokens (5.35 GiB), 395 MiB fixed per stream, 6.5 GiB left after the 3.5 GiB reserve. Prompts run in passes of up to 2048 rows. The device prompt cache holds nothing (`0.0 GiB of device pages`); the entire prompt cache is the 8 GiB pinned-host tier per rank — which is why follow-up turns of a 125K chat take 0.2–0.6 s (host→device page restore, a byte copy).

Model shape constants found in the patches: hidden 4096; 32 attention heads (16 per rank at TP2) over a 512-wide latent (MLA-style, `LW=512`, 512-key chunks, `KT=32`); KDA (gated-delta) linear-attention layers whose per-window recurrent state is ~203–222 MB per stream (patch 0046) — the bulk of the "395 MiB fixed"; a DSA sparse-attention indexer (0022, 0051); MoE with **top-8 of ≤512 routed experts plus 1 shared expert** (`top_k = 8`, patch 0001/0007/0030), routed expert projections ~4096→2048 per rank (gate/up) and 2048→4096 (down); dense FFN gate/up ~4096→12576 per rank (patch 0064's measured shape); LM head bf16; dense linears q8. Routed experts are ~95% of rank memory, so **decode is dominated by streaming routed-expert trellis weights**.

Current numbers (handoff, 66 patches, ABSORB_MMA=1): plain decode step (no drafter) **7.59 ms**; prose ~170 tok/s, code ~277, file-edits ~514 (copy drafts); prefill 5,470–5,553 tok/s at 16K–64K.

### 1.1 The decode/verify pipeline

With DFlash2, each round drafts a chain (block pass = 7 drafts, drafter ≈ **1.37 ms**, patch 0038) and verifies a window of `rows` (1–8, 16 for copy rounds — patch 0039). The engine's own measured cost model (patch 0038, at the 0005–0013 build):

```
verify_ms ≈ 9.4 + 1.31 × rows        draft_ms ≈ 1.37
```

The `tpm` window rule (0036/0038) picks the window width that maximizes expected tokens/ms using online-calibrated acceptance priors (measured on 1,554 forced-width windows; high-chance buckets land 0.86–0.98, mid buckets 0.44–0.62). Multiple streams share one verify window of up to 32 rows (patches 0034/0036/0050 context). Exactness is preserved: replies are always the target's picks.

Per-step TP exchanges: **~90 small all-gather sites** per decode-shaped chain (patch 0034's own bench shape), carried by the one-shot IPC peer-store gather at **3.3 µs exposed per call at 1 row, 8.4 µs at 8 rows** (vs NCCL's 6.1/16.9). Cap 512 KiB/rank = 32 rows × 4096 fp32. Prefill exchanges go over copy engines (`CE_EXCHANGE=1`, 26.4 GB/s each direction both ways, +4.5% prefill).

### 1.2 The prefill pipeline

2048-row passes; paired prompt chunks (twin buffers, exchanges and glue of one chunk under the other's sites — patches 0049–0063, +6–9% at 16K+, ~3% *slower* at 4K per 0059); one expert call per pass (0007: 4.84 → 3.50 ms per 2048 rows); two-stage packed prompt GEMMs (0009: 1.07–1.09×); `ABSORB_MMA=1` puts the MLA absorb_q/expand_v projections on tensor cores (0041: 1.212→0.113 ms and 1.471→0.093 ms at 2048×32 heads, i.e. 10.7×/15.8×, from SIMT kernels that run at **11–13 TFLOPS**; +6.6% prefill end-to-end); `W8A8` (off) adds int8×int8 IMMA prompt linears (0064: 2048×12576×4096 in 0.677 vs 1.114 ms including quantization; **+10% prefill alone, +19% with ABSORB_MMA** per config.sh).

## 2. Where the remaining time goes

**Decode (7.59 ms plain step; verify ≈ 9.4 + 1.31·rows).** Decomposition from measured pieces, with layer counts not stated anywhere in the repo (flagged as estimates):

1. **Routed-expert weight streaming** — 8 experts × ~25 M params × 0.5 B ≈ 100 MB/layer/token/rank, on the EXL3 trellis decode path (grouped kernels, 0030). At ~1.8 TB/s HBM this is the single largest term and explains why the marginal row costs only 1.31 ms (weights amortize across the window's rows via grouping) while the fixed step is ~7.6 ms.
2. **Shared expert + dense + KDA projections + LM head** — the shared expert runs on a side stream (0006: 70.6→60.6 µs/MoE layer at 1 row); KDA f_b/g_b run as one paired launch (0030); dense is q8.
3. **Latent attention decode** — hand-rolled SIMT fp32-FMA streaming kernels (`chunks_wide`, 167 registers, 101,120 B shared → **1 block/SM**), fp8 keys prefetch-ahead (0005: dense chunk 58→35 µs, sparse 72→41–44 µs). Small at short context, grows linearly with it; bounded by the DSA sparse selection at long context.
4. **TP exchanges** — ~90 × 3.3–8.4 µs ≈ **0.30–0.76 ms/step**, already far below NCCL.
5. **Launch/scheduling overhead** — the engine has **no CUDA graphs** (no occurrence anywhere in patches/engine). PDL exists but only for the `glm_mm` kernels (`pdl: bool`, TF_GLM_PDL, default on: "a matmul's weights stream in while the kernel before it finishes"). At ~90 gather launches plus per-layer kernel sequences, the un-hidden launch tail is plausibly several hundred µs to ~1 ms. *Estimate — the one number nobody has measured.*
6. **The verify fixed cost** — 9.4 ms seed vs the current 7.59 ms plain step. The seeds predate wave 3 (plain step was 8.44 then), so today's verify-specific fixed cost (window heads, tap ingest, bookkeeping) is ~1–1.8 ms and **the seeds the tpm rule starts from are stale**.

**Acceptance, not the engine, caps prose decode.** At the priors' mid buckets (~0.5 landing), 170 tok/s is what the cost model predicts. File-edits hit 514 tok/s because copy drafts land ~0.9+. So decode gains come from: (a) cutting the fixed and marginal window costs — engine work; (b) pricing windows correctly — calibration; (c) raising acceptance — drafter-side, mostly fixed by DFlash2's weights.

**Prefill (≈369 ms per 2048-row pass at 64K).** Measured per-layer pieces: experts 3.50 ms, dense gate/up 0.677 ms (W8A8) / 1.114 (bf16-rows), absorb+expand 0.113+0.093 ms (ABSORB_MMA), plus chunk attention, indexer, KDA, exchanges (CE), glue. What is still SIMT: **the chunk attention QK/softmax/PV itself** (same 11–13 TFLOPS kernel family ABSORB_MMA attacked for the projections only) and the DSA indexer's scoring pass. Those are the unclaimed prefill FLOPs.

## 3. Opportunities, prioritized

Format: what / evidence / expected / risk & gate / how to validate. House rules apply to all of them: exact patches ship only with identical greedy hashes in a same-slot ABBA A/B plus a measured gain; numerics changes are judged by NLL against the null spread (`opt/num3/qh.py`), not KL; quality-risk items are gated on glmeval (~2,000 paired tasks per stratum, <1 point loss); opt-in numerics flags stay off unless the user turns them on.

### P0 — bounded, high-confidence, do first

**P0-1. Recalibrate the draft-rule cost seeds and re-profile the verify fixed cost (no code).**
The `tpm` rule's seeds (9.4 / 1.31 / 1.37, patch 0038) were measured at the 0005–0013 build. Wave 3 alone took the plain step from 8.44 to 7.59 ms; every downstream cost moved. Stale seeds misprice every offer a stream sees before its online calibration warms up — on a fresh stream that is its first dozens of windows, and `lambda = 0.15` (tokens/ms floor) is seed-derived too. `TF_GLM_VERIFY_MS=a,b,d`, `TF_GLM_DFLASH2_FORCE=list` and `TF_GLM_DFLASH2_TRACE=1` (all shipped in 0036/0038) make this a pure measurement pass on `prof_up.sh` with `IMAGE=tensorfold-glm53-zig:8d045cb28b66`.
*Expected:* better window widths at every acceptance level; also produces the fixed/marginal split that sizes P0-2 and P0-3. *Risk:* none (env-only, off by default). *Validate:* trace lines before/after; decode tok/s at T=0 on the standard bench prompts.

**P0-2. Bounded CUDA-graph capture of the decode/verify step.**
This is the handoff's own "bounded CUDA-graphs test" idea, now with concrete evidence: no graphs exist anywhere in the patch set; ~90 gather launches per step; PDL covers only `glm_mm`. vLLM's default is FULL_AND_PIECEWISE graphing of decode precisely because launch overhead dominates small-model/MoE decode ([vLLM CUDA graphs design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)); production guides report the same ([launch-overhead practices](https://www.oh-bug.com/posts/llm-cuda-graph-production-practices/)). The engine's sequencing already avoids host syncs (flag waits via `cuStreamWriteValue32`/`WaitValue32`, patches 0034/0035; rank-side "no sync" verify, 0048) — stream memops and the p2p gather's epoch/parity protocol are graph-capturable patterns, and window shapes are enumerable (rows 1–8, 16, shared 32).
*Expected:* 0.3–1.0 ms off the fixed step (4–13% decode at current acceptance); grows with P0-1's windows. *Risk:* capture-unfriendly sites (rule logic, host-side lane bookkeeping) — bounded by graphing only the per-layer body, piecewise, eager outside; the p2p timeout/abort path needs its host check to stay outside the graph. *Gate:* bit-identical greedy hashes (graphs replay the same kernels; feasible). *Validate:* the ABBA harness at rows 1, 8, 32; nsys launch-gap count before/after.

**P0-3. Finish the EXL3 decode-tile work the engine's own note defers.**
Patch 0028 says it outright: production's decode kernel reads **16 contiguous bytes per thread** and **stages 1–4 steps ahead through a warp slice**, while the grouped decode path (`warp_tiles` in `exl3_grouped.cuh`, which wave 3 made the decode path per 0030) got the load-ahead (0027) and the `L1::no_allocate` policy (0028) but "both need a re-tile rather than a load swap, so they are separate work." The routed-expert stream is the dominant decode cost (§2), so the re-tile attacks the biggest term directly, bit-identically (same 32-bit values, same warp-order sums — the 0030 kernels already prove order preservation is checkable).
*Expected:* the 0005 precedent (same technique on the attention path: 58→35 µs, a 40% cut) suggests high-single-digit to low-double-digit % off the marginal row cost — i.e. directly on the 1.31 ms/row term. *Risk:* low (bit-identical by construction; contract harness exists — 0027's planted-fault controls). *Validate:* `decode2_bench` cold-weight tiles, then ABBA greedy hashes at rows 1/8/32.

### P1 — bigger swings, gated

**P1-1. Expert-parallel MoE for decode (whole experts per rank instead of TP-sharded experts).**
Today each rank holds half of *every* expert and both ranks stream all 8 selected experts' halves — ~100 MB/layer/token/rank. EP (rank 0 hosts experts 0–255, rank 1 the rest) halves that stream: ~4 experts' worth, ~50 MB. The exchanges that replace the gathers are the same size class as today's: per MoE layer, an all-gather of the window's hidden rows (32×4096 bf16 = 256 KiB — inside the 512 KiB peer-store cap) and a combine of fp32 partials (512 KiB, ~8 µs). Upstream precedent is strong: TensorRT-LLM's EP scaling work ([blog14](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html)) and DeepSeek-V4 on Blackwell (symmetric-memory fused dispatch/combine, PDL — [write-up](https://jianyuh.github.io/systems/inference/2026/07/20/DeepSeek-V4-TensorRT-LLM-Blackwell.html)).
*Expected:* if experts are 60–75% of the step (§2), halving their stream is **+15–30% decode** — the largest single decode lever found. *Risk:* high effort (weight layout at load, router dispatch, the whole `exl3_experts` pipeline; prefill would stay TP initially); **not bit-identical** — the down-projection's fp32 sum order changes, so this is an opt-in numerics path needing the qh.py NLL gate and the user's explicit opt-in; the same-slot ABBA rule then applies to hashes-under-flag. *First step:* prototype one MoE layer in `decode2_bench` with the existing p2p gather before any engine integration.

**P1-2. Tensor-core (and optionally fp8) chunk attention for prompt and long-context decode.**
The chunk attention (QK^T, softmax, PV over 512-wide latents) is still the SIMT fp32-FMA kernel at 11–13 TFLOPS and 1 block/SM; ABSORB_MMA (0041) moved only the absorb/expand *projections* to MMA and bought +6.6% prefill. The same treatment for the chunk body (bf16 `mma.m16n8k16` with the same fp32 sum-order discipline, as 0041 did) is the remaining attention lever; keeping keys in E4M3 with per-block scales (SageAttention-style, [repo](https://github.com/thu-ml/SageAttention), [paper](https://arxiv.org/html/2410.02367v9), [SageAttention2++](https://arxiv.org/html/2505.21136v1)) would additionally halve attention bandwidth at long context, where the KV stream grows linearly.
*Expected:* must be sized by one profiling pass first (the engine's `prof` spans exist — router/experts are already instrumented in `cuda_moe.zig`); plausibly +5–15% prefill at 64K and a slice off long-context decode. *Risk:* changes summation order (opt-in flag, ABSORB_MMA precedent); the 101 KB smem budget constrains the MMA tiling — the HB16 instance (125 regs) has headroom, HB32 may need a tile-shape change. *Gate:* NLL null-spread + user opt-in; ABBA hashes under flag.

**P1-3. The GPU0 clock deficit: prefer the user's power decision over a re-split.**
Rank 0 waits **~1.2 ms/token** for rank 1 on the desktop-driving, ~10% slower GPU0 (handoff). Equalizing with an uneven TP split needs δ ≈ 2.5–3% of per-layer work moved to rank 0 — about **2–2.5 GiB of weights**, which comes straight out of the shared KV pool (5.35 GiB) and lands the context cap near **~500–550K instead of 864K**, colliding with the user's "keep the context as large as it will go" decision. A split limited to attention/KDA/head placement moves too few bytes to close the gap. The cheap lever is the already-pending user decision: **asymmetric power caps (e.g. 440 W on GPU0 / 360 W on GPU1)** — per the power rule, hand over the command (`sudo nvidia-smi -i 0 -pl 440 && sudo nvidia-smi -i 1 -pl 360`, then make persistent in `nvidia-power-limit.service`) and let the user run it. *Expected if approved:* recovers most of the 1.2 ms (~+10–15% decode) for zero memory cost. A skew-aware *scheduler* fix (the handoff's "light uneven TP split") should be sized only after P0-1's profiling shows where rank 1's step actually lengthens.

**P1-4. Ship the refined W8A8 (GS32, MLP-only) behind the agreed gate.**
Everything needed is specified and half-built: 0064 already has `TF_GLM_W8A8_SKIP_K=1024,6144,8192` (diagnostic) and the handoff names the two prerequisites — `SKIP_K` into the rank fingerprint (0041 set the fingerprint precedent, layout 13) and a rows ≥ 32 guard so decode windows never take the int8 path. Measured upside: **+5.5–6% prefill over ABSORB_MMA**. Gate: the glmeval harness (~2,000 paired tasks/stratum, <1 point) that round 4 asked for; user opt-in. This is the highest-confidence prefill item that does not require new kernel research.

**P1-5. Deeper drafts for high-acceptance streams: chain the MTP head past DFlash2's 7.**
DFlash2's block pass yields exactly 7 drafts, capping the window at 8 rows; the measured priors show high-chance buckets landing 0.86–0.98, so prose/code streams leave acceptance on the table (file-edits at width 15 already prove the ceiling: 514 tok/s). The checkpoint's MTP head exists and chains to `mtp_rows`; patch 0014 deliberately frees it beside DFlash2 (~2 GiB/GPU back). An opt-in hybrid (MTP extends DFlash2's chain only for streams whose calibrated acceptance is high, priced by the same tpm rule) buys depth where it pays and stays off elsewhere. *Expected:* raises the width ceiling for prose/code; sized by the FORCE sweep first. *Risk:* the 2 GiB returns as fixed cost and must come out of the pool (context trade-off — same currency as P1-3); numerics unchanged (drafting only, verification stays exact).

### P2 — capacity, concurrency, decisions

**P2-1. Grow-on-demand KV admission** (design exists: `opt/mem3.md` §4). Admit by prompt + min(max_tokens, --max-tokens), grow pages as needed. Raises how many long requests fit at once — a throughput-at-concurrency lever, no per-request speed change. Endorsed as-is.

**P2-2. Copy-on-write prefix sharing.** A shared system prompt stored once. The paged pool and the snapshot machinery (0015/0017–0025) give it a natural home; biggest win is capacity again, plus faster multi-stream starts.

**P2-3. Decisions that remain the user's** (listed, not advocated): q4 dense (−1.06 ms/step, quality risk); the 3.5 bpw GLMQ quant (−9 GiB/GPU, kept for reference only — would buy pool, not speed, since gate/up is 3-bit in only 32 layers); `--recall` (1.5–3× repeated prompts, writes token ids to disk); the power caps above.

**P2-4. Hygiene found while reading** (no action taken): `.state/context-fit.tsv` holds three superseded fit rows before the current 864,256 ones (harmless — keys differ); `bench.py`'s hard-coded port 8020 → 8888 (already in the handoff); the paired-chunk ~3% regression at 4K prompts (0059) could use an explicit pairing threshold check at the production mix.

## 4. What would *not* pay (checked so it stays checked)

- **NCCL-class comm work** — the p2p gather (0034) and CE exchange (0035) already sit at 3.3 µs and 26.4 GB/s; NVLink-only tricks (multimem/NVLS, symmetric-memory all-reduce) don't exist on PCIe PHB peers.
- **q8 LM head** — measured +1–2%; user decision stands.
- **Moving the monitors off GPU0 / trading context for cache** — explicitly out by user decision; note the device prompt cache is already empty by design (the pool takes it), so the host tier is the only cache and costs nothing the pool wants.
- **Drafter-pass micro-overlap** — the block pass (1.37 ms) is already batched across streams and skipped for copy rounds (0040, +5.3–5.6% file-edits); its chain dependency makes further overlap marginal at batch 1.
- **Batch-size growth for its own sake** — the 32-row shared window already amortizes the fixed cost; beyond that, acceptance priors, not capacity, bind.

## 5. Upstream survey (what was adaptable, what was already here)

| Technique | Upstream | State here |
| --- | --- | --- |
| Piecewise/full CUDA graphs for decode | [vLLM design](https://docs.vllm.ai/en/stable/design/cuda_graphs/) | **absent** → P0-2 |
| PDL / dependent launches on Blackwell | [CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html), [CUTLASS](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html), [TRT-LLM blog14](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html) | present for `glm_mm` only (`TF_GLM_PDL`) → widen in P0-2 |
| Fused MoE dispatch/combine, symmetric memory | [DeepSeek-V4 on Blackwell](https://jianyuh.github.io/systems/inference/2026/07/20/DeepSeek-V4-TensorRT-LLM-Blackwell.html) | partial (one-shot IPC gathers) → P1-1 |
| Small-message custom all-reduce over IPC/PCIe | [vLLM custom_all_reduce](https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/custom_all_reduce/) | **done, ahead of stock** (0034: 3.3 µs one-shot gathers) |
| fp8 attention (per-block scales) | [SageAttention](https://github.com/thu-ml/SageAttention) | keys stored E4M3 but converted to bf16 for math → P1-2 |
| QTIP/EXL3 trellis decode kernels | [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [EXL3 system](https://deepwiki.com/turboderp-org/exllamav3/4-exl3-quantization-system) | own implementation (mcg, mixed widths, grouped tiles); the 16 B/staging re-tile deferred → P0-3 |
| Draft-depth/acceptance pricing (EAGLE-2/3 ideas) | [EAGLE-3](https://arxiv.org/abs/2503.01840) | tpm rule with online calibration already; stale seeds → P0-1; depth ceiling → P1-5 |
| FlashInfer sm_120 coverage | — | **unverified** (search failed); worth one look when the network allows |

## 6. How to validate each item under the house rules

1. One slot at a time (`prof_up.sh` with `IMAGE=tensorfold-glm53-zig:8d045cb28b66`), restore with `./start.sh` + the `391 | Canberra` health check after every leg.
2. Exact patches: same-slot ABBA A/B, greedy hashes identical on all T=0 prompts (`opt/abbench.py` / `opt/abtable.py`), plus the measured gain.
3. Numerics flags: `opt/num3/qh.py` NLL against the null spread, then answers — never KL.
4. Quality-gated items (W8A8, q4): the glmeval harness first (~2,000 paired tasks per stratum, <1 point loss) — the harness itself is unbuilt and is the gating prerequisite.
5. Keep `MEMORY_RESERVE_GIB=3.5`; any proposal that shifts weight bytes between ranks must re-run the context fit and show the pool line from rank0.log.

## 7. Order of attack (summary)

| # | Item | Decode | Prefill | Effort | Gate |
| --- | --- | --- | --- | --- | --- |
| 1 | P0-1 reseed/re-profile the rule (env-only) | +0–5% | — | hours | none |
| 2 | P0-3 EXL3 16 B + staging re-tile (bit-identical) | +5–12% | — | days | ABBA hashes |
| 3 | P0-2 piecewise CUDA graphs + wider PDL | +4–13% | ~0 | days | ABBA hashes |
| 4 | P1-4 W8A8 GS32 MLP-only behind glmeval | — | +5–6% | days | fingerprint+guard, glmeval, user opt-in |
| 5 | P1-2 MMA/fp8 chunk attention | +long-ctx | +5–15% | 1–2 wks | NLL gate, user opt-in |
| 6 | P1-1 expert-parallel MoE decode | +15–30% | later | wks–month | prototype, NLL gate, user opt-in |
| 7 | P1-3 power caps 440/360 (user runs it) | +10–15% | ~0 | command | user + sudo |
| 8 | P2-1/P2-2 KV admission, CoW prefixes | concurrency | concurrency | wks | design review |

Items 1–3 are engine-internal, bit-identical or config-only, and compound on the fixed cost that every draft row rides on; 4–6 are the gated numerics/structure plays; 7–8 are the user's calls. Nothing above touches the supported context length or the KV pool except where a trade-off is stated explicitly (P1-3, P1-5).

*No code, configuration, or production state was changed in producing this report.*
