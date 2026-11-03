# GLM-5.3-Flash on 2× RTX PRO 6000: decode and prefill performance report, v2

**Date:** 2026-10-31. **Scope:** analysis only. **No code, configuration, or production state was changed** — this
report is the deliverable. Production (port 8888, image `tensorfold-glm53-zig:987b580ba9d2`, 91 patches, commit
3a39a84) was not touched, stopped, or measured against.

This is the successor to `docs/glm.md` (v1, written 2026-10-30 at the 66-patch state). It supersedes v1's priority
list: the 2026-10-31 overnight run landed 25 more patches (0067–0091), produced a fresh nsys/NCU profile of the
production path, and tested-and-dropped a long list of ideas with numbers. Everything here is aligned to that
record; items v1 got wrong are corrected in §3 rather than quietly dropped.

## 0. Method and evidence base

| Source | What it gave |
| --- | --- |
| `docs/handoff.md` | production state, measured numbers, user decisions, house rules |
| `docs/glm.md`, `docs/deepseek.md`, `docs/overnight-plan.md` | v1 analysis and its corrections; the night's plan |
| `opt-sm120-notes/opt/overnight/REPORT.md` (read via shell, read-only) | the night's verdicts: what shipped, what was measured and dropped, with numbers |
| `opt-sm120-notes/opt/overnight/prof5/profile5.md` | the fresh profile: per-category step tables, NCU DRAM counters, idle-gap analysis, L6/L7/T6 |
| `opt-sm120-notes/opt/overnight/{pdl5,glue5,exl3r,exl3p,attn5,graph5,integ5}.md`, `integ5/SHIP.md` | per-candidate kernels, exactness gates, off switches, left-open items |
| `opt-sm120-notes/opt/overnight/research-ecosystem.md`, `research-upstream.md` | the night's upstream survey (vLLM, SGLang, FlashInfer, TRT-LLM, ExLlamaV3, fla, llama.cpp, b12x) |
| `.src/tf` at 9246a6743 (= f350cd3e + 66 patches) + `patches/engine/0067–0091` | the engine as built; verifications cited below are from this tree plus the patches |
| `.state/model-config.json` | model shape (verified, see §2) |
| Web (where cited) | checks the night's research flagged unverified, plus items newer than it |

Model shape, verified from `.state/model-config.json`: 45 layers in a **3:1 cadence — 34 KDA (linear-attention)
layers and 11 DSA (DeepSeek-sparse-attention) layers**; FFN dense in the first 3 layers, MoE (288 routed experts,
top-8, plus a shared expert) in the other 42; hidden 4096, MoE intermediate 2048, 64 heads; indexer `index_topk
2048 / index_kpool 4` → **top-512 of pooled keys**; vocab 154,880; one MTP head (`num_nextn_predict_layers 1`);
max positions 1,048,576. Production serves the 4 bpw EXL3 checkpoint with `DENSE=q8`, `HEAD=bf16`, fp8 paged KV
(one shared pool), TP=2 over PCIe Gen5 x8, 4 streams.

## 1. Where production stands (91 patches, 2026-10-31 11:48)

All numbers [M] from the night's combined slot (A S S A against the 66-patch production image, greedy hashes
identical, 33 of 33 metrics faster):

| Measure | Value |
| --- | ---: |
| Plain decode step (no drafter), short context | 7.05 ms |
| Prose / code decode (greedy) | 184.6 / 301.6 tok/s |
| File-edit decode (copy drafts) | 585.3 tok/s |
| 4 streams pooled / 4 streams on 8K contexts | 129.4 / 38.3 tok/s |
| Plain step at 200K / ~690K | 7.72 / 8.56 ms |
| Drafted round at 32K / 131K / 200K | 11.01 / 11.27 / 11.39 ms |
| Prefill 4K / 16K / 64K / 128K / 200K | 5,174 / 5,883 / 5,984 / 5,863 / 5,677 tok/s |
| KV pool at reserve 2.8 | 1,928 pages = 987,136 tokens; one request up to 985,088 |

What 0067–0091 added: many-CTA exact DSA select (`TF_GLM_DSEL`, select at 864K 158 → 19 µs/layer), in-kernel
sparse-attention merge (+2.7–2.8% prefill, 353 MiB freed), an audited PDL chain along the decode path (+3.1%
plain step), decode glue fusions (greedy one launch, two-problem GEMVs, fused router select+group+sum,
shared-window segment launches; 4 streams +5.5%), a new EXL3 prompt-expert loop (1.177×/1.161× on gate/up and
down) plus packed prompt GEMMs (1.12–1.13×), grouped EXL3 decode reading each trellis word once (−9 to −14% at
8–32 rows) run as a dependent-launch chain, KDA step warps by row count, hc_finish/hc_partial8 splits, and decode
index scores on the faster kernel for ≥4-row windows.

## 2. The three structural facts that decide the priorities

These come from the fresh profile and hold after the night's gains:

1. **Prefill speed is GPU0's speed.** At the 400 W cap, GPU0 (rank 1, the desktop card) runs 1,687–1,774 MHz SM
   against GPU1's 2,115–2,167 (−20%), at 80–82 °C. Rank 0 spends **46–49% of a long prefill waiting** for it; on
   rank 1, NCCL and the copy engines are 98–100% overlapped. Only gains on rank 1's path move prefill, and the
   largest lever is a clock/power decision, not a kernel (§7 D2).
2. **Decode is close to bandwidth-bound, and the remaining gap is specific and known.** Per rank per token the
   engine streams ~6.6 GB [C+E: int8 dense ~3.8 GB, routed experts ~2.1 GB, bf16 head 0.63 GB]. At the 7.05 ms
   step that is ~936 GB/s ≈ 55% of the P1 peak (1.71 TB/s at 13,365 MHz). NCU says the int8 GEMVs already run at
   **87% of that peak**; the grouped EXL3 kernel is within **8–12% of its own loads-only speed** at ≥8 rows
   (exl3r: loads-only reaches 1.44–1.59 TB/s). Both cards sit in P1 at 13,365 of 14,001 MHz memory under every
   CUDA load (−4.5%), a root setting (§7 D1). So the exact-work decode headroom is real but bounded — roughly
   0.3–0.8 ms of the 7.05 ms (§5) — and the big remaining decode levers are byte cuts (q4 dense, §6 B1) and the
   user's clock decisions (§7).
3. **Rank 1 (GPU0) is the critical rank for decode too** (skew 0.24 ms a plain step, 1.56 ms a forced 8-row
   round, 0.93 ms at 200K — rank 0 idles inside the P2P gathers). A decode speedup that lands only on rank 0's
   path is worth ~nothing.

## 3. What changed since docs/glm.md (v1) — settled items

v1's list, resolved by the night, so nobody re-litigates them:

| v1 item | Outcome |
| --- | --- |
| P0-1 reseed the draft rule / re-profile verify | **Done as T6.** Fresh fits: lone `10.02 + 1.29·(R−1)` ms, drafter 1.44 ms; shared `11.4–14.8 + 1.17–1.23·(R−1)` by stream count (GB10's shared table priced a row 1.5–2× too cheap). Replay verdict: **keep `cut`**; `tpm` on measured prices is +0.1% lone / +1.5% shared — below the ship rule. Closed. |
| P0-2 bounded CUDA graphs | **Tested and dropped for the drafter** (graph5 c3): the block pass runs 1,091 µs busy in 146 kernels with only 42–48 µs of gaps — graphs recover ≤0.02–0.05 ms. For the whole step, idle is 4.8% (0.39 ms) with launch lead p50 2.3 ms; PDL already recovered +3.1%. The remaining ≥50 µs gaps (0.19 ms/round) are host syncs around sampling/commit, not launch overhead. A full decode graph is no longer a top item; what remains of this idea is in §5 A3/A6. |
| P0-3 EXL3 decode re-tile | **Done as exl3r 0067/0068** (read-once trellis loads, −9 to −14% at 8–32 rows; dependent-launch chain). Remaining kernel headroom is the 8–12% decode+MMA gap (§5 A1). |
| P1-1 expert-parallel MoE | **Dead for TP=2**: EP moves the same bytes per rank (half of 8 sharded experts = 4 whole experts) and adds imbalance; the exchanges it would replace already cost 3.3–8.4 µs. v1's +15–30% was wrong. |
| P1-2 MMA/fp8 chunk attention | **Half wrong**: the chunk attention body is already tensor-core (`mma16816`, `glm_glue_attn.cuh:197,259`). What remains SIMT in attention is the indexer (scores/select) and small glue — addressed by 0067/0068/0091 at decode and 0068 at prefill. The still-open numerics piece is fp8 *math* on E4M3 keys (§6 B4), not MMA-izing. |
| P1-3 power caps | Stands, user decision (§7 D2). v1's uneven-TP-split alternative is dead (changes down-proj sum order → numerics; costs pool). |
| P1-4 refined W8A8 | **Still open**; broad W8A8 failed its NLL gate by 0.0006 nats (§6 B2). |
| P1-5 MTP chain depth | Still open, user decision with a context trade (~2 GiB for the MTP head ≈ a ~360K-token smaller pool; the overnight plan priced it as context ~516K) (§7 D5). |
| P2-1/P2-2 KV admission, CoW prefixes | Still open, capacity items (§7 D7/D8). |
| v1 "FlashInfer sm_120 status unverified" | **Now verified**: FlashInfer ships sm_120 backends (KDA/GDN docs; NVFP4 "cake" decode/prefill; the b12x DSA indexer lives in `flashinfer/experimental/b12x` as sm_120-native). Nothing there is drop-in for an EXL3 TensorFold engine, but the DSA indexer and FlashKDA are design references (§5 A4, §6 B3). |

Also measured-and-dropped since v1 (do not re-propose without new evidence): GPU drafter selector, drafter row
tiles, drafter graphs, drafter sum folds, sparse drafter head (R7: 79–89% coverage vs 99.3% break-even), staged
router (glue5 c7), TPM=128 expert items (0.918×), KDA step row overlap, fused rot_in+gate/up+epilogue (+6.6–9.2%
slower), L2 bulk prefetch (+14–24% slower), deeper register/cp.async rings, V4 staging, evict_first, register
caps, 2× unroll, one-line PDL triggers on kda_out/expand_v (+0.9% slower), L2 prefetch before a GEMV's wait, R3
twin copy-engines (0–1%), R5 L2 prefetch port, upstream #1571 sampler gather (not measurable), `TF_GLM_INDEX_CUDA`
at prefill (+1.0–1.5%, within 2× spread). Upstream techniques the ecosystem survey checked and rejected for this
setup: DCP for speed, tree verification on KDA (no exact kernel), spin-barrier top-k at batch 1 (≈ ours, with
sm_120 hang bugs), cooperative grid barriers under TP2, Machete (Hopper-only), DeepGEMM fp8/fp4 indexer logits
(SM90/100 only).

## 4. Where the time goes now (post-91-patch view)

The only full per-category breakdown is prof5's nsys at the 66-patch build (plain step 7.96 ms captured ≈ 7.6 ms
real; S5 is 7.05 ms, so scale the table by ~×0.93 and re-shape where 0067–0091 acted). Shares, not absolutes, are
what matter here; the last column says what the night already took out of each line.

| Category (rank 0 / rank 1 µs at 66 patches) | Share | State after the night |
| --- | --- | --- |
| attn projections (int8 GEMV), 2,246 / 2,323 | 28% | untouched except two-problem DSA pairs (0073). NCU: 87% of P1 peak — **bandwidth roof; only fewer bytes or fewer launches help** |
| routed experts (EXL3 grouped), 1,915 / 2,007 | 25% | 0081 read-once + 0082 chain: −9…−14% at 8–32 rows, ~−0.02 ms plain. Within 8–12% of loads-only |
| attn core (KDA step, MLA, indexer select+scores), 795 / 821 | 10% | select → many-CTA (0067); scores ≥4-row windows → fast kernel (0091); KDA warps by rows (0084) |
| mHC + norms, 869 / 927 | 11% | hc_finish/hc_partial8 splits (0086/0089), fused topk_group_sum (0074): c9 the only consistent helper, 1–3% |
| router/top-k/combine, 590 / 618 | 8% | 0074 fused select+group+sum (7.0 → 4.1 µs at 1 row) |
| comm (P2P gathers incl. waits), 671 / 412 | 6% | 0069–0071 PDL; the rank-0 excess **is the GPU0 skew**, not comm work |
| LM head (bf16 GEMV), 393 / 397 | 5% | at 1.55–1.6 TB/s — roof; q8 head is a closed user decision (+1–2% measured) |
| shared expert + dense FFN (side stream), 387 / 378 | 5% | mostly hidden under the main stream |
| KDA commit + sampler + other, 90 / 73 | 1% | — |
| GPU idle, 0.39 ms/step | 4.8% | PDL chain took +3.1%; the residual is host syncs around sampling/commit, not launch gaps |

Decode at long context adds on top (66-patch numbers; 0067/0091 already cut the two big ones): at 200K the select
was 0.50/0.56 ms and scores 0.13/0.18 ms a step; at ~690K the indexer was projected at ~2–2.5 ms (~20% of the
step), ~¾ of it the select. After T5(a): select 19–27 µs/layer (≈0.2–0.3 ms/step at 690K); one-row windows keep
`tf_glm_scores` at ~43 µs/layer at 864K (1.29 TB/s — its DRAM floor), ≈0.47 ms/step, and that kernel is now the
largest context-growing decode cost.

Prefill, rank 1 (GPU0, the critical rank) at 64K, 66-patch shares: prompt experts 38% (now ~28–30% after exl3p
1.177×/1.161×), packed prompt GEMMs 27% (now ~24–25% after R6 1.12×), attention 22.6% (now ~21% after the
in-kernel merge), comm 6–8% (fully overlapped), `kda_step` 8.0% — **the largest remaining SIMT kernel in
prefill** — plus ~5% of small SIMT glue (combine, scores, router_tile, hc). At 150–170K attention rises to ~27%.
The 4K-prompt gap (5,174 vs 5,984) is per-chunk fixed cost plus the measured fact that pairing a two-chunk prompt
is ~3% slower than not (0059, so the threshold is right, not a bug).

## 5. Decode opportunities, prioritized

House rules apply to every item: an exact patch ships only with identical greedy hashes in a same-slot ABBA A/B
plus a gain above twice the A-to-A spread; numerics changes are judged by NLL against the null spread
(`opt/num3/qh.py`, `opt/overnight/gate/`) and by answers, never by KL; opt-in flags stay off unless the user sets
them. And remember structural fact 3: only rank-1-path gains count.

### A1. Grouped EXL3 decode: close the 8–12% decode+MMA gap (exact)

**What.** The read-once re-tile (0081) put the grouped kernel within 8–12% of its loads-only speed at ≥8 rows.
The untried lever for that last part is occupancy: the kernels still stage warp partials in
`__shared__ float red[W][RR][NT*16]` (`exl3_grouped.cuh:114,205`, verified in-tree) and run 2 blocks/SM; folding
the split-K partials with warp shuffles in the same warp-order (w ascending) frees shared memory and registers
for a deeper load-ahead or more resident blocks. A second, minor sub-item: gate/up still runs split-K sk=4 by
default (`exl3_experts.zig:47`, verified), writing sk partials to global and folding them in the epilogue —
honest sizing says that traffic is ~0.7% of step bytes, so treat it as an epilogue simplification, not a
bandwidth win (this corrects deepseek.md's D2 estimate of 3–8%).
**Evidence.** exl3r: "loads only reaches 1.44–1.59 TB/s … the decode+MMA work is the remaining 8–12% at ≥8
rows"; register caps for 7–8 blocks/SM were worse, so the path is a registers+smem restructure, not caps.
**Expected.** 0.1–0.2 ms a plain step; 2–5% on drafted rounds (an 8-row round spends ~44–50% of its wall here,
and 0081's −14% at 32 rows translated to +13% file-edits, so the round-level leverage is proven). **Risk:** low —
bit-identical by construction (same 32-bit values, same warp-order sums); 0081's contract harness with planted
faults exists. **Validate:** decode4_bench cold tiles, then ABBA at rows 1/8/16/32.

### A2. Attention projections: fewer launches at the bandwidth roof (exact)

**What.** The int8 GEMV bucket is the step's largest (28%) and already at 87% of the P1 peak, so the only exact
gains left are launch-side: ~202 `glm_mm8_16` launches a step at ~13.8 µs each. glue5 c2 (0073) proved the
pattern — two problems in one launch, each keeping its own split-K and K order, byte-equal. Extend it to the
remaining SK-compatible siblings (the KDA-side in/f_b/g_b and o projections, shared-expert pairs), keeping every
output's accumulation order.
**Evidence.** profile5 §8: "SK-preserving projection batching (236 int8 GEMV launches a step): ≤0.1–0.2 ms";
c2's measured −3.8/−2.5 µs on its two pairs. Upstream: TRT-LLM's fused A-GEMM and vLLM Kimi-K3's
`low_latency_gemm` fuse Q/K/V/G the same way (exact when K order is kept).
**Expected.** 0.1–0.25 ms a plain step (1.5–3.5%), a little more on shared windows. **Risk:** low-medium (SK
mixing changes rounding — the contract is per-problem SK, as 0073 built). **Validate:** bitwise shape coverage
like 0073's, then ABBA.

### A3. PDL widening: the two chains pdl5 left untried (exact)

**What.** pdl5's own hand-in lists the remaining widening, ranked: (a) `router_tile` as `hc_finish`'s dependent
with the MoE fork recorded after it (~42 MoE layers × 1–2 µs; the shared expert's side stream starts ~7 µs later
inside its slack), and (b) the DSA prepare rmsnorms (`q_norm`, `kv_norm`) and `absorb_q` as dependents (11 layers
× ~3 boundaries each, overlapping 0073's two-problem projections). Both sit inside the audited PDL regime;
`tools/pdl_sites.py` (20 sites / 60 kernels today) gates them, and the 0090 lesson — every split kernel needs its
wait — is now a test case.
**Expected.** 0.05–0.15 ms a plain step combined; small, but it compounds with A2 on the same boundaries.
**Risk:** low with the audit; the kda_out/expand_v trigger regression (+0.9%) shows each site must be measured,
not assumed. **Validate:** bench_chain rows 1/8/32, pdl_sites PASS, ABBA.

### A4. Long-context decode: the indexer's remaining DRAM floor (exact)

**What.** After T5(a) and 0091, the context-growing costs are (i) one-row scores on `tf_glm_scores` — 43 µs/layer
at 864K, 1.29 TB/s, i.e. at its DRAM floor: ~0.47 ms/step at ~690K, growing linearly — and (ii) `dsel`'s fixed
cost (0.06–0.09 ms/step, attn5 c5, skipped as too small). The one structural fix left is **sharding the indexer
scores across the ranks**: each rank scores half the pooled keys (each score is an independent dot product, so
per-score arithmetic is unchanged), then an exact (score, id) merge with today's tie order produces the same
top-512 — the same merge proof as vLLM's DCP top-k. The night's ecosystem survey named this "the only way below
the one-row scores' DRAM floor" and it was **not attempted**. Optionally warm-start the select threshold from the
previous step's pools (TRT-LLM GVR, tech blog 21) to drop one refine pass.
**Evidence.** NCU: scores bandwidth-bound at 913–1,182 GB/s (32K–64K pools); ecosystem §4.0: 314 MB/rank/step of
score keys at 864K. Select is already fixed by 0067; 0091 covers ≥4-row windows.
**Expected.** Net −0.10 to −0.20 ms/step at ~690K (bytes halve, minus a small per-layer exchange), ~−0.05 ms at
200K; helps every long-context plain step, which is this recipe's headline use. **Risk:** medium (a per-layer
exchange on the DSA path; merge exactness under ties must be proven like 0067's was). **Validate:**
selbench/scorebench extension, then an A D D A at ~690K — the harness that validated c4.

### A5. Confirm-and-close: the night's own open items (measurement only)

- **attn5 c4 at ~690K**: the +6.2% on 16-row edits at ~690K is one pair. One A D D A confirms or kills it.
- **c9's `2:16` table**: the most consistent leave-one-out (plain −1.2%, code −1.7%, edit −1.3%, par4 −2.9%, all
  n.s. alone). One quiet A M M A resolves its 1–3%.
- **glue5 c6 and c8**: measured ~0 inside S5 — simplify away (less code) or keep; a dedicated A/B each way.
- **Drafter shared-round pricing**: the drafter is ~8–9% of a round and every kernel idea there measured nothing;
  the only item with any replay value left is the calibrated shared table (+1.5% shared rounds only,
  compile-time constant at `cuda_lanes.zig:24-25`) — ship only if a 2–4-stream ABBA clears the bar.

### A6. Safety-adjacent, no perf claim: P2P gather epoch-skew hardening

graph5's planted-fault study found a real pre-existing hazard: under epoch skew the **lagging** rank can meet a
later gather carrying the same epoch number and copy the wrong payload with no local error (the leading rank
fails and stops both ranks — the engine relies on that). The proposed hardening (wait for `flag == epoch`,
equivalent in correct runs; or a per-call tag) is exact, cheap, and turns a "depends on the peer failing"
invariant into a local check. Worth doing as robustness, not speed. Two invariants to keep wherever this area is
touched: re-run `tools/pdl_sites.py` after any change near the EXL3 plan-before-wait reads, and keep
rot_in/epilogue triggering after their waits.

## 6. Prefill opportunities, prioritized

Prefill is bound by GPU0 (§2), so every item below is a rank-1-path item; rank-0-side gains are invisible. Exact
kernel work left is small; the real levers are gated numerics and user decisions.

### B1. q4 dense — the largest gated decode lever, and the gate now exists

**What.** `DENSE=q4` halves the int8 dense bucket (~3.8 GB/rank/token → ~1.9 GB). Handoff's measured number:
**−1.06 ms a plain step (~15% decode)**, plus ~0.9 GiB a GPU of freed weights → a larger pool. v1 deferred it on
evaluation cost; the change since is that **the T7 gate harness now exists and ran 8,497 paired items** (math /
code / knowledge / long-context strata, NLL nulls, predeclared UCB rule), so the evaluation is a slot, not a
project.
**Risk.** Real: q4 symmetric group-64 codes on attention/KDA projections is a bigger perturbation than W8A8's
activation rounding; v1's "user decision" framing stands. **Validate:** the same predeclared gate (NLL against
the null spread + per-stratum accuracy arm, one-sided 95% UCB < 1 point); production stays q8 unless the user
flips it. Note the freed ~0.9 GiB also shifts the context fit — re-run it before quoting a new pool.

### B2. Refined W8A8 (GS32, MLP-only) — the largest gated prefill lever

**What.** Broad W8A8 (0064) measured **+10% prefill alone, +19% with ABSORB_MMA** and failed its NLL check by
0.0006 nats (shift +0.0074 against a largest-null +0.0068) — a marginal miss measured from the low end of the
spread. The refined variant already scoped in the handoff (group size 32, MLP/dense-only via `SKIP_K`, rows ≥ 32
guard so decode never takes it, `SKIP_K` into the rank fingerprint) shrinks the perturbed surface and plausibly
passes where the broad variant missed. Separately, the **T7 accuracy arm was never run** (~2 h slot;
`opt/overnight/gate5.md` has the commands) — running it settles the broad variant with accuracy data instead of
the NLL coin-flip.
**Expected.** +5.5–6% prefill over ABSORB_MMA. **Risk:** numerics, opt-in. **Validate:** freeze policy → pilot →
NLL → accuracy arm, exactly as gate5 specified.

### B3. Chunked KDA prefill (FlashKDA-style) — the largest remaining SIMT block

**What.** `tf_glm_kda_step` is 8.0% of rank 1's 64K prefill (+ prep/out ~1.5%) and is a SIMT serial recurrence
over the chunk's rows (verified: no MMA in `glm_kda.cu`). The standard fix is a chunked scan — intra-chunk work
as MMA batches with fp32 state carried between sub-chunks — which is what FlashKDA (MoonshotAI, MIT; vLLM's
default KDA prefill for GLM-5.3 on sm_12x; up to 3.27× over fla `chunk_kda`), FlashInfer's sm_120 CuTe KDA
prefill, and NVIDIA's new cuDNN DSA/GDN front-end kernels (Sept 2026 docs) all implement.
**No chunked form is bit-compatible with the serial recurrence** (the night's research §3.5) — this is a
reassociation, so it ships as an opt-in numerics flag judged by the NLL gate, like ABSORB_MMA was.
**Expected.** kda_step 8% → ~3–4%: **+3–5% prefill** [E]; no decode effect (decode's KDA step is separate and
already tuned by 0084). **Risk:** medium — a new prefill kernel family; state must stay fp32 and exact across
sub-chunk boundaries. **Validate:** kernel A/B vs the serial step on real shapes, then NLL, then opt-in ABBA
under flag.

### B4. fp8 math on the E4M3 attention path (SageAttention-style) — long-context prefill

**What.** Keys/values are already stored E4M3 on pages but converted to bf16 for the `mma16816` math. Running
QK^T (optionally PV) as `mma.m16n8k32.e4m3` with per-block scales (SageAttention 2 / 2++ design) halves the math
cost where prefill attention is largest — attention is ~21% of rank 1 at 64K and ~27% at 150–170K, and prefill
loses ~5% from 64K to 200K (5,984 → 5,677).
**Expected.** +2–4% prefill at 128–200K, ~1% at 16–64K [E — the widest error bar in this report]. **Risk:**
numerics (Q quantization + per-block rescaling); opt-in. **Validate:** NLL first — this one plausibly fails the
gate; the experiment is cheap because the storage format already exists.

### B5. Small exact prefill items (bundle)

- **SIMT tail**: `tf_glm_combine` 0.19 s, `tf_glm_scores` 0.11 s, `router_tile` 0.06 s of rank 1's 13.6 s at 64K.
  The scores piece has a measured MMA replacement (`TF_GLM_INDEX_CUDA=scores`, +1.0–1.5% — below the ship rule
  but positive and intended-exact; if it is ever shipped, add the mode to the rank fingerprint first — it is not
  there today). Combine/router_tile fusions are each ≤0.5%.
- **Expert item merging**: exl3p's remainder-merge idea (~+1% prefill) was sized but not built; TPM=128 itself
  measured 0.918× and stays dead.
- **4K–8K prompts**: the gap is per-chunk fixed cost; pairing is already tuned (two-chunk prompts measured ~3%
  slower paired). The only honest lever is fewer fixed per-chunk launches — a profile-guided micro-pass, ≤1%.

## 7. User decisions and capacity items (no engine code)

The largest single levers left, and none is a code change. Three need root/sudo; all are the user's call.

| # | Decision | Measured / expected | Notes |
| --- | --- | --- | --- |
| D1 | **Memory clock P1 → P0** | up to ~4.5% decode [E, root cause] | Both cards hold 13,365 of 14,001 MHz under every CUDA load; decode is bandwidth-bound. Root; adjacent to D3's toolkit question |
| D2 | **Asymmetric power caps 440 (GPU0) / 360 (GPU1)** | the largest prefill lever [M: GPU0 −20% SM clocks at 400 W]; at most ~4.7% [E] | `sudo nvidia-smi -i 0 -pl 440 && sudo nvidia-smi -i 1 -pl 360`, then persist in `nvidia-power-limit.service`; one A/B measures it. Per the power rule, the user runs it |
| D3 | **Reinstall the host CUDA toolkit** (`sudo apt install -y cuda-toolkit-13-3`) | unblocks host builds, cuobjdump/nsight, and the SASS/PDL gates outside docker | Lost to `apt autoremove` on 10-31; production unaffected |
| D4 | **`VISION=0` if image input is unused** | ~1.8 GiB back on rank 0 ≈ **+~315K pool tokens (~+32%)** at today's ~5.8 KiB/token — the full 1,048,576 window then fits at the current reserve with headroom left | One line in `scripts/local.sh`; the fit re-runs on restart. Keep it if images are used |
| D5 | **Full window / MTP chain / `--recall`** | full window: `CONTEXT=1048576` pinned (reserve ~2.7 as-is; trivial with D4); MTP chain past DFlash2's 7 drafts raises the prose/code depth ceiling but costs ~2 GiB ≈ ~360K pool tokens (context ~516K); `--recall`: 1.5–3× repeated prompts, writes token ids to disk | User calls; MTP is the only one that trades the headline feature |
| D6 | **Mixed-traffic policy** (`TENSORFOLD_PREFILL_SHARE`) | measured [M]: p=1 (today) → 128K TTFT 26.9 s, decoders 7.4 tok/s; p=0.84 → TTFT +15%, decoders 25.4 (3.4×); p=0.68 → TTFT +42%, decoders 45.8 (6.2×); the max pause stays ~430–520 ms either way (a chunk is atomic) | The knob exists and is output-exact; recommended if concurrent decoding during long prefills matters to the users of 8888 |
| D7 | **Grow-on-demand KV admission** | more long requests fit concurrently; no per-request speed change | Design exists (`opt/mem3.md` §4); pairs with D4/D8 |
| D8 | **Copy-on-write prefix sharing** | a shared system prompt stored once: capacity + faster multi-stream starts | Idea stage; the paged pool and snapshot machinery give it a home |
| D9 | **Housekeeping** | watchdog self-expires 2026-11-03 00:42 (`STOP_WATCHDOG` ends it earlier); 37 engine worktrees, ~14 GB night binaries, ~10 old images (~4.8 GB each) | Ask before deleting anything |

## 8. Drafting, scheduling and concurrency

- **Acceptance, not the engine, still separates prose (185) from code (302) from edits (585).** The drafter is
  ~8–9% of a round and every kernel/selector/graph idea there was measured to nothing (graph5 c1–c4, R7). The
  DFlash2 weights set the acceptance ceiling; the engine's window rule is calibrated (`cut`, T6-closed). The one
  depth lever is the MTP chain (D5) — a user decision with a real context cost.
- **Cross-request suffix drafting (SuffixDecoding-style) is the one untried drafting idea that fits this
  deployment.** Production runs 4 streams on one server; agentic traffic (file edits, repeated system prompts,
  follow-up turns) plausibly shares long suffixes across requests, and copy-style drafting over another request's
  accepted text is exact for the target's outputs (drafts change; verification does not — the same class as
  0039's 16-row copy windows). Upstream reports up to large gains on agentic multi-request traffic. **But** the
  night's R7 closure is the cautionary tale: measure the overlap first. The `usage_sampler.sh` log and the
  prompt-cache hit lines can estimate how much production traffic actually shares suffixes before any engine
  work is scoped. Workload-dependent: 0% on chat, potentially large on edit-heavy multi-session days.
- **Shared-window decode at concurrency is in good shape**: segment launches (0075) and batched writes (0088)
  took 4-stream pooled +14.9% total. What remains in a shared round's ~6 ms base over a lone round is the
  drafter pass covering every stream (1.4–1.6 ms) and per-row sampling draws (0.1–0.23 ms) — both priced and
  small. Bigger windows are capped by acceptance, not capacity (T6: oracle +16–23% is the theoretical room, and
  it is drafter-limited).
- **Scheduling between prefill and decode is the D6 knob**, not a kernel: decoders get one round per 2048-row
  chunk today; k rounds per chunk is exact and trades TTFT linearly.

## 9. What would not pay (checked so it stays checked)

Carried from the night's measurements — each has a number against it now:

- **CUDA graphs for the decode chain** — idle is 0.39 ms/step with launch lead p50 2.3 ms; the drafter graph
  recovered ≤0.02–0.05 ms; the big gaps are host syncs around sampling/commit. PDL is the winning form of this
  idea here, and it is in.
- **L2 prefetch / evict-first / persisting-L2 tricks** — in-kernel bulk prefetch +14–24% slower; pre-GEMV-wait
  prefetch worse everywhere; the KDA state (68 MB fp32/rank) only fits the persisting window with one active
  stream (≤~1.2%).
- **Expert-parallel MoE at TP=2** — byte-neutral, adds imbalance (§3).
- **Drafter-side anything** — selector on GPU, row tiles, graphs, sum folds, sparse head: all measured ≤ spread
  or negative; the pass is GPU-bound at 42–48 µs of gaps.
- **TPM=128 / 4,096-row prompt passes** — 0.918× and +1.9 GiB/rank against the user's context priority.
- **Uneven TP split or expert rebalancing by clock** — changes the down-projection's sum order (numerics) and
  costs pool; the power split (D2) buys the same skew back without either.
- **q8 LM head** — +1–2% measured; user decision stands. **q6 dense** — no gain measured.
- **FlashInfer/vLLM kernel drop-ins** — nothing matches an EXL3 mcg trellis engine; the value of upstream is
  design references (b12x/FlashInfer DSA indexer, SGLang TopK-V2, TRT-LLM indexerTopK split-work, vLLM
  `persistent_topk`), most of which the 0067/0091 select/scores work already mirrors or beats at these shapes.
- **W8A8 broad** — failed its predeclared NLL gate; only the refined variant (B2) or the accuracy arm can reopen
  it.

## 10. How to validate each item under the house rules

1. **One slot at a time** (`slot_begin.sh` / `slot_end.sh`; `prof_up.sh` with the production image for measuring
   containers). Restore + `health.sh` (`OK 391 | Canberra`) after every leg. No builds or host tests during a
   timing slot (the load-81 incident cost a leg 8–23%).
2. **Exact patches**: same-slot ABBA (or a shared bracket `A B B A C C A`), greedy hashes identical on all T=0
   prompts, gain > 2× the A-to-A spread. Expected greedy shas: answer `69d29804284e`, prose_t0 `17318206e1f3`,
   code_t0 `9d917d073c72`.
3. **Numerics changes**: `opt/num3/qh.py` NLL against the null spread first, then the gate harness's accuracy
   arm; never KL, never hashes. Opt-in flags stay off unless the user sets them.
4. **Any PDL-adjacent change**: `opt/overnight/pdl5/tools/pdl_sites.py` must PASS (20 sites / 60 kernels today);
   the six split kernels' waits are enforced by test.
5. **Anything that moves weight bytes between ranks or shrinks scratch** (q4 dense, B3's buffers, A4's exchange
   scratch): re-run the context fit and quote the pool line from rank0.log against today's 1,928 pages.

## 11. Order of attack

Not additive; decode items assume rank-1-path. "Effort" excludes the mandatory A/B.

| # | Item | Decode | Prefill | Effort | Gate |
| --- | --- | ---: | ---: | --- | --- |
| 1 | D2+D1 power split & memory clock (user, ~15 min + A/B) | +0–4.5% | +0–4.7% | command | user + sudo |
| 2 | D6 mixed-traffic policy (knob exists) | — | TTFT ↔ 3.4–6.2× decoders | none | user |
| 3 | A5 confirm-and-close (c4 @690K, c9 table, c6/c8) | 0–6% long-ctx, 1–3% | — | hours | none (measurement) |
| 4 | A2 sibling-GEMV fusion round 2 | +1.5–3.5% | — | days | exact ABBA |
| 5 | A1 grouped-EXL3 occupancy restructure | +1.5–3% plain, +2–5% drafted | — | days | exact ABBA |
| 6 | A3 PDL widening (MoE fork, DSA prepare) | +0.7–2% | — | days | pdl_sites + ABBA |
| 7 | A4 indexer-scores sharding (+c5, GVR) | +1.5–2.5% at ~690K | — | 1–2 wks | exact merge proof + A D D A |
| 8 | B1 q4 dense through the T7 gate | +~15% | +pool | slot + eval | NLL + accuracy arm, user |
| 9 | B2 W8A8 GS32 MLP-only (+ the broad arm) | — | +5.5–6% | slot + eval | NLL + accuracy arm, user |
| 10 | B3 chunked KDA prefill (FlashKDA-style) | — | +3–5% | 1–2 wks | NLL, opt-in flag |
| 11 | D4/D7/D8 vision-off, admission, CoW | concurrency | capacity | config–wks | user; refit |
| 12 | §8 suffix-cache drafting (measure overlap first) | workload-dependent | — | wks | target-outputs exact |

Realistic compound potential if 1–7 land: **plain step ~6.4–6.7 ms (prose ~200–215 tok/s), prefill ~6,200–6,400
tok/s at 16–64K, and ~690K plain step ~8.1–8.3 ms** — before any gated item. B1 alone (if it passed its gate)
would move decode another ~15%; B2/B3 another ~9–11% prefill. The 4.5% memory-clock decision (D1) multiplies
everything decode-side.

## 12. Findings worth knowing (not action items)

- **GPU0 drifts warm under long slots** (78 → 86 °C in slot 3; prefill −4%) and its desktop memory moves by
  ~1 GB over hours, which moves the fit. Every verdict must compare adjacent legs; the daytime/night split in
  the handoff's tables partly reflects this.
- **The p2p gather epoch is host-incremented** (`p2p_gather.zig:55,124–128`, verified) — fine eagerly, a
  capture blocker for graphs, and the skew hazard of A6.
- **`TF_GLM_INDEX_CUDA` is not in the rank fingerprint** (`cuda_index.zig`); if the scores mode is ever enabled,
  fingerprint it first or the ranks can disagree silently.
- **The shipped image's c9 default** (`TF_GLM_HC_PART=2:16`, patch 0089) is built in; the measured slot set it by
  env. The shipped binary's kernels are SASS-identical to the measured ones (integ5 verified all 1,017), so the
  A5 c9 A/B is the only unresolved piece of that stack.
- **The night's report and per-agent files in `opt-sm120-notes/opt/overnight/` are the durable record** — `/tmp`
  is wiped on reboot; the recipe's `docs/` is untracked in git (commit 3a39a84 does not include it).

*No code, configuration, or production state was changed in producing this report. Files read outside this
repository (the `opt-sm120-notes` notes) were opened read-only via shell, as the handoff directs.*
