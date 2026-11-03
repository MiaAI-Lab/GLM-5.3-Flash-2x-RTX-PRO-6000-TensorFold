# Port: campaign wins for the 2x RTX PRO 6000 sister

Everything here is proven on the GB10 pair (see `docs/speed/RESULTS.md`, commits 21d4e36..0c4b811 on master).
The sister (`rtx:/home/zurih/MiaAI_Lab/TensorFold/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold-Zig`, 407dedd,
93 patches) takes three of them. **Numbers do not cross silicon**: every perf claim re-measures there, against
that box's own baseline, under the same discipline that produced them here.

## 1. The 4,096-row prompt chunk - 0094 + 0095 (the real perf port)

`0094-*.patch`, `0095-*.patch`, verbatim. Here: +7.8/+6.5% prefill at 23.5K/95K, byte-identical replies,
TTFT 13.6/54.1 s. Mechanism is silicon-independent (EXL3 trellis decode amortized over chunk rows).
Apply, rebuild, then: prefill A/B against the sister's current numbers, 3 cold trials, fresh content,
`cached_tokens=0`, and the 8-prompt token_sha corpus before/after. The DFlash2 `MAX_CTX` widening rides in 0095.

## 2. `HEAD=q8` - env-only, no rebuild

Set `HEAD=q8` (their config already plumbs it; default bf16). Here: +3.7/+2.2/+2.0% at 1/2/4 streams, greedy
answers byte-identical, paired per-token NLL -0.00050 (CI -0.00156..+0.00056). On the RTX pair's higher memory
bandwidth the head is a smaller slice - expect LESS, maybe much less; the A/B decides, not this paragraph.
Quality case re-established there from scratch: contract corpus + `nllpaired.py`/`cmp.py` before/after, per
passage, never aggregate-only.

## 3. Ops: the stop ladder + verified restart (the wedge insurance)

`0001-ops-stop-ladder.diff` (start.sh/stop.sh: stop -> kill -> verified absent -> abort-for-human; restart
aborts on a failed stop instead of layering a second ~50 GiB pair onto a wedged one) and
`0002-safestart-via-stop.diff` (their docs/safestart.sh, if present, routes through stop.sh so logs archive
and the ladder runs - here the rm -f shortcut silently disabled both for a day). The GB10 wedge post-mortem
(RESULTS, 2026-11-03 05:07Z) is the why; the mechanism is not GB10-specific.

## 4. Measurement harness, whole files, no diffs

`dec.py` (fixed-prompt sweep, None-safe summaries, token_sha contract mode), `leg.sh` (one measured leg),
`aba.py` (A/B/A drift-vs-effect), `nllpaired.py` + `cmp.py` (paired per-token NLL). The house rules they encode
are the transferable asset: fixed prompts, 3+ trials, interleaved A/B/A below ~2%, contract as precondition,
never measure against a pair that is also serving an advisor, verified stops only.

## Not portable (do not carry)

0096/0097 (GB10 cost-table corrections - the sm_120 tables they came FROM are this pair's calibration);
the unified-memory reserve/pool/fill regime and the bimodality/decay phenomenon (GDDR7 has no host-shared
map; the METHOD travels, the physics doesn't); the nsys/CUPTI platform gap (their driver may differ - if nsys
works there, a kda ncu pass on the sister is informative for the engine, not decisive for GB10).

## Order of operations on rtx

1. Bundle/branch the sister's current state first (same standard: restorable before touching).
2. 0094+0095, rebuild, contract, prefill A/B.
3. Ops diffs, exercised once each (a real stop/start cycle).
4. `HEAD=q8` A/B/A with the contract precondition; keep only on their numbers + their quality case.
