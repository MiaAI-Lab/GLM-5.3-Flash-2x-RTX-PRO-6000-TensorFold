Run the agreed unified speed plan for our GLM-5.3-Flash server, unattended, start to finish. I'm away; don't wait
for me. Work autonomously within the rules below, and stop only when the plan is resolved or a stop condition hits.

## Read first (in this order)
1. `/home/zurih/MiaAI_Lab/TensorFold/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold-Zig/docs/handoff.md`: production
   state, how to operate, house rules, pitfalls.
2. `.../docs/unified-plan.md`: **the plan you are executing.** Codex, GLM and Claude agreed it; then I removed q4
   dense and made C1 opt-in. Its item IDs (0.1, 1.3a, 2.3, C1, ...) are the work units.
3. `.../docs/OPUS-v2.md`: the evidence and sizing behind each item.
4. Supporting detail: `/home/zurih/MiaAI_Lab/TensorFold/opt-sm120-notes/opt/opus2/` (`decode.md`, `prefill.md`,
   `spec.md`, `upstream.md`, `review/`), and the night's record `opt-sm120-notes/opt/overnight/` (`REPORT.md`,
   `integ5/SHIP.md`, the per-agent `*.md`). Don't redo what those record as tried and dropped.

## Verify the starting state before anything else
Check each of these:
- Production `glm53-zig` runs image `tensorfold-glm53-zig:987b580ba9d2`, with 91 patches in the recipe's
  `patches/engine`.
- `opt-sm120-notes/opt/overnight/health.sh` prints `OK 391 | Canberra`.
- `watchdog2.sh` is running (pid in `watchdog.pid`), and no `SLOT_ACTIVE` file exists.

If anything differs, record it and adapt; don't assume.

## Hard constraints (never violate)
- **Context and streams:**
  - One session must be able to reach ~985K tokens in the shared KV pool, and 4 streams must stay available.
  - Never trade context or streams for speed.
  - Re-check the context fit (`.state/context-fit.tsv`, the pool line in `logs/current/rank0.log`) whenever a change
    touches memory.
- **The machine:**
  - Monitors stay on GPU0.
  - Never change GPU power limits or clocks: no `nvidia-smi -pl`, `-lmc`, `-lgc` or `-cc`. Plan items 0.3 (power
    split, `-cc 1`) are mine to run; skip them and list them for me.
- **Numerics:**
  - Exact items only change speed.
  - Anything that changes numerics is an opt-in flag that stays **off**.
  - q4 dense is removed: don't touch it.
  - Dense stays q8, the LM head bf16, W8A8 off, ABSORB_MMA=1.
- **Shipping rule for exact patches:**
  - Identical greedy hashes in a same-slot ABBA against production, with a gain above 2× the adjacent A-to-A spread.
    Expected shas: answer `69d29804284e`, prose_t0 `17318206e1f3`, code_t0 `9d917d073c72`.
  - Plus the item's own byte or contract check (ties, NaN, ±0, subnormals).
  - Plus `pdl_sites.py` PASS after any PDL-adjacent change. Run it with `cuobjdump` from the cached
    `nvidia/cuda:13.3.1-devel-ubuntu24.04` image, as integ5 did.
- **Capacity-track items** (C1-C5): identical hashes, no slowdown beyond the A-to-A spread, and the pool line showing
  the gain.
  - C1 (embedding table to host RAM) is built behind `TF_GLM_EMBED_HOST=1` and a recipe knob `EMBED_HOST=0`,
    **default off**.
- **Numerics-gated items** (Phase 4): NLL against the null spread (`opt/num3/qh.py`, `opt/overnight/gate/`), then the
  accuracy arm. Even if one passes, leave it off and report.
- **Every new patch:**
  - numbered from `0092`;
  - applies cleanly on top of 0001-0091;
  - keeps an exact off switch (a `TF_GLM_*` env var), like the night's patches.
- **Git:** no commits or pushes.
- **spark1:** read-only, except `~/codex-glm53-sm120`.
- **Secrets:** never print them; pass `TENSORFOLD_API_KEY` by name only.
- **Code:** the code ban on r0b0tlab and vcruz305 is waived for this work. Don't touch the old 4x recipe
  (`GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold`).
- **No host CUDA toolkit, and don't install one:**
  - build images with the recipe's docker build (`./start.sh` rebuilds when the patches change);
  - compile and inspect kernels inside the cached devel image.

## Operating rules (from the handoff and the night's incidents)
- **GPU time goes through slots only:** `opt/overnight/slot_begin.sh HOLDER MINUTES "WHAT"` (`TIMING=1` for timing
  legs), `slot_extend.sh`, `slot_handover.sh`, `slot_end.sh`. `slot_end.sh` restores production and must print the
  health OK.
  - One slot holder at a time.
  - No builds or host tests during a timing slot.
  - Keep each slot ≤ ~150 min, and leave production up between slots.
  - Prefer quiet periods: `opt/overnight/usage.csv` shows whether anyone is using port 8888.
- **Measuring containers:** `opt/overnight/prof_up.sh` takes `IMAGE=tensorfold-glm53-zig:987b580ba9d2` explicitly;
  its default is stale.
- **A/B harness:** `opt/overnight/prof5/ab/run_ab5.sh` + `table5.py`; requests via `prof5/bench5.py`. Compare against
  adjacent A legs.
  - GPU0 warms over long slots, and its desktop moves memory.
  - Fix CTX across legs.
  - Report 4-stream throughput as total tokens / makespan, not a per-request median.
- **nsys:** costs ~0.7 GiB and inflates timings 4-9%. Preflight memory; use profiled legs for shares and unprofiled
  legs for numbers.
- **The watchdog** (`watchdog2.sh`) restores production outside slots and expires 2026-11-03 00:42. If you will run
  past that, start a new instance with a later deadline (read its header first). Touch
  `opt/overnight/STOP_WATCHDOG` only if you must stop it.
- **Self-matching patterns:** `pgrep -f`/`pkill -f` match their own command line; use bracket patterns.

## Team and advisors
- **Subagents:** use them freely for parallel code work, each in its own worktree
  `/home/zurih/MiaAI_Lab/TensorFold/tensorfold-wt-p1-<name>` off the 91-patch tree. Keep the team ≤ 8 and only one
  of them holding a GPU slot at a time; you, the main session, grant slots.
  - Give each agent the relevant plan IDs, the rules above, and its report file.
- **Codex** (spark1, `codex exec`; recipe in the handoff's "How to operate", bundles under `~/codex-glm53-sm120/`):
  use it for adversarial review of the risky designs before you build them: 2.3 (exact distributed top-k under ties),
  2.5 (twin-chunk aliasing), C4/M6 (scratch aliasing), 1.9/O11 (control plane), 1.6 (eviction contract).
- **GLM** (`http://rtx:8888/v1`, `reasoning_effort: max`, `thinking: true`; helper
  `opt-sm120-notes/opt/opus2/review/ask_glm.py`): it is production itself, so only call it outside slots and never
  during a timing leg.

## Order of work (follow the plan's phases; adapt the order to what Phase 0 shows)
1. **Phase 0.**
   - 0.1: a fresh nsys of the 91-patch build at the production config, plus unprofiled controls and paired-chunk
     route histograms. Re-size every later item from it.
   - 0.2: zero-code A/Bs (`TF_GLM_OVERLAP_PIECES` 1/2/3, attn5 c4 at ~690K, c9 `TF_GLM_HC_PART`).
   - 0.4: the length-matched agentic benchmark.
   - 0.5: the selector trace patch, then the τ replay.
   - 0.6: the T7 accuracy arm, only if a ~2 h quiet slot fits.
2. **Phase 1 probes**, in the plan's order: 1.3a/b, 1.7, 2.9, 1.4a/b/c, 1.5 (site by site), 1.9 (O11 restricted to
   plain greedy steps with the full control-plane gate).
3. **Phase 2:** 2.7, 1.6, 2.6. 2.10 is my policy choice: report the menu, don't change it.
4. **Phase 3, chosen by the profile:** 2.1 (NCU first), 2.2, 2.3 (start its design in parallel with Phase 1), 2.4,
   2.5.
5. **Capacity track:** C1 behind its off-by-default flag; then C2, C3, C5. C4 only after C2/C3, 2.2 and 2.5 settle.
6. **Phase 4:** only if time remains, and only as off flags with the NLL gate: 3.2, 3.3, 3.4.
7. **Integration:** stack every passing item into one candidate. Run one combined slot against production (A S S A,
   the night's 33-metric style plus the agentic benchmark and long-context legs), and leave-one-out where items
   interact.
   - Build the candidate image `tensorfold-glm53-zig:p1-<hash>`.
   - Write `SHIP.md`: patch list, sha256s, switch table, measured table, rollback.

## Ship policy
`PREPARE ONLY` (default): build and fully validate the candidate image, but **leave production on
`987b580ba9d2`**. I'll ship it myself from `SHIP.md`.

(To allow switching, I'd have written instead: `AUTO-SHIP`: if the integrated candidate passes every gate, copy its
patches into the recipe's `patches/engine`, `./stop.sh && ./start.sh`, and confirm health OK and the pool line; roll
back immediately on any failure. That is not granted here.)

## Record keeping (`/tmp` is wiped on reboot)
- **Workspace:** `/home/zurih/MiaAI_Lab/TensorFold/opt-sm120-notes/opt/plan1/`.
  - `STATE.md`: a timestamped running log, enough for a fresh session to resume.
  - `calendar.md` and `slots.log`.
  - Per-item `<ID>.md` with [M]/[C]/[E] labels.
  - Patches under `patches/`.
- **The end:** `REPORT.md`, with:
  - a verdict per plan item: shipped-to-candidate, dropped with numbers, or deferred with the reason;
  - the measured before/after table: plain step, prose/code/edit tok/s, 4-stream aggregate, the agentic benchmark,
    follow-up TTFT, prefill 4K-200K, long-context steps, the pool line;
  - the list of actions waiting for me.
- **The plan doc:** also append a short "Results" section to `docs/unified-plan.md`.

## Stop conditions
- Phases 0-3 and C1 are resolved (built and measured, or dropped with numbers). Phase 4 is optional.
- Or production can't be restored to healthy after a slot. Then stop all GPU work, leave production stopped only if
  restarting makes it worse, and write the failure and the exact state at the top of `STATE.md` and `REPORT.md`.
- Or a hard constraint would have to be broken to continue. Then skip that item and say why.

Before ending, make sure no slot is active, production is healthy on its original image, no measuring containers
remain, and `REPORT.md` is complete.
