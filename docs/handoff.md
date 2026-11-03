# GLM-5.3-Flash on 2× RTX PRO 6000: handoff

> **Superseded on 2026-11-01 by `HANDOFF.md` in the recipe root. This file is kept as history and is out of date in many places.**

> **Update 2026-11-01 (this replaces anything below that disagrees):** `SERVED_NAME` defaults to `GLM-5.3-Flash` (commit 407dedd), at the user's request, with no restart asked for. The running container keeps `glm-5.3-flash` until the next `./start.sh restart`; until then a plain `./start.sh` exits with E_API (/v1/models does not list the name) and leaves the server alone. Requests naming either ID are answered. No rebuild or refit at the next start.
>
> **Update 2026-11-01 02:20 (this replaces anything below that disagrees):** production and the default are image `b2364db4a201` (93 patches, commit 6a392c6): the 91-patch stack plus 0092 (a JSON answer stays JSON beside tools) and 0093 (a grammar freed when its stream cannot start). All 1,017 kernels are identical to `987b580ba9d2`, which is still on disk for a rollback: remove the two patches and `./start.sh restart`. Records: `opt/plan1/agents/main/port/` (`verify.py`, `verify-before.json`, `verify-after.json`, `verify-final.json`). GPUs run at 300/300 W since the user's own change at 2026-11-01 01:20 (persisted in `nvidia-power-limit.service`). Window 960,512; pool 1,939-1,950 pages, varying with what the desktop holds on GPU 1; no device prompt-cache share now, so the cache is 8.0 GiB a rank.

> **Update 2026-10-31 23:30 (this replaces anything below that disagrees):** production and the default are image `987b580ba9d2` (91 patches, commit a03d108), 400/400 W. The user moved the monitors to **GPU1** (`03:00.0`), so `GPUS=0,1` in `scripts/local.sh` keeps the heavier rank 0 off the desktop card; `GPUS=1,0` on this layout fits only 836,608 tokens. Live window 960,512 tokens, 1,880 pages, 4 streams. GPU0 is still the slower, hotter card without the monitors. A2 (`TF_GLM_INDEX_DEEP`) was tried and discarded by the user. Watchdog and usage sampler have not run since the ~22:46 reboot. Record: `opt-sm120-notes/opt/plan1/REPORT.md`.

As of 2026-10-31 12:25. Production is live on port 8888 with 91 engine patches, committed at a03d108. The overnight
agents have all finished; nothing is half-done. Two helper loops still run on their own (the watchdog and a usage
sampler, below). Everything under "Open work" is optional and needs the user's go-ahead.

## Where things are

Everything runs locally on host `rtx` (the 2× RTX PRO 6000 workstation). The only remote machine is spark1, and it hosts only the two advisors.

| What | Location |
| --- | --- |
| Live recipe (git, one commit 6a392c6 on `main`, force-pushed to the private repo `MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold` on 2026-11-01; `docs/` untracked) | `/home/zurih/MiaAI_Lab/TensorFold/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold-Zig` |
| Local settings (gitignored) | `scripts/local.sh` in the recipe |
| Context-fit cache | `.state/context-fit.tsv` in the recipe |
| Engine patches | `patches/engine/0001`–`0091` in the recipe |
| Plan, rules, reports (waves 1-3) | `/home/zurih/MiaAI_Lab/TensorFold/opt-sm120-notes/` (`PLAN.md`, `WAVE2-RULES.md`, `opt/*.md`) |
| The 2026-10-31 night | `opt-sm120-notes/opt/overnight/`: `REPORT.md` (results), `STATE.md` (main's log), `calendar.md`, `slots.log`, per-agent `<agent>.md`, `integ5/SHIP.md` (patch list, switches, rollback) |
| Night plan and its inputs | `docs/overnight-plan.md`, `docs/glm.md`, `docs/deepseek.md` in the recipe |
| Patch series by wave | `opt-sm120-notes/opt/wave2-patches/integ/` (0026–0063), `integ2/` (0064–0066), `opt/overnight/patches/integ5/` (0067–0091; the measured variant in `measured-s5/`) |
| Engine worktrees (37) | `/home/zurih/MiaAI_Lab/TensorFold/tensorfold-wt-*`; the night's are `tensorfold-wt-n1-*` (uncommitted) |
| Night binaries (~14 GB) | `/home/zurih/MiaAI_Lab/TensorFold/toolchain/install/tensorfold-opt-n1-*`; the final engine is `-integ5-s5e` |
| Model (4 bpw) | `~/.cache/aevonix-glm53/model` |
| 3.5 bpw GLMQ quant (reference only) | `/home/zurih/glmq/out/GLM-5.3-Flash-EXL3-3.5bpw-GLMQ` |
| DFlash2 drafter | `~/.cache/aevonix-glm53/dflash2` |
| Running container | `glm53-zig`, image `tensorfold-glm53-zig:987b580ba9d2`, port 8888 |
| Codex advisor | spark1 `~/codex-glm53-sm120/roundN/` and `lanes1/` |
| DeepSeek advisor | `http://spark1:8888/v1`, model `DeepSeek-V4.1-Flash-1017` |
| Old 4× recipe (not in use) | `/home/zurih/MiaAI_Lab/TensorFold/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold` (has uncommitted edits) |

`/tmp` is tmpfs and is wiped on reboot, so anything worth keeping goes in `opt-sm120-notes`.

## Production state

The server runs TensorFold f350cd3e plus 91 patches, built for sm_120 with Zig 0.17.0 (image label `987b580ba9d2`).
Four streams share one fp8 page pool of 1,928 pages (987,136 tokens), and a single request can use up to 985,088
tokens.

| Setting | Value | Set in |
| --- | --- | --- |
| Image | `tensorfold-glm53-zig:987b580ba9d2` | built by `start.sh` from `patches/engine` |
| Model | 4 bpw EXL3 | `local.sh` `MODEL_DIR` |
| Drafter | DFlash2 plus copy drafts | `config.sh` |
| GPU order | `GPUS=1,0` (rank 0 on GPU1, away from the desktop) | `local.sh` |
| KV cache | fp8 (E4M3), paged (`KV_PAGES=on`), 512-row pages | `config.sh` |
| Streams | `PARALLEL=4`, one shared pool | `local.sh` |
| Context | `CONTEXT=auto`; fitted 985,088 at reserve 2.8 | `local.sh`, `.state/context-fit.tsv` |
| Memory reserve | `MEMORY_RESERVE_GIB=2.8` (the user's choice for a ~1M window; was 3.5, then 3) | `local.sh` |
| Rank exchanges | `CE_EXCHANGE=1`, plus the P2P one-shot all-gather | `config.sh` |
| `ABSORB_MMA` | 1 (the user turned it on) | `local.sh` |
| `W8A8` | 0 (failed its quality gate on 2026-10-31) | `config.sh` |
| `TENSORFOLD_PREFILL_SHARE` | unset (1) | — |
| LM head / dense | bf16 / q8 | `config.sh` / `local.sh` |
| Port | 8888 | `local.sh` |

Each rank also gets an 8 GiB host-RAM prompt tier (pinned memory). Follow-up turns take 0.2–0.6 s.

Reserve versus window on 2026-10-31 (desktop memory varies, so treat these as approximate):

| `MEMORY_RESERVE_GIB` | Pool | One request up to |
| --- | --- | --- |
| 3.5 | 1,856 pages (950,272 tokens) | 928,768 |
| 3 | 2,009 pages (1,028,608 tokens) | 1,012,736 |
| 2.8 (now) | 1,928 pages (987,136 tokens) | 985,088 |

At 2.8 the auto-fit landed lower than at 3: its first try at 1,048,576 was refused by a hair, and the retry asks for
93% (`CONTEXT_FIT_MARGIN`) of what fits. A full 1,048,576 window needs `CONTEXT=1048576` pinned with a reserve of about
2.7. Each extra stream costs ~0.39 GiB a GPU of fixed state (4 → ~1.06M pool, 8 → ~786K, 16 → ~261K at 7.4 GiB).

Hardware:

- Both GPUs run at PCIe Gen5 x8. This needs M.2_2 left empty, because it shares lanes with the second x16 slot.
- Both GPUs are capped at 400 W, made persistent by `nvidia-power-limit.service`.
- GPU0 drives the monitors. At 400 W it runs ~20% lower SM clocks than GPU1 under long prefills (80-82 °C) and is the
  critical rank.
- Both cards sit in memory P-state P1 (13,365 of 14,001 MHz) under every CUDA load: ~4.5% of decode bandwidth, a root
  setting.
- **The host CUDA toolkit is gone** (since 2026-10-31 10:55, see pitfalls). Production doesn't need it.

| Measure | Morning 10-30 (x4, 300 W, stock) | Stock at x8, 400 W | 66 patches | 91 patches (now) |
| --- | --- | --- | --- | --- |
| Prose decode | 130 tok/s | 144 tok/s | ~170 tok/s | ~185 tok/s |
| Code decode | 202 tok/s | 231 tok/s | ~277 tok/s | ~302 tok/s |
| File-edit decode | — | — | ~518 tok/s | ~585 tok/s |
| 4 streams pooled | — | — | ~113 tok/s | ~129 tok/s |
| Prefill, 16K–64K prompt | ~2,700 tok/s | ~3,900–3,970 tok/s | ~5,260–5,330 tok/s | ~5,880–5,980 tok/s |
| Plain decode step (no drafter) | ~14 ms | 9.37 ms | 7.61 ms | 7.05 ms |
| Plain step at a 200K context | — | — | 8.62 ms | 7.72 ms |

The 66- and 91-patch columns come from one same-slot A S S A on 2026-10-31 (33 of 33 metrics faster, greedy hashes
identical).

## Patch stack

Patches 0001–0091 live in `patches/engine`, and `start.sh` rebuilds the image when they change. Exact patches passed identical greedy hashes in same-slot A/Bs. 0041 and 0064 are opt-in numerics flags.

| Patches | What | Measured effect |
| --- | --- | --- |
| 0001–0004 | Mixed-bit EXL3 expert support | Needed to load the quants |
| 0005–0012 | Exact decode and prefill speed-ups | Prefill +15–17%, DFlash2 decode +6–7% |
| 0013 | Upstream #1623 (stale DFlash2 taps) | Higher acceptance, no prefill cost |
| 0014–0016 | Frees the MTP head when DFlash2 drafts, adds the host-RAM prompt tier, fixes the GPU-check build | Memory freed; fast follow-up turns |
| 0017–0025 | Paged fp8 KV with a shared pool | 4 streams share one ~900K pool |
| 0026–0033 | decode4: MoE glue, per-width EXL3 decode kernels | Part of the wave-3 gain below |
| 0034–0035 | P2P one-shot all-gather (CUDA IPC), copy-engine prompt exchange | Part of the wave-3 gain |
| 0036–0040 | DFlash2 copy-skip, W16 | Part of the wave-3 gain |
| 0041 | `TF_GLM_ABSORB_MMA` opt-in: absorb_q/expand_v on tensor cores | Prefill +6.6%; on |
| 0042–0048 | mem3 review fixes (TP consistency after a failed begin, zero-byte pages, the off-by-one at `--context`, `mtp==0` with drafts), device cache on pages, the TooMany `fits:` clause, integ fixes | Correctness |
| 0049–0063 | Paired prompt chunks (upstream #1423), extended to DFlash2 taps | Part of the wave-3 gain |
| 0064 | `TF_GLM_W8A8` opt-in, 8-bit prompt linears | Prefill +10% alone, +19% with ABSORB; off |
| 0065–0066 | Test fixes | — |
| 0067 | Many-CTA decode DSA select (attn5) | 200K step +3.3%, ~690K step −11.7% |
| 0068 | Prompt sparse attention merged in the kernel (attn5) | Prefill +2.7%; frees 353 MiB a GPU |
| 0069–0071, 0077, 0083 | Audited programmatic dependent launch along the decode chain (pdl5) | Plain step +3.1% (0069-0071); 0083 +0.3% |
| 0072–0075 | Decode glue fusions (glue5) | 4 streams +5.5%, plain +1.6-2.2% |
| 0078–0080 | Faster EXL3 prompt experts and packed prompt GEMMs (exl3p) | Prefill +8.9–9.8% |
| 0081 | Grouped EXL3 decode reads each trellis word once (exl3r) | Edits +8.8%, 4 streams +4.8% |
| 0082 | The EXL3 decode chain as dependent launches (exl3r) | Plain step +1.0% |
| 0084 | KDA step warps by row count (pdl5) | +0.4% |
| 0086, 0088, 0089 | Split hc_finish, batched shared-window writes, hc_partial8 split (glue5; c9's default 2:16) | ~0 / ~0 / 1-3% inside the stack |
| 0090 | PDL waits for the split kernels of 0086 and 0089 (pdl5) | Required for safety: never ship 0086/0089 without it |
| 0091 | Decode DSA index scores for 4–16-row windows (attn5) | ~0 at 200K, +6% on 16-row edits at ~690K (one pair) |
| 0076, 0079, 0085, 0087 | Test and contract-record changes | — |

Every patch from 0067 keeps an exact off switch (`TF_GLM_DSEL`, `TF_GLM_PDL_CHAIN`, `TF_GLM_MM_TWO`, `TF_GLM_TOPK_SUM`,
`TF_GLM_SHARED_SEGS`, `TF_EXL3_SHFL`, `TF_EXL3_PDL`, `TF_GLM_KDA_STEP_WARPS=8`, `TF_GLM_HC_SPLIT`,
`TF_GLM_SHARED_WRITES`, `TF_GLM_HC_PART`, `TF_GLM_DSCORES`; table in `integ5/SHIP.md`), except 0068, 0072, 0078 and
0080, which are exact by construction.

The 2026-10-31 night's tested-and-dropped work (graph5's drafter changes, glue5 c7, TPM=128, KDA row overlap, R1
#1571, L2 prefetch, and more) is listed with numbers in `opt/overnight/REPORT.md`.

## Decisions and rules

The user's decisions:

- Keep 4 streams on one shared pool, and keep the context as large as it will go. (2026-10-31: asked about 16
  sessions, then said "nvm"; still 4.)
- Don't suggest moving the monitors off GPU0, or trading context windows for prompt cache.
- The 4 bpw model is production. The 3.5 bpw GLMQ quant is kept for reference only.
- `ABSORB_MMA` is on.
- The LM head stays bf16; a q8 head measured only +1–2%.
- Dense stays q8. q6 showed no gain. q4 saves 1.06 ms per step but risks quality.
- `W8A8` stays off. On 2026-10-31 it failed the predeclared NLL check by 0.0006 nats (+0.0074 against a largest null
  of +0.0068). Its accuracy arm was not run (`opt/overnight/gate5.md` has the command).
- The 91-patch image shipped on 2026-10-31 11:48 at the user's request ("Commit, make it the default in the recipe,
  and restart the server with it").

Rules:

- **Shipping:** an exact patch ships only with identical greedy hashes in a same-slot ABBA A/B and a measured gain. Opt-in numerics flags stay off unless the user turns them on.
- **Judging numerics changes:** any 1-ulp change cascades through the MoE top-8 near-ties, giving a KL floor of about 0.15. Judge numerics changes by NLL against the null spread (`opt/num3/qh.py`, `opt/overnight/gate/`) and by answers, not by KL or hashes.
- **PDL:** after any change to a launch with the PDL attribute, run `pdl_sites.py` (in `opt/overnight/pdl5/tools/`); it must PASS (20 sites, 60 kernels today).
- **Git:** commit or push only when the user asks. Commits use MiaAI-Lab `<mia.ailab1@gmail.com>`.
- **Power:** never change GPU power limits; give the user the command, since they have sudo.
- **spark1:** read-only, except `~/codex-glm53-sm120`.
- **Secrets:** never print secrets; pass `TENSORFOLD_API_KEY` by name only.
- **Banned sources:** the code ban on r0b0tlab and vcruz305 is waived for this work, by the user's call.

## How to operate

1. **Start or restore:** in the recipe, run a plain `./start.sh` with no overrides.
   - It rebuilds the image if the patches changed (in docker, so it works without the host toolkit), fits the
     context, and starts `glm53-zig` on 8888.
   - `./stop.sh` stops it. While the watchdog runs, touch `opt-sm120-notes/opt/overnight/STOP_WATCHDOG` first, or it
     restarts the server within ~10 minutes.
2. **Health check:** `opt-sm120-notes/opt/overnight/health.sh` must print `OK 391 | Canberra` (it asks "What is 17*23?
   Answer with just the number, then name the capital of Australia." at temperature 0 on 8888).
   - For speed: `opt/overnight/prof5/bench5.py` with `BENCH_URL=http://127.0.0.1:8888 CNAME=glm53-zig` and a
     `BENCH_OUT` of your own (modes `answer`, `t0 1`, `nd`, `par`, `prefill`, `long`, ...). Greedy shas to expect:
     answer 69d29804284e, prose_t0 17318206e1f3, code_t0 9d917d073c72.
3. **Change a setting:** edit `scripts/local.sh`, then restart.
   - A setting in the fit key (model, HEAD, DENSE, KV, KV_PAGES, reserve, and so on) triggers a new context fit.
   - `CONTEXT=auto` tries `CONTEXT_MAX` first, then follows the `fits: … --context T` hint at a 0.93 margin.
4. **GPU slots:** only one job at a time may stop the server. Use `opt/overnight/slot_begin.sh HOLDER MINUTES "WHAT"`
   (writes `SLOT_ACTIVE`, stops production; `TIMING=1` forbids builds), `slot_extend.sh`, `slot_handover.sh`, and
   `slot_end.sh` (plain `./start.sh` plus the health check; retires the flag only on OK).
   - `opt/prof_up.sh` brings up measuring containers; its default IMAGE is still `f20c005d1c5d`, so pass
     `IMAGE=tensorfold-glm53-zig:987b580ba9d2`.
   - Don't edit `start.sh` while a restore may be running.
5. **A/B method:** run ABBA legs in the same slot, compare greedy hashes, and tabulate with
   `opt/overnight/prof5/ab/run_ab5.sh` and `table5.py` (older: `opt/abbench.py`, `opt/abtable.py`).
6. **Codex** (on spark1, with a brief in `brief.md`):

   ```
   ssh spark1
   cd ~/codex-glm53-sm120/roundN
   setsid nohup ~/.local/bin/codex exec -s read-only -C <dir> --skip-git-repo-check --ephemeral \
     -c model_reasoning_effort="xhigh" -o answer.md - < brief.md > log 2>&1 & disown
   ```

7. **DeepSeek:** an OpenAI-style chat call to `http://spark1:8888/v1`, model `DeepSeek-V4.1-Flash-1017`, with `max_tokens` 262144, `reasoning_effort` xhigh and no client timeout. The user said "let it think". The old helper script lived in the session scratchpad under `/tmp`, so it is gone after a reboot.

## Running in the background

- **`opt/overnight/watchdog2.sh`** (pid in `watchdog.pid`, log `watchdog.log`): health-checks 8888 every 5 minutes and
  restores with `./start.sh` after two failures or a hung engine. It stays out of active slots. It exits on
  `opt/overnight/STOP_WATCHDOG` or at 2026-11-03 00:42.
- **`opt/overnight/usage_sampler.sh`**: one line a minute of 8888's load counters into `usage.csv`; exits with the
  watchdog's STOP file.

## Open work

None of this has started, and each item needs the user's go-ahead.

| Item | What it gives | State and gate |
| --- | --- | --- |
| A full 1,048,576 window | One request at the model's native max | `CONTEXT=1048576` pinned, reserve ~2.7; thinner margin for GPU0's desktop |
| T7 accuracy arm | Whether W8A8's NLL shift costs accuracy per stratum | ~2 h slot; commands in `opt/overnight/gate5.md`; cannot change the FAIL |
| attn5 c4 at ~690K | Confirm its +6% on long-context verify | One A D D A at ~690K |
| c9 table (`TF_GLM_HC_PART=2:16`) | Resolve its 1-3% | A quiet A M M A |
| Simplify 0086 and 0088 | Less code | Both measured ~0 inside the stack |
| Mixed traffic policy | Decoders get ~1 round per 2048-row chunk during a long prefill | `TENSORFOLD_PREFILL_SHARE`: p=0.84 → TTFT +15%, decoders 3.4×; p=0.68 → +42%, 6.2× |
| Grow-on-demand KV admission | Admit by prompt + min(max_tokens, `--max-tokens`) and grow pages as needed, so more long requests fit at once | Design only: `opt/mem3.md` §4 |
| Copy-on-write prefix sharing | A prefix shared across streams is stored once | Idea |
| q4 dense or a refined W8A8 (GS32, MLP-only) | −1.06 ms per step (q4); +5.5–6% prefill (W8A8) | Gated on the T7 harness (`opt/overnight/gate/`) |
| Cleanup | 37 engine worktrees, ~14 GB of night binaries, 10 old `tensorfold-glm53-zig` images (~4.8 GB each), `glue5/trees/` | Ask before deleting |

The user still has decisions to make:

- **Reinstall the host CUDA toolkit:** `sudo apt install -y cuda-toolkit-13-3` (needed for host builds and SASS tools).
- **Memory clock** P1 → P0 (up to ~4.5% decode). Root.
- **Asymmetric power caps**, for example 440/360 W with the higher cap on the slower GPU0. This needs sudo.
- **`--recall`**, which makes repeated prompts 1.5–3× faster but writes token ids to disk.

## Known pitfalls

- **`apt autoremove` removes the CUDA toolkit.** On 2026-10-31 10:55 `apt purge ghostty && apt autoremove -y` took
  all of CUDA 13.3 (nvcc, cuobjdump, nsight, libraries) because those packages were marked auto-installed. Installing
  `cuda-toolkit-13-3` explicitly marks it manual. Until then, compile and inspect kernels inside
  `nvidia/cuda:13.3.1-devel-ubuntu24.04` (cached locally).
- **Lane sharing:** a drive in M.2_2 drops GPU1 to PCIe x4.
- **After a reboot:** check that the caps are still 400 W with `nvidia-smi --query-gpu=power.limit --format=csv`. The unit was once a disabled 300 W one.
- **Memory reserve:** below 3.5 the margin for GPU0's desktop shrinks (it moved ~0.7 GiB within an hour on 10-31). At 4, the plan fails TooMany, 0.05 GiB short of 4 streams. If long sessions hit out-of-memory errors, go back to 3.5.
- **Self-matching patterns:** `pgrep -f` and `pkill -f` match their own command line, including a `while pgrep -f …` loop's own shell. Use a bracket pattern such as `[r]un_c\.sh`, or wait on a pid with `kill -0`.
- **On spark1:** never `pkill -f` the Codex round dir, because it kills your own ssh shell. A trailing `< /dev/null` replaces the brief and Codex prints "No prompt provided via stdin".
- **DeepSeek token cap:** a capped `max_tokens` gets used up by reasoning and the answer comes back empty.
- **Sharing GPU slots:** two agents sharing a slot spoils both sets of timings, and editing a script while it runs loses that leg. Host builds during a timing slot slowed legs 8-20%; daytime desktop use on GPU0 moved one leg ~4%.
- **Stopped agents:** an agent the user stopped can't be resumed with SendMessage. Start a fresh one from its report.
- **KV page tables differ between streams by construction:** a state compare across streams must skip the table and compare page contents.
