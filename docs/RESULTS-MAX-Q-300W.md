# Measured results: Max-Q (300 W) edition

Reported by a deployment on 2× **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition**
(96 GiB each, sm_120, driver 580.178.04, CUDA 13.3) — the 300 W variant, not the 400 W
configuration the README's official table was measured on.

The point of this page is the delta between the two power envelopes and the
engine settings that recovered part of it. Nothing here changes the recipe's
defaults; every row is a measurement on the checkpoint and image pinned by this
repository.

## Hardware difference

| | Max-Q (this report) | Official table |
| --- | --- | --- |
| Part | RTX PRO 6000 Blackwell **Max-Q** Workstation Edition | RTX PRO 6000 Blackwell |
| Memory | 96 GiB GDDR7 ECC, 512-bit, 1,792 GB/s | same |
| CUDA cores / Tensor cores | 24,064 / 752 (GB202) | same |
| Power cap | **300 W default, 325 W hard maximum** | **400 W** |
| SM clock under load | ~2.05 GHz at 300 W, ~2.22 GHz at 325 W (peak 3.09 GHz) | not stated |

The die, the memory capacity and the memory bandwidth are identical across the
RTX PRO 6000 Blackwell family; the Max-Q edition differs in board power and
cooling only. `nvidia-smi` reports `Max Power Limit: 325.00 W` on this part, so
`-pl` cannot reach 400 W.

Under sustained decode both cards sit exactly on the cap (300.1 W / 300.1 W at the
300 W setting, 317.6 W / 323.1 W at 325 W) with `SW Power Cap` active. That is the
source of the decode gap below, not a configuration error.

## Results

Checkpoint: `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit` (the gated,
abliterated sibling; same EXL3 4-bit layout as the published checkpoint).
Engine: the image this repository builds (`tf.patches=b2364db4a201`).
Decode figures are the engine's own per-request `tok/s` log line (summed across
streams for the aggregate), temperature 0.7, thinking off, 600-token completions,
3 repetitions each.

### Tuned configuration

```
PARALLEL=8
CONTEXT=262144
KV=fp8
DENSE=q4
HEAD=q6
MEMORY_RESERVE_GIB=3.5
W8A8=1
ABSORB_MMA=1
export TF_GLM_DRAFT_CONFIDENCE=0.85
GPUS=0,1
```

GPU clocks: `-pm 1`, `-pl 325` on both cards (the hardware maximum).

| Decode, prose | 1 stream | 2 streams | 4 streams |
| --- | ---: | ---: | ---: |
| tok/s | 197.9 | 290.6 | 395.0 |
| vs official (400 W) | 81% | 86% | 90% |

| Prefill, prompt tokens | ~3.7K | ~14.8K | ~59K | ~118K |
| --- | ---: | ---: | ---: | ---: |
| tok/s | 5,375 | 7,134 | 8,672 | 12,324 |
| TTFT | 0.69 s | 2.07 s | 6.81 s | 9.57 s |
| vs official (400 W) | 90% | 124% | 144% | 213% |

Prefill is a short burst and stays inside the power envelope, so it reaches and
exceeds the 400 W numbers. Decode is a sustained load and is capped, which is why
the loss is concentrated there.

### What each setting contributed

Measured against the previous row, one setting at a time, on the same host.

| Setting | Decode, 1 stream | Decode, 4 streams | Prefill, 59K |
| --- | --- | --- | --- |
| Recipe defaults (`DENSE=q8`, `HEAD=bf16`, no `W8A8`) | 187 tok/s | 377 tok/s | 6,128 tok/s |
| `+ DENSE=q4` | 193.4 | 391.3 | — |
| `+ HEAD=q6` | 205.0 | 389.7 | — |
| `+ W8A8=1, ABSORB_MMA=1` | 186.6 | 383.1 | 6,987 |
| `+ TF_GLM_DRAFT_CONFIDENCE=0.85` | **197.9** | **395.0** | 8,672 |

`W8A8` and `ABSORB_MMA` do not move decode (they are prompt GEMM paths); the last
row combines the drafting change with the full tuned set, which is why its prefill
is higher than the row above.

### Drafting threshold sweep

`TF_GLM_DRAFT_CONFIDENCE` (the DFlash2 chain's stopping point, engine default 0.7)
was swept on real prose prompts:

| Value | Chain acceptance | Decode, 1 stream | Decode, 4 streams |
| --- | ---: | ---: | ---: |
| 0.7 (default) | 55.6% | 200.9 | 380.0 |
| **0.85** | **62.4%** | **202.2** | **397.1** |
| 0.95 | 65.5% | 195.0 | 392.1 |

Raising the threshold keeps raising acceptance, but throughput peaks at 0.85:
past it, the longer chains cost more to verify than the extra accepted tokens
return. Acceptance rate alone is not a proxy for speed — it has to be measured.

### Rejected after measurement

- `HEAD=q4`: 196.3 tok/s at 1 stream, below `HEAD=q6`'s 205.0. Not worth the
  output-quality risk.
- `DENSE=q3`: refused — the engine accepts `bf16`, `q8`, `q6` or `q4` only.
- `PARALLEL=8` vs 4: 8 streams admit and run (8 × 395 MiB fixed plus KV pages fit
  after `MEMORY_RESERVE_GIB=3.5`), but aggregate throughput at 4 streams already
  saturates the cards; beyond 8 requests queue (`queued=` in the rank log grows).
- Power cap 300 → 325 W: +2-4% decode. The board maximum, so there is no more to take.

## Deployment notes

- **Cold start is ~15-16 minutes** on this host: each rank reads and hashes the
  156.4 GiB checkpoint once. The digest is cached in `CACHE_DIR`, so later starts
  skip it (`83/83 cached files`) — but the weights are still read from disk and
  loaded, which dominated every restart measured here.
- **`ABLIT=1` and `MODEL_SOURCE=local` are mutually exclusive.** `ABLIT=1` implies
  `MODEL_SOURCE=download`; to serve a gated copy already on disk, set `ABLIT=0`,
  `MODEL_SOURCE=local` and point `MODEL_DIR` at it. The config refuses the other
  combination with a clear message, but it costs a start attempt to find out.
- **`ABLIT=1` changes the expected directory** to `DATA_DIR/model-ablit`, not
  `DATA_DIR/model`. A checkpoint placed in the wrong one is refused as "holds files
  this recipe did not download".
- **The NVIDIA container runtime must exist before the first start**
  (`nvidia-ctk runtime configure --runtime=docker`). `docker run --gpus all`
  working is not sufficient: the recipe checks the runtime entry itself.
- **Downloads over a mirror** need `HF_ENDPOINT` as well as `HF_TOKEN`; the drafter
  is fetched with the same CLI, so the endpoint has to be exported for `start.sh`
  too, not only for a manual `hf download`.
- **`--mtp-drafts` is refused for this model** on the CUDA backend ("drafts at the
  lanes' depth"), and `--mtp-confidence` conflicts with `--drafter`. The drafting
  chain depth is not user-tunable here; `TF_GLM_DRAFT_CONFIDENCE` is the knob that
  is.
