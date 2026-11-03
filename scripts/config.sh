#!/usr/bin/env bash
# Settings in precedence order: exported environment, scripts/local.sh, .env, the defaults below.
# local.sh is trusted Bash. .env is read as literal KEY=value lines, never executed.
# Sourced by scripts/prepare.sh, start.sh and stop.sh; it runs no Docker, GPU or network command.
[[ "${_GLM_CONFIG_LOADED:-0}" == 1 ]] && return 0
_GLM_CONFIG_LOADED=1
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -f "$REPO_ROOT/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _cfg_key; do _cfg_env[$_cfg_key]=${!_cfg_key}; done < <(compgen -e)
  # shellcheck source=/dev/null
  source "$REPO_ROOT/scripts/local.sh"
  for _cfg_key in "${!_cfg_env[@]}"; do export "$_cfg_key=${_cfg_env[$_cfg_key]}"; done
  unset _cfg_env
fi
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r _cfg_line || [[ -n "$_cfg_line" ]]; do
    [[ "$_cfg_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _cfg_key=${BASH_REMATCH[2]}; _cfg_value=${BASH_REMATCH[3]}
    if [[ "$_cfg_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_cfg_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _cfg_value=${BASH_REMATCH[1]}
    else
      _cfg_value=${_cfg_value%%#*}; _cfg_value=${_cfg_value%"${_cfg_value##*[![:space:]]}"}
    fi
    [[ -n "${!_cfg_key+set}" ]] || export "$_cfg_key=$_cfg_value"
  done < "$REPO_ROOT/.env"
fi
unset _cfg_key _cfg_line _cfg_value

# --- pinned engine build (hashed into the image tag with docker/ and patches/engine/*.patch) ----------------------
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-f350cd3ea253a042e205746928271f9d8f71a82d}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026}"
SM="${SM:-120}"                                  # RTX PRO 6000 Blackwell: sm_120
IMAGE="${IMAGE:-}"                               # empty: tensorfold-glm53-zig:<PATCHES_HASH>
IMAGE_REPO="${IMAGE_REPO:-tensorfold-glm53-zig}"
DOCKER_BUILD_FLAGS="${DOCKER_BUILD_FLAGS:-}"     # extra docker build flags, word-split (e.g. --progress=plain)
TF_BIN="${TF_BIN:-/usr/local/bin/tensorfold-native}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-zig}"   # never the Python recipe's glm53-tf
CONTAINER_USER="${CONTAINER_USER-$(id -u):$(id -g)}"   # empty or root: run the ranks as root

# --- weights ---------------------------------------------------------------------------------------------------------
# download (default): fetch the pinned 4 bpw checkpoint into MODEL_DIR with the hf CLI when it is missing.
# local: serve the EXL3 checkpoint at MODEL_DIR read-only, no download.
DATA_DIR="${DATA_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/aevonix-glm53}"
MODEL_SOURCE="${MODEL_SOURCE:-download}"
# ABLIT=1 serves the Ablit weights (the same checkpoint, abliterated; README "Ablit weights") instead of the published
# one; 0 (default) the published one. The Ablit repository is gated: fetching it needs HF_TOKEN, from an account that
# accepted the terms on the model's page. It downloads beside the published checkpoint, which stays in the cache.
ABLIT="${ABLIT:-0}"
ABLIT_ID="Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit"
_id="Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"; _dir=model
if [[ "$ABLIT" == 1 ]]; then _id=$ABLIT_ID; _dir=model-ablit; fi
MODEL_ID="${MODEL_ID:-$_id}"
# The pins belong to the checkpoints above; another MODEL_ID needs its own MODEL_REVISION.
case "$MODEL_ID" in
  Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) _rev=76c0b5173166d2795dd48860f45d8224817f894c ;;
  "$ABLIT_ID") _rev=57edefd2f5d9b371c8345883304d5af68b52fa24 ;;
  *) _rev="" ;;
esac
MODEL_REVISION="${MODEL_REVISION:-$_rev}"
MODEL_DIR="${MODEL_DIR:-$DATA_DIR/$_dir}"
unset _id _dir _rev
# HF_TOKEN from scripts/local.sh or .env reaches the hf CLI only when exported. It is never passed to the ranks.
[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
DFLASH2_ID="${DFLASH2_ID:-incoai/GLM-5.3-Flash-DFlash2}"          # CC BY-NC-ND 4.0: non-commercial
DFLASH2_REVISION="${DFLASH2_REVISION:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
DFLASH2_DIR="${DFLASH2_DIR:-$DATA_DIR/dflash2}"                   # downloaded at the pinned revision when missing
# A mixed-width EXL3 quant names avg_bits, not bits; the stock GLM check wants quantization_config.bits == 4.
# auto: mount a copy of config.json with bits 4 added over /model/config.json (only when bits is missing); 0: never.
CONFIG_OVERLAY="${CONFIG_OVERLAY:-auto}"
CACHE_DIR="${CACHE_DIR:-$DATA_DIR/zig-home}"     # the ranks' HOME: the engine's per-file checkpoint digest cache
STATE_DIR="${STATE_DIR:-$REPO_ROOT/.state}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
LOG_KEEP="${LOG_KEEP:-10}"                       # newest compressed session logs kept

# --- placement: one host, one container, one tensor-parallel rank per GPU ----------------------------------------------
# GPUS order is rank order: GPUS=0,1 puts rank 0 (HTTP server, vision tower, heavier) on GPU 0, rank 1 on GPU 1 (the desktop's GPU).
GPUS="${GPUS:-0,1}"
NUM_GPUS=$(( $(tr -cd , <<< "$GPUS" | wc -c) + 1 ))   # TF_GLM_WORLD
PORT="${PORT:-8020}"
HOST="${HOST:-0.0.0.0}"
NCCL_PORT="${NCCL_PORT:-29583}"                  # NCCL_COMM_ID=127.0.0.1:NCCL_PORT; the rank watch uses NCCL_PORT+1
SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash}"

# --- serving --------------------------------------------------------------------------------------------------------
DRAFTER="${DRAFTER:-dflash2}"                    # dflash2 (--drafter DIR) | mtp (the checkpoint's MTP head) | none (--no-drafts)
DRAFTER_BITS="${DRAFTER_BITS:-4}"                # --drafter-bits: 4 (default) or 0 (bf16 drafter)
MTP_DRAFTS="${MTP_DRAFTS:-}"                     # --mtp-drafts N (empty: the engine's policy)
MTP_CONFIDENCE="${MTP_CONFIDENCE:-}"             # --mtp-confidence F (empty: off)
PARALLEL="${PARALLEL:-4}"                        # --parallel: streams at once (a number is required to fit; auto: up to 8)
# TENSORFOLD_PREFILL_SHARE: the share of wall time a long prompt's 2,048-row chunks get while other streams decode. 1: one
# decode round per chunk, so a stream decoding beside a long prefill crawls (~4-5 tok/s during a 128K prompt). 0.68: the
# chunk owes the decoders ~10 rounds (34-40 tok/s a stream during that prefill; its own first token +42%, 26.9 -> 38.1 s);
# 0.84: ~4 rounds (19-22 tok/s, +15%). Replies are the same at any value: it only moves when the next chunk runs.
PREFILL_SHARE="${PREFILL_SHARE:-0.68}"
# TENSORFOLD_PREFILL_BUDGET: prompt rows filled in one visit while a stream waits to decode (empty: no cap)
PREFILL_BUDGET="${PREFILL_BUDGET:-}"
# auto: the largest window that fits. The first start tries CONTEXT_MAX (the model's 1,048,576), and when the engine
# refuses it with its memory figures, starts once more with the window they allow; the window that served is kept in
# STATE_DIR/context-fit.tsv for the next start with the same memory settings.
CONTEXT="${CONTEXT:-auto}"
CONTEXT_MAX="${CONTEXT_MAX:-1048576}"
CONTEXT_RETRY="${CONTEXT_RETRY:-1}"              # 1: one more start with a fitting window after a memory refusal
CONTEXT_FIT_MARGIN="${CONTEXT_FIT_MARGIN:-0.93}" # share of the computed window the retry asks for
KV="${KV:-fp8}"                                  # --kv-cache: fp8 (a window per stream, or one shared pool with KV_PAGES=on) | bf16 (shared KV pages)
# TF_KV_PAGES. on: every stream draws its KV from one shared pool of 512-row pages, and CONTEXT becomes one request's
# cap (prompt + max_tokens), up to 1048576; a request that doesn't fit waits for pages. off: with KV=fp8 a fixed window
# per stream. auto: the engine's default (bf16 on pages, fp8 windows). fp8 on pages needs engine patches 0017-0025.
KV_PAGES="${KV_PAGES:-on}"
DENSE="${DENSE:-q8}"                             # TF_GLM_DENSE_BITS: bf16 (16) | q8 | q6 | q4 (symmetric int codes, groups of 64)
HEAD="${HEAD:-bf16}"                             # TF_GLM_HEAD_BITS for the LM head, same choices
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-4}"    # TENSORFOLD_MEMORY_RESERVE_GIB per GPU (engine default on 96 GB: 3)
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"         # --prompt-cache-gib (empty: engine default, 16 GiB at most of what is left)
VISION="${VISION:-1}"                            # --vision: images (rank 0's tower)
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"     # --vision-max-images: images per request (engine default 4, at most 64)
# --chat-template: auto = with VISION=1, templates/glm-5.3-flash-vision.jinja (the checkpoint's own template, whose media
# branch writes GLM's image and video markers instead of a "cannot see" note); none = the checkpoint's; or a file path
CHAT_TEMPLATE="${CHAT_TEMPLATE:-auto}"
THINKING="${THINKING:-0}"                        # --thinking / --no-thinking default; requests override
REASONING_EFFORT="${REASONING_EFFORT:-high}"     # --reasoning-effort low|medium|high|xhigh (empty: omit)
MAX_TOKENS="${MAX_TOKENS:-32768}"                # --max-tokens default reply limit
CLEAR_THINKING="${CLEAR_THINKING:-1}"            # TF_GLM_CLEAR_THINKING: 1 drops earlier turns' reasoning, 0 keeps it
CE_EXCHANGE="${CE_EXCHANGE:-1}"                  # TF_GLM_CE_EXCHANGE: 1 moves prompt exchanges by copy engines (+4.5% prefill, same bits)
# Opt-in prompt numerics (engine patches 0041 on): faster prefill, not bit-identical to stock (outputs at stock's own
# rounding-noise level on our eval). ABSORB_MMA=1: absorb_q/expand_v on tensor cores (+6.6% prefill, fp32 reorder only).
ABSORB_MMA="${ABSORB_MMA:-0}"                    # TF_GLM_ABSORB_MMA
W8A8="${W8A8:-0}"                                # TF_GLM_W8A8: int8 activations in prompt dense GEMMs (+10% prefill, +19% with ABSORB_MMA;
                                                 # a real rounding of activations: check your own eval before turning it on)
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"         # more tensorfold-native serve flags, word-split, given to every rank

# --- the GPUs' link ----------------------------------------------------------------------------------------------------
# Both ranks share the host's network namespace and talk over loopback; NCCL may use PCIe peer-to-peer up to SYS
# (the two cards sit behind the CPU's host bridge). Any other exported NCCL_* variable is passed to the ranks as well.
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# --- checks and timeouts -----------------------------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"                          # 1: print the plan; no Docker, GPU, download or file change
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"             # seconds for the API to come up (digest, load, graphs)
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-180}"
STOP_TIMEOUT="${STOP_TIMEOUT:-45}"               # docker stop grace: rank 0 drains for TF_DRAIN_S (20 s) first
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-88000}"    # per selected GPU
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-4096}"     # memory in use, and the largest compute process tolerated (desktop)
MIN_HOST_FREE_GIB="${MIN_HOST_FREE_GIB:-16}"     # MemAvailable below this warns (load staging and page cache)
IMAGE_FREE_GIB="${IMAGE_FREE_GIB:-40}"
MODEL_FREE_GIB="${MODEL_FREE_GIB:-180}"
DFLASH2_FREE_GIB="${DFLASH2_FREE_GIB:-10}"
