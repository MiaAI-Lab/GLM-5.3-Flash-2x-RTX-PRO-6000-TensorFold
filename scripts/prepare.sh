#!/usr/bin/env bash
# Build the engine image and check the weights once. Also sourced by start.sh and stop.sh for its helpers.
# Usage: scripts/prepare.sh [--image|--weights|--check|--label]
#   (none)     image and weights          --image    the image only       --weights  the weights only
#   --check    platform, GPUs and CUDA peer access (needs the built image and idle GPUs)
#   --label    print the image hash (tf.patches label) and exit
# DRY_RUN=1 prints the plan without Docker, GPUs, downloads or file changes.
set -euo pipefail
# shellcheck source=scripts/config.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

log() { printf '%s\n' "$*"; }
fail() { printf '[%s] %s\n' "$1" "$2" >&2; exit 1; }
warn() { printf '[%s] %s\n' "$1" "$2" >&2; }
# Words as a shell would read them back: plain when safe, else single-quoted.
shq() { local a; for a in "$@"; do if [[ "$a" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then printf '%s ' "$a"; else printf "'%s' " "${a//\'/\'\\\'\'}"; fi; done; }
plan() { printf '  '; shq "$@"; printf '\n'; }
run() { if [[ "$DRY_RUN" == 1 ]]; then plan "$@"; else "$@"; fi; }
is_number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

validate_config() {
  local key path
  [[ "$DRY_RUN" =~ ^[01]$ ]] || fail E_CONFIG 'DRY_RUN must be 0 or 1.'
  [[ "$MODEL_SOURCE" =~ ^(local|download)$ ]] || fail E_CONFIG 'MODEL_SOURCE must be local or download.'
  [[ "$DRAFTER" =~ ^(dflash2|mtp|none)$ ]] || fail E_CONFIG 'DRAFTER must be dflash2, mtp or none.'
  [[ "$KV" =~ ^(fp8|bf16)$ ]] || fail E_CONFIG 'KV must be fp8 or bf16 (--kv-cache).'
  [[ "$KV_PAGES" =~ ^(on|off|auto)$ ]] || fail E_CONFIG 'KV_PAGES must be on, off or auto (TF_KV_PAGES).'
  [[ "$CE_EXCHANGE" =~ ^[01]$ && "$ABSORB_MMA" =~ ^[01]$ && "$W8A8" =~ ^[01]$ ]] || fail E_CONFIG 'CE_EXCHANGE, ABSORB_MMA and W8A8 must be 0 or 1.'
  if [[ "$DENSE" == fp8 ]]; then
    warn W_DENSE 'The native engine has no FP8 dense weights: DENSE=fp8 is served as DENSE=q8 (8-bit codes, groups of 64).'
    DENSE=q8
  fi
  for key in DENSE HEAD; do
    [[ "${!key}" =~ ^(bf16|q8|q6|q4)$ ]] || fail E_CONFIG "$key must be bf16, q8, q6 or q4 (TF_GLM_${key}_BITS 16, 8, 6 or 4)."
  done
  for key in MAX_TOKENS PORT NCCL_PORT LOG_KEEP WAIT_TIMEOUT SMOKE_TIMEOUT STOP_TIMEOUT MIN_GPU_FREE_MIB MAX_GPU_USED_MIB \
             MIN_HOST_FREE_GIB IMAGE_FREE_GIB MODEL_FREE_GIB DFLASH2_FREE_GIB CONTEXT_MAX SM; do
    [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || fail E_CONFIG "$key must be a positive integer."
  done
  [[ "$PARALLEL" == auto || ( "$PARALLEL" =~ ^[1-9][0-9]*$ && PARALLEL -le 64 ) ]] || fail E_CONFIG 'PARALLEL must be auto or 1-64.'
  (( CONTEXT_MAX <= 1048576 )) || fail E_CONFIG 'CONTEXT_MAX must be at most 1048576 (the model window).'
  [[ "$CONTEXT" == auto || ( "$CONTEXT" =~ ^[1-9][0-9]*$ && CONTEXT -le 1048576 ) ]] || fail E_CONFIG 'CONTEXT must be auto or a token count from 1 to 1048576.'
  [[ "$CONTEXT_RETRY" =~ ^[01]$ ]] || fail E_CONFIG 'CONTEXT_RETRY must be 0 or 1.'
  is_number "$CONTEXT_FIT_MARGIN" && python3 -c 'import sys; sys.exit(0 if 0 < float(sys.argv[1]) <= 1 else 1)' "$CONTEXT_FIT_MARGIN" \
    || fail E_CONFIG 'CONTEXT_FIT_MARGIN must be a fraction in (0, 1].'
  (( PORT <= 65535 && NCCL_PORT < 65535 && PORT != NCCL_PORT && PORT != NCCL_PORT + 1 )) \
    || fail E_CONFIG 'PORT and NCCL_PORT must be distinct ports (the ranks also use NCCL_PORT+1).'
  [[ "$VISION_MAX_IMAGES" =~ ^[1-9][0-9]*$ && VISION_MAX_IMAGES -le 100 ]] || fail E_CONFIG 'VISION_MAX_IMAGES must be 1-100.'
  for key in THINKING VISION CLEAR_THINKING; do [[ "${!key}" =~ ^[01]$ ]] || fail E_CONFIG "$key must be 0 or 1."; done
  [[ -z "$REASONING_EFFORT" || "$REASONING_EFFORT" =~ ^(low|medium|high|xhigh)$ ]] || fail E_CONFIG 'REASONING_EFFORT must be low, medium, high, xhigh or empty.'
  [[ "$EFFORT_LINE" =~ ^(thinking|always)$ ]] || fail E_CONFIG 'EFFORT_LINE must be thinking or always.'
  [[ "$DRAFTER_BITS" =~ ^[04]$ ]] || fail E_CONFIG 'DRAFTER_BITS must be 4 or 0 (bf16).'
  [[ -z "$MTP_DRAFTS" || ( "$MTP_DRAFTS" =~ ^[0-9]+$ && MTP_DRAFTS -le 64 ) ]] || fail E_CONFIG 'MTP_DRAFTS must be empty or 0-64.'
  [[ -z "$MTP_CONFIDENCE" ]] || { is_number "$MTP_CONFIDENCE" && python3 -c 'import sys; sys.exit(0 if 0 <= float(sys.argv[1]) <= 1 else 1)' "$MTP_CONFIDENCE"; } \
    || fail E_CONFIG 'MTP_CONFIDENCE must be empty or a probability in [0, 1].'
  is_number "$MEMORY_RESERVE_GIB" && python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 0.5 else 1)' "$MEMORY_RESERVE_GIB" \
    || fail E_CONFIG 'MEMORY_RESERVE_GIB must be a number of GiB, at least 0.5 (the engine floor on discrete GPUs).'
  [[ -z "$PROMPT_CACHE_GIB" ]] || is_number "$PROMPT_CACHE_GIB" || fail E_CONFIG 'PROMPT_CACHE_GIB must be empty or a number of GiB (0: none).'
  [[ -z "$PROMPT_CACHE_MIN_GIB" ]] || is_number "$PROMPT_CACHE_MIN_GIB" \
    || fail E_CONFIG 'PROMPT_CACHE_MIN_GIB must be empty (no check) or a number of GiB.'
  is_number "$LEARN_GIB" || fail E_CONFIG 'LEARN_GIB must be a number of GiB (0: off).'
  is_number "$LEARN_MIN_FREE_GIB" || fail E_CONFIG 'LEARN_MIN_FREE_GIB must be a number of GiB.'
  [[ "$KV_SHARD" =~ ^[01]$ ]] || fail E_CONFIG 'KV_SHARD must be 0 or 1 (TF_GLM_KV_SHARD).'
  [[ "$L2PF" =~ ^[01]$ ]] || fail E_CONFIG 'L2PF must be 0 or 1 (TF_GLM_L2PF).'
  [[ "$EXTRA_EOS" == none || "$EXTRA_EOS" =~ ^[^[:space:],]+(,[^[:space:],]+)*$ ]] \
    || fail E_CONFIG 'EXTRA_EOS must be none or a comma-separated list of tokens, e.g. <|assistant|>.'
  [[ "$KV_SHARD" == 0 || ( "$KV" == fp8 && "$KV_PAGES" == on && "$DRAFTER" != mtp && "$PARALLEL" != 1 && "$GPUS" =~ ^[0-9]+,[0-9]+$ ) ]] \
    || fail E_CONFIG 'KV_SHARD=1 needs two GPUs in GPUS, KV=fp8, KV_PAGES=on, DRAFTER=dflash2 or none, and PARALLEL=auto or above 1.'
  [[ "$WITH_VIDEO" =~ ^[01]$ ]] || fail E_CONFIG 'WITH_VIDEO must be 0 or 1.'
  [[ "$CONFIG_OVERLAY" =~ ^(auto|0)$ ]] || fail E_CONFIG 'CONFIG_OVERLAY must be auto or 0.'
  # An NCCL exclusion list (^lo,docker,...), often exported for the whole box from /etc/profile.d, outranks
  # scripts/local.sh and would keep NCCL off loopback, the ranks' link here: lo is used instead, with a warning.
  if [[ "$NCCL_SOCKET_IFNAME" == ^* ]]; then
    warn W_CONFIG "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME is an NCCL exclusion list, not the ranks' netdev: using lo, their loopback link (set NCCL_SOCKET_IFNAME=<netdev> to choose one)."
    NCCL_SOCKET_IFNAME=lo
  fi
  [[ "$GPUS" =~ ^[0-9]+(,[0-9]+){1,3}$ ]] || fail E_CONFIG 'GPUS must list two to four GPU indices in rank order, such as 1,0.'
  IFS=, read -r -a GPU_IDS <<< "$GPUS"
  [[ "$(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l)" == "$NUM_GPUS" ]] || fail E_CONFIG 'GPUS contains duplicate GPU indices.'
  [[ "$CONTAINER_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail E_CONFIG 'CONTAINER_NAME contains unsupported characters.'
  [[ "$CONTAINER_NAME" != glm53-tf ]] || fail E_CONFIG 'glm53-tf is the Python recipe'"'"'s container; give this one another CONTAINER_NAME.'
  [[ -z "$CONTAINER_USER" || "$CONTAINER_USER" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*(:[a-zA-Z0-9_][a-zA-Z0-9_.-]*)?$ ]] || fail E_CONFIG 'CONTAINER_USER must be UID[:GID], a user name, root or empty.'
  [[ "$SERVED_NAME" =~ ^[^[:space:]]+$ ]] || fail E_CONFIG 'SERVED_NAME must be one word.'
  [[ "$HOST" =~ ^[A-Za-z0-9.:_-]+$ ]] || fail E_CONFIG 'HOST must be an address or host name.'
  [[ "$IMAGE" == '' || "$IMAGE" =~ ^[a-z0-9][a-z0-9._/:@-]*$ ]] || fail E_CONFIG 'IMAGE is not a valid image reference.'
  [[ "$TF_BIN" == /* ]] || fail E_CONFIG 'TF_BIN must be an absolute path inside the image.'
  for key in DATA_DIR MODEL_DIR DFLASH2_DIR CACHE_DIR STATE_DIR LOG_DIR; do
    path=${!key}
    [[ "$path" == /* && "$path" != *:* && "$path" != *$'\n'* && "$path" != *,* ]] || fail E_CONFIG "$key must be an absolute path without colons, commas or newlines."
  done
  [[ "$ABLIT" =~ ^[01]$ ]] || fail E_CONFIG 'ABLIT must be 0 or 1.'
  [[ "$ABLIT" == 0 || "$MODEL_SOURCE" == download ]] || fail E_CONFIG 'ABLIT=1 fetches the Ablit weights, so it needs MODEL_SOURCE=download. To serve a copy you already have, set ABLIT=0, MODEL_SOURCE=local and MODEL_DIR.'
  [[ "$MODEL_SOURCE" != download || -n "$MODEL_REVISION" ]] || fail E_CONFIG "MODEL_ID=$MODEL_ID has no pinned revision: set MODEL_REVISION."
  # A download never overwrites another checkpoint or revision: their files share names, so the result would be a mix.
  if [[ "$MODEL_SOURCE" == download && -f "$MODEL_DIR/.revision" && "$(cat "$MODEL_DIR/.revision")" != "$MODEL_REVISION" ]]; then
    fail E_CONFIG "MODEL_DIR=$MODEL_DIR holds revision $(cat "$MODEL_DIR/.revision"), not $MODEL_ID @ $MODEL_REVISION. Point MODEL_DIR at another directory (unset it to use the default for this checkpoint)."
  fi
  # A download never writes into a checkpoint it did not fetch (a local quant named by MODEL_DIR in local.sh). An
  # interrupted download leaves .downloading, so the next start resumes it.
  if [[ "$MODEL_SOURCE" == download && -e "$MODEL_DIR" && ! -f "$MODEL_DIR/.revision" && ! -f "$MODEL_DIR/.downloading" ]] && [[ -n "$(ls -A -- "$MODEL_DIR" 2>/dev/null)" ]]; then
    fail E_CONFIG "MODEL_SOURCE=download would write into MODEL_DIR=$MODEL_DIR, which holds files this recipe did not download. Point MODEL_DIR at an empty or recipe-made directory (default: $DATA_DIR/model)."
  fi
  if [[ "$DRAFTER" == none && ( -n "$MTP_DRAFTS" || -n "$MTP_CONFIDENCE" ) ]]; then
    warn W_CONFIG 'DRAFTER=none serves without drafts: MTP_DRAFTS and MTP_CONFIDENCE are not passed.'
  fi
  (( NUM_GPUS == 2 )) || warn W_CONFIG "This recipe is sized and checked for two ranks; $NUM_GPUS ranks start the same way (TP=3's uneven heads are unqualified upstream)."
}

# The image tag and its tf.patches label: docker/Dockerfile, the patches/engine/*.patch files (sorted by name; the
# build context holds nothing else) and the pins. Editing scripts or docs never forces a rebuild.
image_inputs() {
  (cd "$REPO_ROOT" && { [[ ! -f docker/Dockerfile ]] || printf 'docker/Dockerfile\0'
    find patches/engine -maxdepth 1 -type f -name '*.patch' -print0 2>/dev/null | LC_ALL=C sort -z; } | tr '\0' '\n')
}
set_image() {
  local f
  mapfile -t IMAGE_FILES < <(image_inputs)
  PATCH_COUNT=$(printf '%s\n' "${IMAGE_FILES[@]}" | grep -c '^patches/engine/' || true)
  PATCHES_HASH=$({
    printf 'TF_REPO=%s\nTF_REF=%s\nZIG_VERSION=%s\nZIG_SHA256=%s\nSM=%s\n' "$TF_REPO" "$TF_REF" "$ZIG_VERSION" "$ZIG_SHA256" "$SM"
    for f in "${IMAGE_FILES[@]}"; do [[ -n "$f" ]] || continue; printf '\0%s\0' "$f"; cat -- "$REPO_ROOT/$f"; done
    [[ "$WITH_VIDEO" != 1 ]] || printf '\0WITH_VIDEO=1\0'  # its own tag; the default's hash is as it was
  } | sha256sum | cut -c1-12)
  IMAGE=${IMAGE:-$IMAGE_REPO:$PATCHES_HASH}
}
image_ready() { [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)" == "$PATCHES_HASH" ]]; }

check_platform() {
  local cmd
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail E_PLATFORM 'This recipe requires Linux x86_64.'
  for cmd in docker curl python3 flock gzip tar sha256sum nvidia-smi; do
    command -v "$cmd" >/dev/null || fail E_DEPENDENCY "Missing command: $cmd. See README requirements."
  done
  docker info >/dev/null 2>&1 || fail E_DOCKER 'Cannot talk to Docker. Start the daemon and check socket permissions.'
  if (( EUID != 0 )) && ! id -nG | tr ' ' '\n' | grep -qx docker; then
    fail E_DOCKER_GROUP 'Your login is not in the docker group. Add it and start a new login session.'
  fi
  docker info --format '{{json .Runtimes}}' | python3 -c 'import json,sys; sys.exit(0 if "nvidia" in json.load(sys.stdin) else 1)' \
    || fail E_NVIDIA_RUNTIME 'Docker has no NVIDIA runtime. Install and configure NVIDIA Container Toolkit.'
  log '[I_PLATFORM] Linux x86_64, Docker access and NVIDIA runtime are ready.'
}

# Each selected GPU needs MIN_GPU_FREE_MIB free and less than MAX_GPU_USED_MIB in use. A desktop's own programs
# (the compositor, a browser) register as small compute processes on the display GPU: only larger ones count as busy.
check_gpus() {
  local gpu row free used name processes
  for gpu in "${GPU_IDS[@]}"; do
    row=$(nvidia-smi -i "$gpu" --query-gpu=name,memory.free,memory.used --format=csv,noheader,nounits) || fail E_GPU "Cannot query GPU $gpu. Check the driver and the GPUS selection."
    IFS=, read -r name free used <<< "$row"
    free=${free//[[:space:]]/}; used=${used//[[:space:]]/}
    [[ "$free" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || fail E_GPU "GPU $gpu returned invalid memory information."
    (( free >= MIN_GPU_FREE_MIB )) || fail E_GPU_MEMORY "GPU $gpu ($name) has $free MiB free; at least $MIN_GPU_FREE_MIB MiB is required. Stop other GPU workloads (another TensorFold server?)."
    (( used < MAX_GPU_USED_MIB )) || fail E_GPU_BUSY "GPU $gpu uses $used MiB (MAX_GPU_USED_MIB=$MAX_GPU_USED_MIB). Stop other GPU workloads before launch."
    log "[I_GPU] GPU $gpu:$name, $free MiB free, $used MiB in use."
  done
  processes=$(nvidia-smi -i "$GPUS" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits) || fail E_GPU 'Cannot list GPU compute processes.'
  processes=$(awk -F', *' -v max="$MAX_GPU_USED_MIB" '$2 ~ /^[0-9]+$/ && $2 + 0 >= max {print $1}' <<< "$processes")
  if [[ "$processes" =~ [0-9] ]]; then fail E_GPU_BUSY "A selected GPU has a compute process (PID ${processes//$'\n'/, }) of $MAX_GPU_USED_MIB MiB or more. Stop it before launch."; fi
}

check_host_memory() {
  local kib
  kib=$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo 2>/dev/null || true)
  [[ "$kib" =~ ^[0-9]+$ ]] || { warn W_RAM 'Cannot read MemAvailable.'; return 0; }
  if (( kib / 1048576 < MIN_HOST_FREE_GIB )); then
    warn W_RAM "Only $((kib / 1048576)) GiB of host memory is available (MIN_HOST_FREE_GIB=$MIN_HOST_FREE_GIB): the ranks stage every layer through host memory while they load."
  else
    log "[I_RAM] $((kib / 1048576)) GiB of host memory available."
  fi
}

check_disk() {
  local path ancestor device free need label docker_root i
  local -a paths=() sizes=() labels=()
  if [[ "$MODEL_SOURCE" == download ]] && ! weights_ready "$MODEL_DIR" "$MODEL_REVISION"; then
    need=${MODEL_FREE_GIB%.*}
    # Resuming: what an interrupted download already holds doesn't have to fit again.
    if [[ -f "$MODEL_DIR/.downloading" ]]; then
      need=$(( need - $(du -s --block-size=1G -- "$MODEL_DIR" | cut -f1) )); (( need > 0 )) || need=1
    fi
    paths+=("$MODEL_DIR"); sizes+=("$need"); labels+=(checkpoint)
  fi
  if [[ "$DRAFTER" == dflash2 ]] && ! drafter_ready; then paths+=("$DFLASH2_DIR"); sizes+=("$DFLASH2_FREE_GIB"); labels+=(DFlash2); fi
  if ! image_ready; then
    docker_root=$(docker info --format '{{.DockerRootDir}}') || fail E_DISK 'Cannot locate Docker storage.'
    paths+=("$docker_root"); sizes+=("$IMAGE_FREE_GIB"); labels+=(image)
  fi
  local -A need_by_device=() free_by_device=() names_by_device=()
  for (( i=0; i<${#paths[@]}; i++ )); do
    path=${paths[i]}; need=${sizes[i]}; label=${labels[i]}; ancestor=$path
    while [[ ! -e "$ancestor" ]]; do ancestor=$(dirname -- "$ancestor"); done
    read -r device free < <(df -Pk -- "$ancestor" | awk 'END {print $1, $4}')
    [[ "$free" =~ ^[0-9]+$ ]] || fail E_DISK "Cannot inspect available disk space for $label."
    need_by_device[$device]=$(( ${need_by_device[$device]:-0} + need * 1024 * 1024 ))
    free_by_device[$device]=$free
    names_by_device[$device]="${names_by_device[$device]:-} $label"
  done
  for device in "${!need_by_device[@]}"; do
    (( free_by_device[$device] >= need_by_device[$device] )) || fail E_DISK "Need $((need_by_device[$device]/1024/1024)) GiB free for${names_by_device[$device]}; only $((free_by_device[$device]/1024/1024)) GiB available on that filesystem."
  done
  log '[I_DISK] Enough disk space for what is still missing.'
}

prepare_image() {
  local -a build_flags=()
  read -r -a build_flags <<< "$DOCKER_BUILD_FLAGS"
  if [[ ! -f "$REPO_ROOT/docker/Dockerfile" ]]; then
    [[ "$DRY_RUN" == 1 ]] || fail E_IMAGE 'docker/Dockerfile is missing: the image cannot be built.'
    warn W_DOCKERFILE 'docker/Dockerfile is missing; the plan shows the build it would run.'
  fi
  if [[ "$DRY_RUN" != 1 ]]; then
    if image_ready; then log "Image is ready: $IMAGE (tf.patches=$PATCHES_HASH)"; return; fi
    local other
    other=$(docker images -q --filter "label=tf.patches=$PATCHES_HASH" | head -n1)
    if [[ -n "$other" ]]; then
      docker tag "$other" "$IMAGE" || fail E_IMAGE "Cannot tag image $other as $IMAGE."
      log "Image is ready: $other tagged $IMAGE (tf.patches=$PATCHES_HASH)"; return
    fi
  else
    log "Skipped when an image labelled tf.patches=$PATCHES_HASH exists (docker image inspect $IMAGE):"
  fi
  log "Build $IMAGE: TensorFold ${TF_REF:0:12} with $PATCH_COUNT engine patches, Zig $ZIG_VERSION, sm_$SM."
  run docker build -f "$REPO_ROOT/docker/Dockerfile" -t "$IMAGE" --label "tf.patches=$PATCHES_HASH" \
    --build-arg "TF_REPO=$TF_REPO" --build-arg "TF_REF=$TF_REF" --build-arg "ZIG_VERSION=$ZIG_VERSION" \
    --build-arg "ZIG_SHA256=$ZIG_SHA256" --build-arg "SM=$SM" --build-arg "PATCHES_HASH=$PATCHES_HASH" \
    --build-arg "WITH_VIDEO=$WITH_VIDEO" "${build_flags[@]}" "$REPO_ROOT" || fail E_IMAGE 'Image build failed. Check disk space, network access and the first build error.'
  run docker run --rm --network none --entrypoint "$TF_BIN" "$IMAGE" --version \
    || fail E_IMAGE "The built image has no working $TF_BIN."
}

# --- weights -------------------------------------------------------------------------------------------------------
weights_ready() { [[ -f "$1/config.json" && -f "$1/.revision" && "$(cat "$1/.revision")" == "$2" ]]; }
drafter_ready() { [[ -f "$DFLASH2_DIR/config.json" ]] && compgen -G "$DFLASH2_DIR/*.safetensors" >/dev/null; }

# Reads MODEL_DIR's config.json (read-only, also in a dry run): MODEL_DESC, and NEEDS_OVERLAY=1 when an EXL3
# quantization_config names no bits (a mixed-width quant such as the 3.5 bpw GLMQ one).
inspect_model() {
  NEEDS_OVERLAY=0; MODEL_DESC="$MODEL_DIR"
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    [[ "$MODEL_SOURCE" == download ]] && { MODEL_DESC="$MODEL_ID@${MODEL_REVISION:0:12} (to download)"; return 0; }
    local why="MODEL_DIR has no config.json: $MODEL_DIR. Set MODEL_DIR to a GLM-5.3-Flash EXL3 checkpoint, or MODEL_SOURCE=download."
    [[ "$DRY_RUN" == 1 ]] || fail E_MODEL "$why"
    warn E_MODEL "$why"; return 0
  fi
  local out
  out=$(python3 - "$MODEL_DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
c = json.load(open(os.path.join(d, "config.json")))
mt = c.get("model_type")
if mt != "glm5_next":
    print(f"E_MODEL\tmodel_type {mt!r}: this recipe serves GLM-5.3-Flash (glm5_next)"); sys.exit()
q = c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config") or {}
if not glob.glob(os.path.join(d, "*.safetensors")):
    print("E_MODEL\tno *.safetensors files"); sys.exit()
if not os.path.exists(os.path.join(d, "tokenizer.json")):
    print("E_MODEL\tno tokenizer.json"); sys.exit()
method = q.get("quant_method", "none")
if method != "exl3":
    print(f"W_MODEL\t{method} checkpoint: the recipe is written for EXL3; served as stored\t0"); sys.exit()
bits = q.get("bits"); avg = q.get("avg_bits")
width = f"{bits}-bit" if bits is not None else (f"mixed {avg} bpw" if avg is not None else "unknown width")
vision = "vision" if c.get("vision_config") else "no vision tower"
print(f"OK\tEXL3 {width} experts, {q.get('codebook', '?')} codebook, {vision}\t{1 if bits is None else 0}")
PY
  ) || fail E_MODEL "Cannot read $MODEL_DIR/config.json."
  local kind desc overlay
  IFS=$'\t' read -r kind desc overlay <<< "$out"
  case "$kind" in
    OK) MODEL_DESC="$(basename -- "$MODEL_DIR"): $desc" ;;
    W_MODEL) warn W_MODEL "$desc"; MODEL_DESC="$(basename -- "$MODEL_DIR")" ;;
    *) fail E_MODEL "$MODEL_DIR: $desc." ;;
  esac
  [[ "$CONFIG_OVERLAY" == auto && "${overlay:-0}" == 1 ]] && NEEDS_OVERLAY=1
  return 0
}

# The overlay: config.json with quantization_config.bits 4 added, mounted read-only over /model/config.json.
# The engine reads each expert matrix's width from its trellis shape (with the engine patches); only the family's
# admission check reads bits. The checkpoint itself is never changed.
model_overlay() {
  OVERLAY_MOUNT=()
  (( NEEDS_OVERLAY == 1 )) || return 0
  local overlay="$STATE_DIR/model-config.json"
  if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$STATE_DIR"
    python3 - "$MODEL_DIR/config.json" "$overlay" <<'PY' || fail E_MODEL 'Cannot write the config.json overlay.'
import json, os, sys
c = json.load(open(sys.argv[1]))
q = c.get("quantization_config")
if q is None:
    q = c["text_config"]["quantization_config"]
q["bits"] = 4
text = json.dumps(c, indent=2) + "\n"
old = open(sys.argv[2]).read() if os.path.exists(sys.argv[2]) else None
if old != text:  # rewritten only when it changes: the engine's digest cache keeps the file's identity
    tmp = sys.argv[2] + ".tmp"
    open(tmp, "w").write(text)
    os.replace(tmp, sys.argv[2])
PY
  fi
  log "config.json overlay: quantization_config.bits 4 added for the engine's EXL3 check ($overlay -> /model/config.json, read-only)."
  OVERLAY_MOUNT=(-v "$overlay:/model/config.json:ro")
}

# Downloads use the host's hf CLI (huggingface_hub). HF_TOKEN, when set, is read by hf from the environment and never printed.
fetch_weights() {
  local repo=$1 rev=$2 dir=$3
  if weights_ready "$dir" "$rev"; then log "Weights are ready: $repo @ ${rev:0:12} ($dir)"; return; fi
  [[ "$DRY_RUN" == 1 ]] || command -v hf >/dev/null || fail E_DEPENDENCY 'Missing command: hf (python3 -m pip install -U "huggingface_hub[cli]").'
  [[ "$DRY_RUN" == 1 ]] || { mkdir -p "$dir"; printf '%s\n' "$rev" > "$dir/.downloading"; }
  log "Download $repo at pinned revision $rev into $dir."
  run hf download "$repo" --revision "$rev" --local-dir "$dir" || fail E_DOWNLOAD 'Download failed. Check free space, access and HF_TOKEN; rerun to resume.'
  [[ "$DRY_RUN" == 1 ]] || { printf '%s\n' "$rev" > "$dir/.revision"; rm -f "$dir/.downloading"; }
}

# The gated Ablit weights need HF_TOKEN only while they still have to be fetched. One small file is asked for before the
# image work, so a refusal says why at once. The token goes to curl on stdin, never on a command line.
check_weights_access() {
  [[ "$MODEL_SOURCE" == download && "$MODEL_ID" == "$ABLIT_ID" ]] || return 0
  ! weights_ready "$MODEL_DIR" "$MODEL_REVISION" || return 0
  local steps code
  steps="1. Open https://huggingface.co/$MODEL_ID, log in and agree to its terms. 2. Create a token with read access at https://huggingface.co/settings/tokens (the same account). 3. Set HF_TOKEN=hf_... in scripts/local.sh, .env or the environment, and run this again. Or set ABLIT=0 to serve the published checkpoint."
  if [[ -z "${HF_TOKEN:-}" ]]; then
    [[ "$DRY_RUN" == 1 ]] && { warn W_HF_TOKEN "The Ablit weights are gated and HF_TOKEN is not set: a real start stops here."; return 0; }
    fail E_HF_TOKEN "The Ablit weights ($MODEL_ID) are gated on Hugging Face. $steps"
  fi
  [[ "$DRY_RUN" != 1 ]] || return 0
  code=$(printf 'Authorization: Bearer %s\n' "$HF_TOKEN" |
    curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H @- "https://huggingface.co/$MODEL_ID/resolve/$MODEL_REVISION/config.json" || true)
  case "$code" in
    2??|3??) log "HF_TOKEN reaches the gated $MODEL_ID." ;;
    401) fail E_HF_TOKEN "Hugging Face does not accept HF_TOKEN (401). $steps" ;;
    403) fail E_HF_TOKEN "HF_TOKEN may not read $MODEL_ID yet (403): its account has not agreed to the terms, or a fine-grained token lacks read access to public gated repositories. $steps" ;;
    404) fail E_HF_TOKEN "Hugging Face has no $MODEL_ID @ $MODEL_REVISION (404). Check MODEL_REVISION." ;;
    *) warn W_HF_TOKEN "Could not check HF_TOKEN against $MODEL_ID (HTTP $code); trying the download." ;;
  esac
}

prepare_weights() {
  if [[ "$MODEL_SOURCE" == download ]]; then
    fetch_weights "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR"
    [[ "$DRY_RUN" == 1 ]] || inspect_model
  else
    log "Checkpoint: $MODEL_DESC (local, mounted read-only; no download)."
  fi
  if [[ "$DRAFTER" == dflash2 ]]; then
    log 'DFlash2 drafter weights: CC BY-NC-ND 4.0, non-commercial. DRAFTER=mtp or DRAFTER=none serves without them.'
    if drafter_ready; then
      log "DFlash2 is ready: $DFLASH2_DIR"
      if [[ -f "$DFLASH2_DIR/.revision" && "$(cat "$DFLASH2_DIR/.revision")" != "$DFLASH2_REVISION" ]]; then
        warn W_DFLASH2 "$DFLASH2_DIR holds revision $(cat "$DFLASH2_DIR/.revision"), not the pinned $DFLASH2_REVISION."
      fi
    else
      fetch_weights "$DFLASH2_ID" "$DFLASH2_REVISION" "$DFLASH2_DIR"
    fi
  fi
}

# NVML's view of the selected GPUs' peer-to-peer status, from the engine's own `capabilities --json` (no model load).
# Not OK is a warning: NCCL then falls back to shared memory, which works but slows every exchange.
check_p2p() {
  if [[ "$DRY_RUN" == 1 ]]; then
    plan docker run --rm --gpus "\"device=$GPUS\"" --network none --entrypoint "$TF_BIN" "$IMAGE" capabilities --json
    log '  Warn unless NVML reports peer-to-peer read and write OK between the selected GPUs.'
    return 0
  fi
  local caps
  caps=$(docker run --rm --gpus "\"device=$GPUS\"" --network none --entrypoint "$TF_BIN" "$IMAGE" capabilities --json 2>/dev/null) \
    || { warn W_P2P 'Could not read the engine capabilities; the ranks report their links in their logs (p2p: lines).'; return 0; }
  local kind text
  IFS=$'\t' read -r kind text < <(python3 - "$caps" <<'PY'
import json, sys
try:
    p = json.loads(sys.argv[1]).get("p2p")
except ValueError:
    p = None
if not p or not p.get("pairs"):
    print("W_P2P\tno NVML peer-to-peer summary"); sys.exit()
pairs = [(q["gpus"], q.get("read"), q.get("write"), q.get("path")) for q in p["pairs"]]
bad = [f"GPUs {g[0]}-{g[1]} read {r} write {w} ({path})" for g, r, w, path in pairs if r != "OK" or w != "OK"]
if bad:
    print("W_P2P\t" + "; ".join(bad))
else:
    print("OK\t" + ", ".join(f"GPUs {g[0]}-{g[1]} over {path}" for g, _, _, path in pairs))
PY
  ) || true
  if [[ "$kind" == OK ]]; then log "[I_P2P] NVML reports peer-to-peer read and write OK: $text."
  else warn W_P2P "${text:-unreadable capabilities}. NCCL falls back to slower paths; check IOMMU/ACS settings."; fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cd "$REPO_ROOT"
  mode=${1:-}
  case "$mode" in
    --label) set_image; printf '%s\n' "$PATCHES_HASH"; exit 0 ;;
    ''|--image|--weights|--check) ;;
    -h|--help) sed -n '2,7s/^# //p' "$0"; exit 0 ;;
    *) printf 'Usage: scripts/prepare.sh [--image|--weights|--check|--label]\n' >&2; exit 2 ;;
  esac
  validate_config
  set_image
  [[ "$mode" == --image ]] || inspect_model
  if [[ "$DRY_RUN" == 1 ]]; then log '[I_DRY_RUN] Plan only: no Docker, hardware access, downloads or file changes.'
  else
    check_platform
    mkdir -p "$STATE_DIR"
    exec 8>"$STATE_DIR/start.lock"
    flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
    [[ "$mode" == --check ]] || check_disk
  fi
  case "$mode" in
    --image) prepare_image ;;
    --weights) check_weights_access; prepare_weights ;;
    --check) [[ "$DRY_RUN" == 1 ]] || { check_gpus; check_host_memory; image_ready || fail E_IMAGE "Build the image first: scripts/prepare.sh --image ($IMAGE)."; }; check_p2p ;;
    *) check_weights_access; prepare_image; prepare_weights ;;
  esac
  log 'Prepared. Start the API with ./start.sh.'
fi
