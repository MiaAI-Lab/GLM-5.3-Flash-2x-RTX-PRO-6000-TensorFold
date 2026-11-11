#!/usr/bin/env bash
# One command: check, build the engine image, check the weights, start every rank in one container,
# wait for the API and smoke-test it.
# Usage: ./start.sh [restart]
# Configuration: scripts/config.sh defaults < .env < scripts/local.sh < exported environment.
# DRY_RUN=1 ./start.sh prints the full plan without Docker, GPUs, downloads or file changes.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/prepare.sh
source ./scripts/prepare.sh
# shellcheck source=scripts/banner.sh
source ./scripts/banner.sh
case "${1:-}" in
  '') MODE=start ;;
  restart) MODE=restart ;;
  help|-h|--help) sed -n '2,7s/^# //p' "$0"; exit 0 ;;
  *) fail E_CONFIG 'Usage: ./start.sh [restart]. Set serving options in the environment or scripts/local.sh.' ;;
esac
(( $# <= 1 )) || fail E_CONFIG 'Usage: ./start.sh [restart].'
validate_config
set_image
inspect_model

step() { printf '\n[%s/6] %s\n' "$1" "$2"; }
bits() { case "$1" in bf16) echo 16 ;; q8) echo 8 ;; q6) echo 6 ;; q4) echo 4 ;; esac; }
API_HOST="$HOST"
[[ "$HOST" != 0.0.0.0 && "$HOST" != :: ]] || API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

# Rank r's GPU as the container numbers it: the selected GPUs in PCI order (CUDA_DEVICE_ORDER=PCI_BUS_ID, the order
# nvidia-smi's indices follow). GPUS=0,1 on this two-GPU host: rank 0 -> container GPU 0 (host 0), rank 1 -> 1 (host 1).
RANK_DEVICES=$(for gpu in "${GPU_IDS[@]}"; do printf '%s\n' "${GPU_IDS[@]}" | sort -n | grep -nx -- "$gpu" | cut -d: -f1; done \
  | awk '{ $1 = $1 - 1 } 1' | paste -sd,)

# --- the context window ----------------------------------------------------------------------------------------------
FIT_FILE="$STATE_DIR/context-fit.tsv"
fit_key() {
  printf '%s|' "$PATCHES_HASH" "$MODEL_DIR" "$GPUS" "$PARALLEL" "$KV" "$KV_PAGES" "$DENSE" "$HEAD" "$VISION" "$VISION_MAX_IMAGES" "$DRAFTER" "$DRAFTER_BITS" \
    "$MEMORY_RESERVE_GIB" "$PROMPT_CACHE_GIB" "$LEARN_GIB" "$KV_SHARD" "$CONTEXT_MAX" "$SERVE_EXTRA_ARGS" | sha256sum | cut -c1-16
}
remembered_context() {
  [[ -f "$FIT_FILE" ]] || return 0
  awk -F'\t' -v k="$1" '$1 == k { v = $2 } END { if (v != "") print v }' "$FIT_FILE"
}
remember_context() {
  local tmp="$FIT_FILE.tmp"
  mkdir -p "$STATE_DIR"
  { if [[ -f "$FIT_FILE" ]]; then awk -F'\t' -v k="$1" '$1 != k' "$FIT_FILE"; fi
    printf '%s\t%s\t%s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$tmp"
  mv -- "$tmp" "$FIT_FILE"
}
# A smaller window the engine's memory refusals in the ranks' logs allow (nothing when they show none). The native
# engine states the bytes, not the window: the streams' caches at the asked window and the room left on that rank. Every
# rank's figures count and the tightest wins (rank 1 shares GPU 1 with the desktop). A stream has a fixed part (KDA
# state, rings: CONTEXT_FIT_FIXED_GIB) besides its per-token caches; CONTEXT_FIT_MARGIN leaves room for the rest.
fit_context() {
  python3 - "$1" "$2" "$CONTEXT_FIT_MARGIN" "$CONTEXT" "${CONTEXT_FIT_FIXED_GIB:-0.4}" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
asked, margin, auto, fixed = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "auto", float(sys.argv[5])
num = r"([0-9]+(?:\.[0-9]+)?)"
w = None
named = re.findall(r"largest fitting prompt-plus-reply window: ([0-9]+) tokens", text)
if named:
    w = min(int(x) for x in named)
if w is None:  # windows a stream (--kv-cache fp8): the engine's own stream is in the model's bytes, so N share room + one
    m = re.findall(r"--parallel ([0-9]+) needs " + num + r" GiB for its streams at a ([0-9]+)-token window, "
                   r"and the CUDA memory budget leaves " + num + " GiB", text)
    fits = []
    for g in m:
        n, x, w0, r = int(g[0]), float(g[1]), int(g[2]), float(g[3])
        s = x / max(n - 1, 1)                    # one stream at w0
        f = min(fixed, s / 2)                    # its fixed part
        if s > f > 0 or s > 0:
            per_token = (s - f) / w0             # GiB a token
            fits.append(((r + s) / n - f) / per_token * margin)
    if fits:
        w = min(fits)
if w is None:  # shared KV pages (bf16, or fp8 with KV_PAGES=on): the engine's own window first, the other streams' fixed parts after
    m = re.findall(r"one at a ([0-9]+)-token window takes " + num + r" GiB of KV pages, and the budget leaves " + num + " GiB", text)
    if m:
        w0, p, r = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
        if p > 0:
            w = w0 * r / p * margin * 0.9
if w is None:
    m = re.findall(r"one at a ([0-9]+)-token window takes " + num + r" GiB, and " + num + " GiB is left after", text)
    if m:
        w0, s, r = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
        if s > 0:
            w = w0 * r / s * margin
if w is None and auto and re.search(r"\(OutOfMemory\)|CUDA_ERROR_OUT_OF_MEMORY", text):
    w = asked / 2  # the engine's own window did not allocate: no figures to scale by
if w is not None:
    w = int(w) // 1024 * 1024
    if 8192 <= w < asked:
        print(w)
PY
}
FIT_KEY=$(fit_key)
if [[ "$CONTEXT" == auto ]]; then
  CTX=$(remembered_context "$FIT_KEY")
  if [[ -n "$CTX" ]]; then CTX_NOTE="auto: the window that served these settings last time ($FIT_FILE)"
  else CTX=$CONTEXT_MAX; CTX_NOTE="auto: try $CONTEXT_MAX, then once the largest window the engine's memory refusal allows"; fi
else
  CTX=$CONTEXT; CTX_NOTE=explicit
fi

# --- what every rank is given --------------------------------------------------------------------------------------
case "$CHAT_TEMPLATE" in
  auto) TEMPLATE_FILE=; [[ "$VISION" != 1 ]] || TEMPLATE_FILE="$PWD/templates/glm-5.3-flash-vision.jinja" ;;
  none) TEMPLATE_FILE= ;;
  /*) TEMPLATE_FILE=$CHAT_TEMPLATE ;;
  *) fail E_CONFIG 'CHAT_TEMPLATE must be auto, none or an absolute file path.' ;;
esac
[[ -z "$TEMPLATE_FILE" || -f "$TEMPLATE_FILE" ]] || fail E_CONFIG "Chat template not found: $TEMPLATE_FILE"
# EFFORT_LINE=thinking: the served template is the chosen one (the checkpoint's own with VISION=0) with its effort line
# set from reasoning_effort, which the engine passes with thinking on only (null with thinking off: prompt.zig), so a
# thinking-off prompt no longer carries `<|system|>Reasoning Effort: Max`. Written to STATE_DIR every start; the base
# line must be the one the checkpoint ships, else the start stops rather than serve a template it did not understand.
# With VISION=0 the base is the checkpoint's own file, which the default MODEL_SOURCE=download fetches in step 3: until
# that download is complete (weights_ready), step 3 writes the template after it (EFFORT_AFTER_DOWNLOAD). Every other
# start checks and writes it here, before anything runs or a serving pair is stopped.
effort_template() {
  if [[ "$DRY_RUN" == 1 && "$EFFORT_AFTER_DOWNLOAD" == 1 && ! -f "$EFFORT_BASE" ]]; then
    log "Then write $STATE_DIR/chat_template.jinja: the downloaded $EFFORT_BASE with its effort line taken from reasoning_effort (EFFORT_LINE=thinking)."
    return 0
  fi
  [[ -f "$EFFORT_BASE" ]] || fail E_CONFIG "Chat template not found: $EFFORT_BASE (EFFORT_LINE=thinking rewrites its effort line)."
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$STATE_DIR"
  python3 - "$EFFORT_BASE" "$STATE_DIR/chat_template.jinja" "$DRY_RUN" <<'PY' || fail E_CONFIG "EFFORT_LINE=thinking: $EFFORT_BASE has no effort line this recipe knows; set EFFORT_LINE=always or CHAT_TEMPLATE to a file."
import sys
base = "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}"
line = "{%- set effective_reasoning_effort = (reasoning_effort if reasoning_effort in ['low', 'high'] else 'max') if (reasoning_effort is defined and reasoning_effort is not none) else none -%}"
src = open(sys.argv[1], encoding="utf-8").read()
if src.count(base) != 1:
    sys.exit(1)
if sys.argv[3] != "1":  # DRY_RUN=1 checks the base and writes nothing
    open(sys.argv[2], "w", encoding="utf-8").write(src.replace(base, line))
PY
}
EFFORT_AFTER_DOWNLOAD=0
if [[ "$CHAT_TEMPLATE" == auto && "$EFFORT_LINE" == thinking ]]; then
  EFFORT_BASE=${TEMPLATE_FILE:-$MODEL_DIR/chat_template.jinja}
  if [[ -z "$TEMPLATE_FILE" && "$MODEL_SOURCE" == download ]] && ! weights_ready "$MODEL_DIR" "$MODEL_REVISION"; then
    EFFORT_AFTER_DOWNLOAD=1
  else
    effort_template
  fi
  TEMPLATE_FILE="$STATE_DIR/chat_template.jinja"
fi

build_serve_args() {
  local -a extra=()
  SERVE_ARGS=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT" --context "$1" --parallel "$PARALLEL"
    --max-tokens "$MAX_TOKENS" --kv-cache "$KV" --no-update-check)
  if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
  [[ -z "$REASONING_EFFORT" ]] || SERVE_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
  [[ "$VISION" != 1 ]] || SERVE_ARGS+=(--vision --vision-max-images "$VISION_MAX_IMAGES")
  [[ -z "$TEMPLATE_FILE" ]] || SERVE_ARGS+=(--chat-template /templates/chat_template.jinja)
  case "$DRAFTER" in
    dflash2) SERVE_ARGS+=(--drafter /dflash2 --drafter-bits "$DRAFTER_BITS") ;;
    none) SERVE_ARGS+=(--no-drafts) ;;
  esac
  if [[ "$DRAFTER" != none ]]; then
    [[ -z "$MTP_DRAFTS" ]] || SERVE_ARGS+=(--mtp-drafts "$MTP_DRAFTS")
    [[ -z "$MTP_CONFIDENCE" ]] || SERVE_ARGS+=(--mtp-confidence "$MTP_CONFIDENCE")
  fi
  [[ -z "$PROMPT_CACHE_GIB" ]] || SERVE_ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
  # LEARN_GIB: engine patch 0103 keeps the prompt states on disk too (CACHE_DIR/learn holds every rank's part)
  [[ "$LEARN_GIB" == 0 ]] || SERVE_ARGS+=(--learn-conversations --learn-dir /state/learn --learn-gib "$LEARN_GIB"
    --learn-min-free-gib "$LEARN_MIN_FREE_GIB")
  read -r -a extra <<< "$SERVE_EXTRA_ARGS"
  SERVE_ARGS+=("${extra[@]}")
}

# Recipe-managed names; any other exported TF_*, TENSORFOLD_* or NCCL_* variable reaches the ranks by name (values,
# TENSORFOLD_API_KEY's included, are never printed). The per-rank ones are set by the runner.
MANAGED_ENV=' HOME HF_HUB_OFFLINE CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES TENSORFOLD_NO_UPDATE_CHECK TENSORFOLD_NO_LIVE
 TENSORFOLD_MEMORY_RESERVE_GIB TF_GLM_DENSE_BITS TF_GLM_HEAD_BITS TF_GLM_CLEAR_THINKING TF_GLM_RANK TF_GLM_WORLD TF_CUDA_DEVICE
 TF_KV_PAGES TF_GLM_CE_EXCHANGE TF_GLM_ABSORB_MMA TF_GLM_W8A8 TENSORFOLD_PREFILL_SHARE TENSORFOLD_PREFILL_BUDGET TF_GLM_KV_SHARD TF_GLM_L2PF TF_EXTRA_EOS
 TF_BIN TF_REPO TF_REF NCCL_COMM_ID NCCL_P2P_LEVEL NCCL_SOCKET_IFNAME NCCL_IB_DISABLE RANK_DEVICES '
build_env_args() {
  local key
  ENV_ARGS=(-e HOME=/state/home -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_NO_LIVE=1
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB"
    -e "TF_GLM_DENSE_BITS=$(bits "$DENSE")" -e "TF_GLM_HEAD_BITS=$(bits "$HEAD")" -e "TF_GLM_CLEAR_THINKING=$CLEAR_THINKING"
    -e "TF_KV_PAGES=$KV_PAGES" -e "TF_GLM_CE_EXCHANGE=$CE_EXCHANGE" -e "TF_GLM_ABSORB_MMA=$ABSORB_MMA" -e "TF_GLM_W8A8=$W8A8"
    -e "NCCL_COMM_ID=127.0.0.1:$NCCL_PORT" -e "NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL" -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" -e "TF_BIN=$TF_BIN" -e "RANK_DEVICES=$RANK_DEVICES"
    -e "TENSORFOLD_PREFILL_SHARE=$PREFILL_SHARE")
  [[ -z "$PREFILL_BUDGET" ]] || ENV_ARGS+=(-e "TENSORFOLD_PREFILL_BUDGET=$PREFILL_BUDGET")
  # KV_SHARD=1: engine patch 0104 keeps each stream's latent rows on its owner GPU (every rank reads the same value)
  [[ "$KV_SHARD" != 1 ]] || ENV_ARGS+=(-e "TF_GLM_KV_SHARD=1")
  # L2PF=1: engine patch 0105 prefetches each decode window's next weights into L2 (every rank must agree)
  [[ "$L2PF" != 1 ]] || ENV_ARGS+=(-e "TF_GLM_L2PF=1")
  # EXTRA_EOS: engine patch 0108 ends a reply at these tokens too (every rank alike)
  [[ "$EXTRA_EOS" == none ]] || ENV_ARGS+=(-e "TF_EXTRA_EOS=$EXTRA_EOS")
  while IFS= read -r key; do
    [[ "$MANAGED_ENV" != *" $key "* ]] || continue
    case "$key" in TF_*|TENSORFOLD_*|NCCL_*) ENV_ARGS+=(-e "$key") ;; esac
  done < <(compgen -e | LC_ALL=C sort)
}

# The ranks' command lines as the runner starts them (rank 1 first; it retries its line to rank 0 for 300 s).
show_ranks() {
  local r d vis role
  local -a devs
  IFS=, read -r -a devs <<< "$RANK_DEVICES"
  for (( r = NUM_GPUS - 1; r >= 0; r-- )); do
    vis=${devs[r]}
    for d in "${devs[@]}"; do [[ "$d" == "${devs[r]}" ]] || vis+=",$d"; done
    role=''; (( r != 0 )) || role=' (HTTP server, vision tower, follows no one)'
    printf '  rank %s on host GPU %s%s:\n    ' "$r" "${GPU_IDS[r]}" "$role"
    shq env "CUDA_VISIBLE_DEVICES=$vis" TF_CUDA_DEVICE=0 "TF_GLM_RANK=$r" "TF_GLM_WORLD=$NUM_GPUS" \
      "NCCL_COMM_ID=127.0.0.1:$NCCL_PORT" "$TF_BIN" serve /model "${SERVE_ARGS[@]}"
    if (( r == 0 )); then printf '2>&1 | tee -a /logs/rank0.log\n'; else printf '> /logs/rank%s.log 2>&1\n' "$r"; fi
  done
}

write_runner() {
  cat > "$RUN_LOG_DIR/runner.sh" <<'RUNNER'
#!/usr/bin/env bash
# Written by start.sh: every GLM-5.3-Flash rank in this one container. Rank r opens container GPU RANK_DEVICES[r] as
# its CUDA device 0 (CUDA_VISIBLE_DEVICES lists it first, the other ranks' GPUs after it for NCCL peer-to-peer and
# CUDA IPC); TF_GLM_RANK of TF_GLM_WORLD places it, NCCL_COMM_ID (from the container) names rank 0's address.
# Rank 0 serves HTTP; the others follow its steps and never bind a port. Every rank gets the same flags ("$@").
set -u
IFS=, read -r -a devs <<< "$RANK_DEVICES"
world=${#devs[@]}
visible() { local d out=${devs[$1]}; for d in "${devs[@]}"; do [[ "$d" == "${devs[$1]}" ]] || out+=",$d"; done; printf '%s' "$out"; }
declare -a pids=()
alive() { local p; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
stop_ranks() {
  local i
  # rank 0 first: it drains its requests (TF_DRAIN_S, 20 s) and ends the followers' steps; then whatever is left
  kill -TERM "${pids[0]}" 2>/dev/null || true
  for (( i = 0; i < 300; i++ )); do kill -0 "${pids[0]}" 2>/dev/null || break; sleep 0.1; done
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for (( i = 0; i < 100; i++ )); do alive || break; sleep 0.1; done
  kill -KILL "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap 'printf "[runner] stop signal: stopping the ranks\n" | tee -a /logs/rank0.log; stop_ranks; exit 143' TERM INT
printf '[runner] %s ranks; rank r on container GPU %s[r]; %s\n' "$world" "$RANK_DEVICES" "$("$TF_BIN" --version 2>&1)" | tee -a /logs/rank0.log
for (( rank = world - 1; rank >= 1; rank-- )); do
  env CUDA_VISIBLE_DEVICES="$(visible "$rank")" TF_CUDA_DEVICE=0 TF_GLM_RANK="$rank" TF_GLM_WORLD="$world" \
    "$TF_BIN" serve /model "$@" > "/logs/rank$rank.log" 2>&1 &
  pids[rank]=$!
done
env CUDA_VISIBLE_DEVICES="$(visible 0)" TF_CUDA_DEVICE=0 TF_GLM_RANK=0 TF_GLM_WORLD="$world" \
  "$TF_BIN" serve /model "$@" > >(tee -a /logs/rank0.log) 2>&1 &
pids[0]=$!
wait -n "${pids[@]}"
code=$?
gone=''
for r in "${!pids[@]}"; do kill -0 "${pids[r]}" 2>/dev/null || gone+=" $r"; done
printf '[runner] rank%s exited (code %s); stopping the other ranks\n' "$gone" "$code" | tee -a /logs/rank0.log
stop_ranks
exit "$code"
RUNNER
  chmod +x "$RUN_LOG_DIR/runner.sh"
}

# The optional key is passed by environment name and as a header from a file descriptor; never printed.
api_curl() {
  if [[ -n "${TENSORFOLD_API_KEY:-}" ]]; then
    curl --header @<(printf 'Authorization: Bearer %s\n' "${TENSORFOLD_API_KEY%%,*}") "$@"
  else
    curl "$@"
  fi
}

launch() {
  local ctx=$1
  local -a user_args=() mounts=()
  build_serve_args "$ctx"
  build_env_args
  [[ -z "$CONTAINER_USER" ]] || user_args=(--user "$CONTAINER_USER")
  RUN_LOG_DIR="$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)-$$-$ATTEMPT"
  mounts=(-v "$MODEL_DIR:/model:ro" "${OVERLAY_MOUNT[@]}")
  [[ "$DRAFTER" != dflash2 ]] || mounts+=(-v "$DFLASH2_DIR:/dflash2:ro")
  [[ -z "$TEMPLATE_FILE" ]] || mounts+=(-v "$TEMPLATE_FILE:/templates/chat_template.jinja:ro")
  mounts+=(-v "$CACHE_DIR:/state" -v "$RUN_LOG_DIR:/logs")
  log "Runner ($RUN_LOG_DIR/runner.sh) starts, inside the container:"
  show_ranks
  if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$RUN_LOG_DIR" "$CACHE_DIR/home" "$CACHE_DIR/learn"
    write_runner
    ln -sfn -- "$RUN_LOG_DIR" "$LOG_DIR/current"
  fi
  local -a cmd=(docker run -d --name "$CONTAINER_NAME" --label tf.recipe=glm53-zig --label "tf.patches=$PATCHES_HASH"
    --init --restart=no --gpus "\"device=$GPUS\"" --ipc=host --network host
    --ulimit memlock=-1 --ulimit stack=67108864 "${user_args[@]}"
    "${ENV_ARGS[@]}" "${mounts[@]}" --entrypoint /logs/runner.sh "$IMAGE" "${SERVE_ARGS[@]}")
  if [[ "$DRY_RUN" == 1 ]]; then plan "${cmd[@]}"; return 0; fi
  "${cmd[@]}" >/dev/null || fail E_START 'Docker could not launch the server. Check its error and rerun.'
  log "Started $CONTAINER_NAME (logs: $RUN_LOG_DIR)."
}

stop_follower() { kill "${log_follower:-}" 2>/dev/null || true; wait "${log_follower:-}" 2>/dev/null || true; trap - EXIT; }
# 0 once /health answers (rank 0 listens only after every rank loaded and agreed); 1 when the container stopped.
wait_ready() {
  local started=$SECONDS heartbeat=$SECONDS
  docker logs --follow --tail 5 "$CONTAINER_NAME" 8>&- & # closes fd 8: a follower left behind must not hold start.lock
  log_follower=$!
  trap stop_follower EXIT
  until curl --fail --silent --max-time 5 "$URL/health" >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" != true ]]; then
      stop_follower; return 1
    fi
    if (( SECONDS - started >= WAIT_TIMEOUT )); then
      stop_follower
      fail E_API "The API did not come up within $WAIT_TIMEOUT s. Inspect $LOG_DIR/current/rank*.log; raise WAIT_TIMEOUT for a first start (checkpoint digest)."
    fi
    if (( SECONDS - heartbeat >= 30 )); then
      log "Waiting for the API: $((SECONDS - started)) s (first start: each rank hashes the checkpoint once; then the load and graphs)."
      heartbeat=$SECONDS
    fi
    sleep 5
  done
  stop_follower
}
show_failure() {
  local f
  for f in "$RUN_LOG_DIR"/rank*.log; do
    [[ -f "$f" ]] || continue
    printf '\n--- %s (last 15 lines) ---\n' "$f" >&2
    tail -n 15 -- "$f" >&2
  done
}

# The prompt cache rank 0's "prompt cache: X GiB of device pages and Y GiB of pinned host memory a rank" line funds, in
# GiB a rank (X + Y); without that line, the startup line's device cache ("..., a 1.2 GiB prompt cache; ...") or 0 for
# "no prompt cache"; empty when the log has neither.
funded_cache_gib() {
  local line
  line=$(grep -h 'prompt cache: .* GiB of device pages and .* GiB of pinned host memory a rank' "$1" 2>/dev/null | tail -1) || true
  if [[ "$line" =~ prompt\ cache:\ ([0-9][0-9.]*)\ GiB\ of\ device\ pages\ and\ ([0-9][0-9.]*)\ GiB ]]; then
    python3 -c 'import sys; print(round(float(sys.argv[1]) + float(sys.argv[2]), 1))' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi
  line=$(grep -hE ' streams? at once' "$1" 2>/dev/null | tail -1) || true  # one stream: "1 stream at once"
  [[ -n "$line" ]] || return 0
  if [[ "$line" =~ ,\ a\ ([0-9][0-9.]*)\ GiB\ prompt\ cache ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; else printf '0\n'; fi
}
# The newest log archive stop.sh wrote (it removes the run directory once archived): the names carry a UTC stamp, so
# the last in glob order is the newest.
last_archive() {
  local a=("$LOG_DIR"/glm53-zig-*.tar.gz)
  if [[ -e "${a[-1]}" ]]; then printf '%s\n' "${a[-1]}"; else printf 'none written (see %s)\n' "$LOG_DIR"; fi
}
# PROMPT_CACHE_MIN_GIB: the server is stopped before LIVE when rank 0 funds less (PROMPT_CACHE_GIB is a cap, not a
# floor). Rank 0 alone prints these lines; the other ranks follow its plan.
check_cache_min() {
  [[ -n "$PROMPT_CACHE_MIN_GIB" ]] || return 0
  local f="$RUN_LOG_DIR/rank0.log" got
  got=$(funded_cache_gib "$f")
  if [[ -z "$got" ]]; then
    GLM_STOP_LOCK_HELD=1 "$REPO_ROOT/stop.sh"
    fail E_CACHE "PROMPT_CACHE_MIN_GIB=$PROMPT_CACHE_MIN_GIB: rank 0's log has no prompt cache line, so the start cannot be checked; the server is stopped. Logs: $(last_archive)."
  fi
  if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$got" "$PROMPT_CACHE_MIN_GIB"; then
    GLM_STOP_LOCK_HELD=1 "$REPO_ROOT/stop.sh"
    fail E_CACHE "Rank 0 funds a $got GiB prompt cache a rank, under PROMPT_CACHE_MIN_GIB=$PROMPT_CACHE_MIN_GIB; the server is stopped. The cache is what this start's room leaves (device pages after the KV pool, a host tier from the available memory), capped by PROMPT_CACHE_GIB: lower PARALLEL or CONTEXT, free host memory, or set TF_GLM_HOST_CACHE_GIB. Logs: $(last_archive)."
  fi
  log "Prompt cache: rank 0 funds $got GiB a rank (PROMPT_CACHE_MIN_GIB=$PROMPT_CACHE_MIN_GIB)."
}

live_message() {
  brand_style
  printf '%sGLM-5.3-Flash-EXL3 is now LIVE! on port %s%s\n%sEndpoint: %s/v1%s\n' \
    "$AE_SILVER" "$PORT" "$AE_RESET" "$AE_WHITE" "$URL" "$AE_RESET"
}

SMOKE_JSON=$(python3 - "$SERVED_NAME" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
                  "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}))
PY
)

finish_start() {
  if [[ "$DRY_RUN" == 1 ]]; then
    log "Poll every 5 s for up to $WAIT_TIMEOUT s (rank 0 listens once every rank has loaded and agreed):"
    plan curl --fail --silent --max-time 5 "$URL/health"
    if [[ "$CONTEXT_RETRY" == 1 ]]; then
      log "If a rank exits first and rank0.log holds the engine's memory refusal (streams' caches vs room left), ./stop.sh"
      log "archives that run and the container starts once more with the window the figures allow (x$CONTEXT_FIT_MARGIN)."
    fi
    [[ -z "${TENSORFOLD_API_KEY:-}" ]] || log '  (requests below carry Authorization: Bearer from TENSORFOLD_API_KEY; not shown)'
    plan curl --fail --silent --max-time 10 "$URL/v1/models"
    plan curl --fail --silent --show-error --max-time "$SMOKE_TIMEOUT" "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$SMOKE_JSON"
    log 'Would check that /v1/models lists the served name and the reply has nonempty content, then print:'
    live_message
    return 0
  fi
  api_curl --fail --silent --show-error --max-time 10 "$URL/v1/models" > "$RUN_LOG_DIR/models.json" \
    || fail E_API "GET /v1/models failed (an API key mismatch?). Inspect $LOG_DIR/current/rank0.log."
  python3 -c 'import json,sys; ids=[m.get("id") for m in json.load(open(sys.argv[1])).get("data", [])]; sys.exit(0 if sys.argv[2] in ids else 1)' \
    "$RUN_LOG_DIR/models.json" "$SERVED_NAME" || fail E_API "/v1/models does not list $SERVED_NAME (see $RUN_LOG_DIR/models.json)."
  api_curl --fail --silent --show-error --max-time "$SMOKE_TIMEOUT" "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$SMOKE_JSON" > "$RUN_LOG_DIR/smoke.json" \
    || fail E_SMOKE "The smoke request failed. Inspect $LOG_DIR/current/rank0.log and retry."
  python3 - "$RUN_LOG_DIR/smoke.json" <<'PY' || fail E_SMOKE "The smoke test returned no answer, or one character repeated (the weights load but the model does not answer). Inspect $LOG_DIR/current/smoke.json and rank0.log."
import json, sys
reply = json.load(open(sys.argv[1], encoding="utf-8"))
# Explicit checks, not asserts: PYTHONOPTIMIZE would skip those. curl --fail has already refused an HTTP error.
if reply.get("error") is not None: sys.exit("an error reply")
message = reply["choices"][0]["message"]
content = message.get("content")
if message.get("role") != "assistant" or not isinstance(content, str) or not content.strip(): sys.exit("no answer from the assistant")
text = content.strip()
if len(text) >= 8 and len(set(text)) <= 2: sys.exit("one character repeated")
print("Smoke test passed: " + text[:120])
PY
  printf '\n'
  grep -h -m1 -E 'CUDA sm_[0-9]+ device' "$RUN_LOG_DIR/rank0.log" 2>/dev/null || true
  grep -h -m1 -E '^.*serving .* at http' "$RUN_LOG_DIR/rank0.log" 2>/dev/null || true
  nvidia-smi -i "$GPUS" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU /' || true
  printf 'Health: %s/health\nLogs: %s/current/rank*.log\n' "$URL" "$LOG_DIR"
  live_message
}

banner
printf '\n%s\n%s at once | context %s (%s)\n%s dense, %s head, %s KV | %s drafts | vision %s | reserve %s GiB | port %s\n' \
  "$MODEL_DESC" "$PARALLEL" "$CTX" "$CTX_NOTE" "$DENSE" "$HEAD" "$KV$([[ "$KV_PAGES" != on ]] || echo ' (one shared page pool; context = one request'"'"'s cap)')" "$DRAFTER" "$VISION" "$MEMORY_RESERVE_GIB" "$PORT"

step 1 'Check Linux, Docker, the NVIDIA runtime, idle GPUs, host memory, disk space and ports'
if [[ "$DRY_RUN" == 1 ]]; then
  log '[I_DRY_RUN] Plan only: no Docker command, GPU access, download or file change is executed.'
  log "Check Linux x86_64, docker group access, the NVIDIA runtime; GPUs $GPUS with $MIN_GPU_FREE_MIB MiB free each and under"
  log "$MAX_GPU_USED_MIB MiB in use (the desktop's processes tolerated); $MIN_HOST_FREE_GIB GiB host memory (warn); disk for what is missing;"
  log "port $PORT on $HOST and $NCCL_PORT-$((NCCL_PORT + 1)) on 127.0.0.1 free. Refuse if another start or stop holds $STATE_DIR/start.lock."
  if [[ "$MODE" == restart ]]; then DRY_RUN=1 "$REPO_ROOT/stop.sh"; fi
else
  check_platform
  mkdir -p "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
  running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)
  if [[ "$running" == true && "$MODE" == start ]]; then
    log "$CONTAINER_NAME is already running. Endpoint: $URL/v1"
    log 'Use ./start.sh restart to apply changed settings, or ./stop.sh to stop it.'
    RUN_LOG_DIR=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/logs"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME")
    [[ -d "$RUN_LOG_DIR" ]] || fail E_LOG 'The running container has no accessible log directory. Inspect its mounts.'
    wait_ready || { show_failure; fail E_START "A rank exited. Inspect $LOG_DIR/current/rank*.log; ./stop.sh archives them."; }
    finish_start
    exit 0
  fi
  # Settings are validated before an existing server is stopped; its logs are archived first.
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    GLM_STOP_LOCK_HELD=1 "$REPO_ROOT/stop.sh"
  fi
  check_gpus
  check_host_memory
  check_disk
  python3 - "$HOST" "$PORT" "$NCCL_PORT" <<'PY' || fail E_PORT "Port $PORT or $NCCL_PORT-$((NCCL_PORT + 1)) is in use (another server?). Stop its owner or change PORT / NCCL_PORT."
import socket, sys
host, port, nccl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for h, p in ((host, port), ("127.0.0.1", nccl), ("127.0.0.1", nccl + 1)):
    family = socket.AF_INET6 if ":" in h else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((h, p))
PY
  log "[I_PORT] Port $PORT and $NCCL_PORT-$((NCCL_PORT + 1)) are free."
fi

check_weights_access

step 2 "Build the TensorFold native image with $PATCH_COUNT engine patches (skip when ready)"
prepare_image

step 3 'Download or check the checkpoint and the drafter weights'
prepare_weights
model_overlay
[[ "$EFFORT_AFTER_DOWNLOAD" == 0 ]] || effort_template

step 4 'Check CUDA peer-to-peer between the selected GPUs'
check_p2p

step 5 "Start one container with $NUM_GPUS ranks (GPUS=$GPUS: rank r on the r-th GPU listed)"
ATTEMPT=1
launch "$CTX"

step 6 'Wait for the API and run a greedy smoke test'
if [[ "$DRY_RUN" != 1 ]] && ! wait_ready; then
  # Up to CONTEXT_RETRIES more starts, each at the window the last refusal allows (the estimate tightens every time).
  while :; do
    new=''
    # Every rank's log: when one rank leaves first, the runner can end rank 0 before its refusal line is written.
    [[ "$CONTEXT_RETRY" != 1 ]] || new=$(fit_context <(cat "$RUN_LOG_DIR"/rank*.log 2>/dev/null) "$CTX" || true)
    # No figures at all under auto on the first refusal: the newest window any settings served, when smaller.
    if [[ -z "$new" && "$CONTEXT_RETRY" == 1 && "$CONTEXT" == auto && "$ATTEMPT" == 1 && -f "$FIT_FILE" ]]; then
      new=$(sort -t$'\t' -k3,3 "$FIT_FILE" | awk -F'\t' -v asked="$CTX" 'END { if ($2 + 0 >= 8192 && $2 + 0 < asked) print $2 }')
    fi
    # On KV pages the plan sets the engine's own --context window aside first, then the other streams' fixed parts. A
    # refusal there names what fits ("fits: --parallel N, TENSORFOLD_MEMORY_RESERVE_GIB=G (now G), --context T"): under
    # CONTEXT=auto take its --context; otherwise, or when it names none, say so and stop.
    if grep -qh "for its other streams' fixed parts" "$RUN_LOG_DIR"/rank*.log 2>/dev/null; then
      fits=$(grep -ho 'fits: .*' "$RUN_LOG_DIR"/rank*.log 2>/dev/null | tail -1)
      t=$(grep -o -- '--context [0-9]*' <<< "$fits" | grep -o '[0-9]*$' || true)
      if [[ "$CONTEXT" == auto && -n "$t" ]] && (( t >= 8192 && t < CTX )); then
        new=$(( t * 99 / 100 / 1024 * 1024 ))
      else
        show_failure
        fail E_START "The shared KV page pool can't hold $PARALLEL streams' fixed parts beside one $CTX-token window ($(grep -ho "needs [0-9.]* GiB for its other streams' fixed parts.*" "$RUN_LOG_DIR"/rank*.log | tail -1)). ${fits:+The engine says $fits. }Lower CONTEXT or MEMORY_RESERVE_GIB (now $MEMORY_RESERVE_GIB), or PARALLEL."
      fi
    fi
    if [[ -z "$new" ]] || (( ATTEMPT > ${CONTEXT_RETRIES:-3} )); then
      show_failure
      (( ATTEMPT == 1 )) && fail E_START "A rank exited during startup. Inspect $LOG_DIR/current/rank*.log; ./stop.sh archives them."
      fail E_START "Start $ATTEMPT (--context $CTX) failed too. Inspect $LOG_DIR/current/rank*.log; lower CONTEXT or PARALLEL, or raise MEMORY_RESERVE_GIB."
    fi
    warn W_CONTEXT "This start's memory holds less than a $CTX-token window at --parallel $PARALLEL; starting once more with --context $new."
    GLM_STOP_LOCK_HELD=1 "$REPO_ROOT/stop.sh"
    ATTEMPT=$(( ATTEMPT + 1 ))
    CTX=$new
    launch "$CTX"
    wait_ready && break
  done
fi
[[ "$DRY_RUN" == 1 ]] || check_cache_min
finish_start
if [[ "$DRY_RUN" != 1 ]]; then
  if [[ "$CONTEXT" == auto ]]; then
    remember_context "$FIT_KEY" "$CTX"
    log "Context: $CTX tokens (kept for the next start with these settings; delete $FIT_FILE to probe again)."
    # Windows (KV_PAGES=off, or auto with KV=fp8) have no pages line, so no pool to report and no pipefail exit.
    pool=$(grep -ho 'taking [0-9]* pages of 512 rows' "$RUN_LOG_DIR/rank0.log" 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1) || true
    [[ -z "$pool" ]] || log "KV pool: $pool pages = $(( pool * 512 )) tokens shared by $PARALLEL streams (one request up to $CTX)."
    # A few MiB more fixed memory can quietly drop a stream or, with the device leftover gone, the whole prompt cache.
    # One stream reads "1 stream at once"; auto names no count.
    [[ "$PARALLEL" == auto ]] || grep -qE "(^|[^0-9])$PARALLEL streams? at once" "$RUN_LOG_DIR/rank0.log" 2>/dev/null ||
      warn W_STREAMS "The engine serves fewer than PARALLEL=$PARALLEL streams at once; see the startup line in $RUN_LOG_DIR/rank0.log."
    [[ "$PROMPT_CACHE_GIB" == 0 ]] || grep -q 'pinned host memory' "$RUN_LOG_DIR/rank0.log" 2>/dev/null ||
      warn W_CACHE "No host prompt-cache tier this start: follow-up turns re-read their whole prompt. See the prompt cache line in $RUN_LOG_DIR/rank0.log."
  elif (( ATTEMPT == 2 )); then
    log "Context: $CTX tokens, not the $CONTEXT asked; set CONTEXT=$CTX in scripts/local.sh to skip the refused start."
  fi
fi
