#!/usr/bin/env bash
# PROMPT_CACHE_MIN_GIB (the 2x DGX Spark recipe's issue 16): start.sh reads the prompt cache rank 0's log funds a rank
# (device pages plus the pinned host tier) and stops the server before LIVE when it funds less. The test runs
# funded_cache_gib and check_cache_min from start.sh on lines in the engine's own formats, with the stop stubbed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
# The functions alone, from start.sh (sourcing start.sh would run the start).
fns=$(awk '/^(funded_cache_gib|last_archive|check_cache_min)\(\) \{/ { on = 1 } on { print } on && /^\}/ { on = 0 }' start.sh)
[[ "$fns" == *'funded_cache_gib()'* && "$fns" == *'check_cache_min()'* ]] || { echo 'functions not found in start.sh' >&2; exit 1; }
tiers() { printf 'info: prompt cache: %s GiB of device pages and %s GiB of pinned host memory a rank (16.0 GiB over the 2 ranks; TF_GLM_HOST_CACHE_GIB unset: a quarter of the host'"'"'s available memory, at most 16 GiB)\n' "$1" "$2"; }
start_line() { printf '[tensorfold] CUDA sm_120 device 0 (NVIDIA RTX PRO 6000 Blackwell Workstation Edition): model 81.39 GiB; 4 streams at once, the engine'"'"'s own and 3 more taking 1592 pages of 512 rows (4.84 GiB) as they need them (enough for one at a 813056-token window), 395 MiB fixed each, of 7.2 GiB left after a 2.8 GiB reserve; prompts in passes of up to 2048 rows%s; own glue kernels\n' "$1"; }

# run MIN LOGFILE_CONTENT: prints check_cache_min's exit status, then what it logged, failed and stopped.
run() {
  local d="$tmp/run.$RANDOM"; mkdir -p "$d"
  printf '%s' "$2" > "$d/rank0.log"
  # shellcheck disable=SC2034,SC2329 # the variables and stubs are read by the functions eval'd below
  (
    set +e
    PROMPT_CACHE_MIN_GIB=$1 RUN_LOG_DIR=$d REPO_ROOT=$tmp/repo LOG_DIR=$tmp/logs
    mkdir -p "$REPO_ROOT" "$LOG_DIR"
    # as stop.sh: archive the run directory, then remove it
    printf '#!/bin/sh\necho stopped\ntar -czf "%s/glm53-zig-1-1.tar.gz" -C "%s" . && rm -rf -- "%s"\n' "$LOG_DIR" "$d" "$d" > "$REPO_ROOT/stop.sh"; chmod +x "$REPO_ROOT/stop.sh"
    log() { echo "log: $*"; }
    fail() { echo "fail: $1 logs=${2##*Logs: }"; exit 1; }
    eval "$fns"
    check_cache_min
    echo "status $?"
  ) 2>&1 | sed "s#$tmp/logs#LOGS#g" | tr '\n' '|'
}

check() { # check NAME GOT WANT
  if [[ "$2" == "$3" ]]; then echo "cache-min: $1: pass"; else echo "cache-min: $1: got [$2], want [$3]" >&2; exit 1; fi
}

eval "$fns"
f="$tmp/one.log"
{ tiers 1.2 8.0; start_line ', a 1.2 GiB prompt cache'; } > "$f"; check 'both tiers add up' "$(funded_cache_gib "$f")" 9.2
{ tiers 0.0 8.0; start_line ', no prompt cache'; } > "$f"; check 'host tier alone' "$(funded_cache_gib "$f")" 8.0
start_line ', a 1.2 GiB prompt cache' > "$f"; check 'startup line alone is its device cache' "$(funded_cache_gib "$f")" 1.2
start_line ', no prompt cache' > "$f"; check 'no cache is 0' "$(funded_cache_gib "$f")" 0
: > "$f"; check 'no line is empty' "$(funded_cache_gib "$f")" ''
start_line ', a 1.2 GiB prompt cache' | sed 's/4 streams at once/1 stream at once/' > "$f"
check 'one stream (PARALLEL=1) startup line alone is read' "$(funded_cache_gib "$f")" 1.2

ok="$(tiers 1.2 8.0)"
check 'off when empty' "$(run '' "$ok")" 'status 0|'
check 'enough' "$(run 9 "$ok")" 'log: Prompt cache: rank 0 funds 9.2 GiB a rank (PROMPT_CACHE_MIN_GIB=9).|status 0|'
check 'short stops the server' "$(run 12 "$ok")" 'stopped|fail: E_CACHE logs=LOGS/glm53-zig-1-1.tar.gz.|'
check 'no line stops the server' "$(run 1 '')" 'stopped|fail: E_CACHE logs=LOGS/glm53-zig-1-1.tar.gz.|'
