#!/usr/bin/env bash
# KV_SHARD=1 (engine patch 0104) runs on fp8 KV pages, with DFlash2 or no drafter, at more than one stream. The engine
# refuses anything else when the ranks start, after `./start.sh restart` has stopped the serving pair, so
# validate_config refuses it first. Each case starts from the defaults, given here so that scripts/local.sh and .env
# cannot change them, with KV_SHARD=1 unless it sets KV_SHARD itself.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
fails=0
check() { # check <name> <want: accepted|refused> [VAR=value]...
  local name=$1 want=$2 err rc got
  shift 2
  err=$(mktemp)
  env -i PATH="$PATH" HOME="$HOME" GPUS=0,1 KV=fp8 KV_PAGES=on DRAFTER=dflash2 PARALLEL=4 KV_SHARD=1 "$@" bash -c '
    source scripts/prepare.sh
    validate_config >/dev/null' 2>"$err" && rc=0 || rc=$?
  if (( rc == 0 )); then got=accepted
  elif grep -q 'E_CONFIG.*KV_SHARD=1' "$err"; then got=refused
  else got="error $rc: $(head -1 "$err")"; fi
  rm -f -- "$err"
  if [[ "$got" == "$want" ]]; then echo "kv-shard: $name: pass"; else echo "kv-shard: $name: FAIL (got $got, want $want)" >&2; fails=1; fi
}
check 'the defaults are accepted' accepted
check 'no drafter is accepted' accepted DRAFTER=none
check 'PARALLEL=auto is accepted' accepted PARALLEL=auto
check 'KV_PAGES=off is refused' refused KV_PAGES=off
check 'KV_PAGES=auto (fp8 windows) is refused' refused KV_PAGES=auto
check 'PARALLEL=1 is refused' refused PARALLEL=1
check 'the MTP drafter is refused' refused DRAFTER=mtp
check 'KV=bf16 is refused' refused KV=bf16
check 'KV_SHARD=0 leaves KV_PAGES=off alone' accepted KV_SHARD=0 KV_PAGES=off
check 'three GPUs are refused' refused GPUS=0,1,2
exit $fails
