#!/usr/bin/env bash
# One measured leg: apply env settings, restart the pair, measure, check the output contract.
#
#   docs/leg.sh <label> [KEY=VALUE ...]
#
# Only exported TF_*/TENSORFOLD_*/NCCL_* variables reach the ranks (start.sh:147-180), and each leg runs
# in its own shell, so a leg with no KEY=VALUE is the baseline configuration. Settings that are recipe
# variables (DENSE, HEAD, ...) are exported too; start.sh reads them from the environment.
#
# Writes docs/speed/leg-<label>.log, dec-<label>.json, contract-<label>.json and prints the
# token_sha diff against contract-baseline.json. Never edits the repo's shipped settings.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
LABEL=${1:?usage: leg.sh <label> [KEY=VALUE ...]}; shift
OUT=docs/speed; mkdir -p "$OUT"
LOG=$OUT/leg-$LABEL.log
: > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "### leg $LABEL  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for kv in "$@"; do
  [[ "$kv" == *=* ]] || { say "BAD ARG $kv"; exit 2; }
  export "$kv"
  say "  set $kv"
done
(( $# )) || say "  (no overrides: baseline configuration)"

say "--- restart ---"
if ! bash docs/safestart.sh >> "$LOG" 2>&1; then
  say "FAILED: ./start.sh restart (see $LOG). Leg aborted; the pair may be down."
  exit 3
fi
grep -hE "CUDA sm_121 device 0" logs/current/rank0.log | tee -a "$LOG"
grep -hE "dflash2 windows:" logs/current/rank0.log | tee -a "$LOG"

say "--- decode sweep ---"
python3 -I docs/dec.py "$LABEL" 2>&1 | tee -a "$LOG"
say "--- contract corpus ---"
python3 -I docs/dec.py "$LABEL" --contract 2>&1 | tee -a "$LOG"

say "--- token_sha contract vs baseline ---"
python3 -I - "$LABEL" 2>&1 <<'PY' | tee -a "$LOG"
import json, os, sys
d = os.path.join("docs", "speed")
label = sys.argv[1]
base = os.path.join(d, "contract-baseline.json")
cur = os.path.join(d, f"contract-{label}.json")
if not os.path.exists(base):
    print("  NO BASELINE CAPTURE - cannot check the contract"); sys.exit(0)
a, b = json.load(open(base)), json.load(open(cur))
bad = []
for k in a:
    sa, sb = a[k].get("token_sha"), b.get(k, {}).get("token_sha")
    flag = "OK  " if sa == sb else "DIFF"
    if sa != sb: bad.append(k)
    print(f"  {flag} {k:9s} baseline {sa} leg {sb}")
print(f"  CONTRACT: {'IDENTICAL on all ' + str(len(a)) + ' prompts' if not bad else 'BROKEN on ' + ','.join(bad)}")
PY
say "### leg $LABEL done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
