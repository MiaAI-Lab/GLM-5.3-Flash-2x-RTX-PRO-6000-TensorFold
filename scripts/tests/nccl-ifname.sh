#!/usr/bin/env bash
# An NCCL exclusion list exported for the whole box (NCCL_SOCKET_IFNAME=^lo,docker,... from /etc/profile.d; the 2x DGX
# Spark recipe's issue 17) outranks scripts/local.sh and would keep NCCL off loopback, the ranks' link on this host:
# validate_config warns and uses lo. A netdev of your own is kept, and empty is the default lo.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
fails=0
check() { # check <name> <value> <want: kept|lo|error>
  local out err rc got
  err=$(mktemp)
  out=$(env -i PATH="$PATH" HOME="$HOME" NCCL_SOCKET_IFNAME="$2" bash -c '
    source scripts/prepare.sh
    validate_config >/dev/null
    printf "%s" "$NCCL_SOCKET_IFNAME"' 2>"$err") && rc=0 || rc=$?
  if (( rc != 0 )); then got="error $rc: $(head -1 "$err")"
  elif [[ "$out" == "$2" ]]; then got=kept
  elif [[ "$out" == lo ]]; then got=lo
  else got="changed to $out"; fi
  if [[ "$3" == lo && "$2" == ^* ]] && ! grep -q 'W_CONFIG.*exclusion list' "$err"; then got="$got, no warning"; fi
  rm -f -- "$err"
  if [[ "$got" == "$3" ]]; then echo "nccl-ifname: $1: pass"; else echo "nccl-ifname: $1: FAIL (got $got, want $3)" >&2; fails=1; fi
}
check 'an exclusion list is lo, with a warning' '^lo,docker,tailscale,enP7s7,wlP9s9,veth' lo
check 'a netdev is kept' 'eno1' kept
check 'lo is kept' 'lo' kept
check 'empty is lo' '' lo
exit $fails
