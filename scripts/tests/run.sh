#!/usr/bin/env bash
# The recipe's checks that need no GPU: every script parses, shellcheck when it is installed, and the tests here.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
bash -n start.sh stop.sh scripts/*.sh scripts/tests/*.sh
echo 'syntax: pass'
if command -v shellcheck >/dev/null; then
  shellcheck -x start.sh stop.sh scripts/*.sh scripts/tests/*.sh && echo 'shellcheck: pass'
else
  echo 'shellcheck: not installed, skipped'
fi
for f in scripts/*.py; do
  python3 -c 'import ast, sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$f"
done
echo 'python: pass'
for t in scripts/tests/*.sh; do
  [[ "$t" != scripts/tests/run.sh ]] || continue
  bash "$t"
done
