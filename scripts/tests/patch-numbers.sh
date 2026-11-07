#!/usr/bin/env bash
# Each number in patches/engine/ names one patch (the 2x DGX Spark recipe's issue 15). The image applies the patches in filename order, so two
# files with one number still build, but their order then hangs on the words after the number, and "patch 0103" in a
# commit, the CHANGELOG or a log no longer names one file. Only numbered files are read: a NNNN-*.patch placeholder
# in a PR does not trip it.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
dir=${PATCH_DIR:-patches/engine}
dups=$(find "$dir" -maxdepth 1 -name '[0-9][0-9][0-9][0-9]-*.patch' -printf '%f\n' | cut -c1-4 | sort | uniq -d)
[[ -z "$dups" ]] || { echo "patch numbers used twice in $dir: $(echo $dups)" >&2; exit 1; }
echo 'patch numbers: pass'
