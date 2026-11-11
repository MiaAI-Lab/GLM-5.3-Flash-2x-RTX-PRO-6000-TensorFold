#!/usr/bin/env bash
# scripts/graft.py builds a partial abliteration overlay by moving self_attn.o_proj bytes from a donor tree into a copy
# of a parent tree. A graft must take only the named layers, leave every other tensor as the parent had it, and never
# touch the parent itself. Two synthetic 2-layer trees stand in for the 164 GiB ones.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

python3 - "$tmp" <<'PY'
import json, os, struct, sys
root = sys.argv[1]
def write(path, tensors):
    header, blobs, off = {}, b"", 0
    for name, blob in tensors.items():
        header[name] = {"dtype": "F32", "shape": [len(blob) // 4], "data_offsets": [off, off + len(blob)]}
        blobs += blob
        off += len(blob)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + blobs)
for tree, fill in (("parent", b"\x01"), ("donor", b"\x02")):
    d = os.path.join(root, tree)
    os.makedirs(d)
    names = [f"model.language_model.layers.{L}.self_attn.o_proj.weight" for L in (0, 1)]
    write(os.path.join(d, "model-00001-of-00001.safetensors"), {n: fill * 16 for n in names})
    json.dump({"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"model_type": "glm"}, open(os.path.join(d, "config.json"), "w"))
PY

before=$(cd "$tmp/parent" && sha256sum * | sort)
python3 scripts/graft.py "$tmp/parent" "$tmp/donor" "$tmp/out" --layers 1 --verify > "$tmp/log" 2>&1
after=$(cd "$tmp/parent" && sha256sum * | sort)
fails=0
check() { # check <name> <condition...>
  local name=$1; shift
  if "$@"; then echo "graft: $name: pass"; else echo "graft: $name: FAIL"; fails=$((fails + 1)); fi
}

check "the parent is untouched" [ "$before" = "$after" ]
check "the grafted layer is the donor's" python3 -c '
import json, struct, sys
p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    for name, e in h.items():
        f.seek(8 + n + e["data_offsets"][0])
        blob = f.read(e["data_offsets"][1] - e["data_offsets"][0])
        want = b"\x02" if ".layers.1." in name else b"\x01"
        if blob != want * 16:
            raise SystemExit(f"{name}: {blob[:4]!r} is not the {want!r} tree")
' "$tmp/out/model-00001-of-00001.safetensors"
check "the other layer is the parent's" python3 -c '
import json, struct, sys
p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    name = "model.language_model.layers.0.self_attn.o_proj.weight"
    f.seek(8 + n + h[name]["data_offsets"][0])
    raise SystemExit(0 if f.read(16) == b"\x01" * 16 else 1)
' "$tmp/out/model-00001-of-00001.safetensors"
check "the rest of the tree is copied" test -f "$tmp/out/config.json"
check "the index is copied" cmp -s "$tmp/parent/model.safetensors.index.json" "$tmp/out/model.safetensors.index.json"
check "a missing layer is refused" bash -c '! python3 scripts/graft.py "$1/parent" "$1/donor" "$1/out2" --layers 7 >/dev/null 2>&1' _ "$tmp"
ln -s "$tmp/parent" "$tmp/alias"  # OUT spelled through a symlink to PARENT: refused, and the parent keeps every file
check "an OUT that resolves to PARENT is refused" bash -c '! python3 scripts/graft.py "$1/parent" "$1/donor" "$1/alias" --layers 1 >/dev/null 2>&1' _ "$tmp"
check "the parent is untouched after that refusal" [ "$before" = "$(cd "$tmp/parent" && sha256sum * | sort)" ]
# both steps inside the check, so a refusal that does not come is a FAIL line, not a silent set -e exit
shape_mismatch_refused() {
  python3 -c '
import json, struct, sys
d = sys.argv[1] + "/donor"
p = d + "/model-00001-of-00001.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); body = f.read()
name = "model.language_model.layers.1.self_attn.o_proj.weight"
h[name]["shape"] = [8]
blob = json.dumps(h).encode()
open(p, "wb").write(struct.pack("<Q", len(blob)) + blob + body)
' "$tmp" && ! python3 scripts/graft.py "$tmp/parent" "$tmp/donor" "$tmp/out3" --layers 1 >/dev/null 2>&1
}
check "a dtype or shape mismatch is refused" shape_mismatch_refused
(( fails == 0 )) || { echo "graft: $fails check(s) failed"; exit 1; }
