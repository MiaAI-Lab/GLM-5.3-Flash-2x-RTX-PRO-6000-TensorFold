#!/usr/bin/env python3
"""Build a partial abliteration overlay: a tree like the parent's, with chosen layers' self_attn.o_proj taken from a donor.

Both trees are read as safetensors: the tensor is copied byte for byte out of the donor's shard into the parent's shard at
the offsets its own header gives, so neither tree is loaded into memory and no framework is needed. Everything the graft
does not touch is hardlinked from the parent, so the overlay costs only the shards it rewrites.

    scripts/graft.py PARENT DONOR OUT --layers 15-29 [--verify]

PARENT and DONOR are checkpoint directories that hold model.safetensors.index.json (the parent's published weights and
the Ablit repository, for instance); OUT is created, or reused if it already holds this overlay. --layers takes a list
with ranges (15-29,45). --verify re-reads every grafted tensor and compares it with the donor's bytes.

Served as MODEL_SOURCE=local MODEL_DIR=OUT, after the same HF_TOKEN check the download path does.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

DEFAULT_TENSOR = "model.language_model.layers.{layer}.self_attn.o_proj.weight"
SKIP = {".cache", ".downloading", ".revision"}
HEADER_MAX = 128 << 20


def fail(msg):
    raise SystemExit(f"graft: {msg}")


def parse_layers(text):
    layers = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                lo, hi = int(lo), int(hi)
            except ValueError:
                fail(f"--layers: '{part}' is not a range of layer numbers")
            if hi < lo:
                fail(f"--layers: '{part}' counts down")
            layers.extend(range(lo, hi + 1))
        else:
            try:
                layers.append(int(part))
            except ValueError:
                fail(f"--layers: '{part}' is not a layer number")
    if not layers:
        fail("--layers: no layers given")
    return sorted(set(layers))


def read_header(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            fail(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", head)[0]
        if n < 2 or n > HEADER_MAX or 8 + n > size:
            fail(f"{path}: header length {n} does not fit a {size}-byte file")
        header = json.loads(f.read(n))
    return n, header, size


def read_index(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(idx):
        fail(f"{path}: no model.safetensors.index.json")
    with open(idx) as f:
        return json.load(f)["weight_map"]


def copy_or_link(src, dst):
    """Hardlink when the parent still owns the inode; copy when dst exists and was already broken by a graft."""
    if os.path.exists(dst):
        os.unlink(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def build_tree(parent, out, skip):
    for name in sorted(os.listdir(parent)):
        if name in SKIP or name in skip:
            continue
        src, dst = os.path.join(parent, name), os.path.join(out, name)
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for sub in sorted(os.listdir(src)):
                d2 = os.path.join(dst, sub)
                if not os.path.exists(d2):
                    try:
                        os.link(os.path.join(src, sub), d2)
                    except OSError:
                        shutil.copyfile(os.path.join(src, sub), d2)
        else:
            copy_or_link(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parent", help="checkpoint directory to take everything but the grafted tensors from")
    ap.add_argument("donor", help="checkpoint directory to take the grafted tensors from")
    ap.add_argument("out", help="directory to build the overlay in")
    ap.add_argument("--layers", required=True, help="layer numbers to graft, with ranges (15-29,45)")
    ap.add_argument("--tensor", default=DEFAULT_TENSOR,
                    help=f"tensor name template with {{layer}} (default: {DEFAULT_TENSOR})")
    ap.add_argument("--verify", action="store_true", help="re-read each grafted tensor and compare it with the donor's")
    a = ap.parse_args()

    parent, donor, out = (os.path.abspath(p) for p in (a.parent, a.donor, a.out))
    for path, what in ((parent, "parent"), (donor, "donor")):
        if not os.path.isdir(path):
            fail(f"{what} {path}: not a directory")
    if out == parent or out == donor:
        fail("OUT must differ from PARENT and DONOR")
    if out.startswith(parent + os.sep):
        fail("OUT must not sit inside PARENT (its hardlinks would follow the parent's files)")

    layers = parse_layers(a.layers)
    parent_map, donor_map = read_index(parent), read_index(donor)
    plan = {}  # shard -> [(tensor, parent_offsets, donor_shard, donor_offsets, size)]
    for layer in layers:
        name = a.tensor.format(layer=layer)
        if name not in parent_map:
            fail(f"{name}: not in the parent's index (layers {layers[0]}-{layers[-1]} asked)")
        if name not in donor_map:
            fail(f"{name}: not in the donor's index")
        plan.setdefault(parent_map[name], []).append((name, donor_map[name]))
    missing = [L for L in layers if a.tensor.format(layer=L) not in parent_map]
    if missing:
        fail(f"layers {missing} have no {a.tensor.format(layer='<n>')} in the parent")

    os.makedirs(out, exist_ok=True)
    build_tree(parent, out, skip=set(plan))
    grafted = 0
    for shard, items in sorted(plan.items()):
        src, dst = os.path.join(parent, shard), os.path.join(out, shard)
        # A copy, never a hardlink: the graft writes through dst, and a link would rewrite the parent's bytes.
        if os.path.exists(dst):
            os.unlink(dst)
        shutil.copyfile(src, dst)
        np_, hp, psize = read_header(src)
        for name, donor_shard in items:
            dpath = os.path.join(donor, donor_shard)
            nd, hd, dsize = read_header(dpath)
            pe, de = hp.get(name), hd.get(name)
            if pe is None or de is None:
                fail(f"{name}: header entry missing")
            if pe.get("dtype") != de.get("dtype") or pe.get("shape") != de.get("shape"):
                fail(f"{name}: donor is {de.get('dtype')}{de.get('shape')}, parent is {pe.get('dtype')}{pe.get('shape')}")
            ps, pstop = pe["data_offsets"]
            ds, dstop = de["data_offsets"]
            if pstop - ps != dstop - ds:
                fail(f"{name}: {pstop - ps} bytes in the parent, {dstop - ds} in the donor")
            if pstop > psize - (8 + np_) or dstop > dsize - (8 + nd):
                fail(f"{name}: data offsets run past the file")
            with open(dpath, "rb") as f:
                f.seek(8 + nd + ds)
                blob = f.read(dstop - ds)
            with open(dst, "r+b") as f:
                f.seek(8 + np_ + ps)
                f.write(blob)
            if a.verify:
                with open(dst, "rb") as f:
                    f.seek(8 + np_ + ps)
                    if f.read(len(blob)) != blob:
                        fail(f"{name}: wrote {len(blob)} bytes but read back something else")
            grafted += len(blob)
    print(f"graft: {len(layers)} layers ({layers[0]}-{layers[-1]}), {len(plan)} shards, {grafted / 2**30:.2f} GiB grafted into {out}")
    print(f"graft: {os.path.getsize(os.path.join(out, 'model.safetensors.index.json'))} bytes of index; "
          f"serve with MODEL_SOURCE=local MODEL_DIR={out}")


if __name__ == "__main__":
    main()
