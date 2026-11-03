#!/usr/bin/env python3
# Compare two or three dec.py outputs (ABA). usage: aba.py A1.json [B.json A2.json]
# Prints per-load trial tps, medians, the A1-A2 drift bound and the B-vs-A effect.
# Also compares round ms and the per-trial token/sha identity across arms.
import json, statistics as st, sys

def load(p):
    return json.load(open(p))

def arm(path):
    d = load(path)
    out = {}
    for k, v in d.items():
        tr = v["trials"]
        out[k] = {
            "tps": [t["total_tps"] for t in tr],
            "rms": [t["round_ms"][0] if "round_ms" in t else None for t in tr],
            "tok": [t["tokens"] for t in tr],
            "rnd": [t.get("sum_rounds", sum(t.get("rounds", []))) for t in tr],
            "sha": [t["shas"][0] if t.get("shas") else None for t in tr],
        }
    return out

def med(x): return st.median(x)
def mean(x): return st.mean(x)
def se(x):
    n = len(x)
    return st.stdev(x) / n ** 0.5 if n > 1 else 0.0

names = [p.split("dec-")[-1].replace(".json", "") for p in sys.argv[1:]]
arms = {n: arm(p) for n, p in zip(names, sys.argv[1:])}
a1, b, a2 = (arms.get(n) for n in (names + [None, None])[:3])

print(f"arms: {', '.join(names)}")
for k in a1:
    print(f"\n== load {k} stream(s) ==")
    for n, d in arms.items():
        t = d[k]["tps"]
        rms = [x for x in d[k]["rms"] if x is not None]
        rms_s = f"| round ms med {med(rms):6.2f}" if rms else ""
        print(f"  {n:8s} tps  {' '.join(f'{x:6.2f}' for x in t)}   med {med(t):6.2f}  mean {mean(t):6.2f} +- {se(t):.2f}  {rms_s}")
    ma, mb = med(a1[k]["tps"]), (med(b[k]["tps"]) if b else None)
    ma2 = med(a2[k]["tps"]) if a2 else None
    drift = abs(ma - ma2) / ma * 100 if ma2 else None
    pooled = [x for x in a1[k]["tps"]] + ([x for x in a2[k]["tps"]] if a2 else [])
    eff = (mb - mean(pooled)) / mean(pooled) * 100 if b else None
    if drift is not None:
        print(f"  drift |A1-A2|  = {drift:5.2f}%   (median {ma:.2f} vs {ma2:.2f})")
    if eff is not None:
        print(f"  effect B vs pooled A = {eff:+5.2f}%  (B med {mb:.2f} vs pooled mean {mean(pooled):.2f} n={len(pooled)})")
    # identity checks across arms
    ref = a1[k]
    for n, d in list(arms.items())[1:]:
        same_tok = d[k]["tok"] == ref["tok"]
        same_sha = d[k]["sha"] == ref["sha"]
        if not (same_tok and same_sha):
            print(f"  IDENTITY {n}: tokens match A1={same_tok}  shas match A1={same_sha}"
                  f"  ({d[k]['tok']} vs {ref['tok']})")
print("\n(no IDENTITY lines above = tokens and shas identical across arms)")
