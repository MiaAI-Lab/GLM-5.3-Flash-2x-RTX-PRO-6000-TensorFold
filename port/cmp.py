#!/usr/bin/env python3
"""Paired per-token comparison of two nll-*.json runs: mean delta, standard error, sign test, per passage."""
import json, math, sys
a=json.load(open(f"nll-{sys.argv[1]}.json")); b=json.load(open(f"nll-{sys.argv[2]}.json"))
print(f"# {sys.argv[2]} minus {sys.argv[1]}  (positive = {sys.argv[2]} assigns HIGHER logprob = better)")
alld=[]; rows=[]
for k in a["per_passage"]:
    pa, pb = a["per_passage"][k], b["per_passage"][k]
    if [t for t,_ in pa] != [t for t,_ in pb]:
        print(f"  {k}: TOKEN MISMATCH, skipped"); continue
    d=[lb-la for (_,la),(_,lb) in zip(pa,pb)]
    alld+=d
    rows.append((k,len(d),-sum(la for _,la in pa)/len(pa),-sum(lb for _,lb in pb)/len(pb),sum(d)/len(d)))
print(f"{'passage':14s} {'tok':>5s} {'NLL a':>9s} {'NLL b':>9s} {'mean d':>9s}")
for k,n,na,nb,m in rows: print(f"  {k:12s} {n:5d} {na:9.5f} {nb:9.5f} {m:+9.5f}")
n=len(alld); mean=sum(alld)/n
sd=math.sqrt(sum((x-mean)**2 for x in alld)/(n-1)); se=sd/math.sqrt(n)
better=sum(1 for x in alld if x>0); worse=sum(1 for x in alld if x<0); same=n-better-worse
print(f"\n  tokens {n}")
print(f"  mean per-token logprob delta {mean:+.5f}  (se {se:.5f}, 95% CI {mean-1.96*se:+.5f} .. {mean+1.96*se:+.5f})")
print(f"  NLL: a {-sum(la for k in a['per_passage'] for _,la in a['per_passage'][k])/n:.5f}"
      f"  b {-sum(lb for k in b['per_passage'] for _,lb in b['per_passage'][k])/n:.5f}")
print(f"  tokens where b better: {better} ({better/n:.1%}), worse: {worse} ({worse/n:.1%}), identical: {same} ({same/n:.1%})")
if same==n: print("  -> bit-identical on every token")
