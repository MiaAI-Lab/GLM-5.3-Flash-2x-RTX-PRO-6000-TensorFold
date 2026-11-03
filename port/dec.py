#!/usr/bin/env python3
"""Decode throughput at 1/2/4 concurrent streams, and the greedy token-ID contract corpus.

Two modes:
  dec.py <label>            decode sweep: 1/2/4 streams x TRIALS, fresh content per trial, writes
                            docs/speed/dec-<label>.json and prints per-trial tok/s and the spread.
  dec.py <label> --contract the frozen contract corpus (prose, code, boundary, long context) at
                            temperature 0, recording the engine's token_sha per prompt.

Why the engine's own fields: /v1/chat/completions refuses `logprobs` on this backend, but every reply
carries tensorfold.token_sha (a hash of the generated token IDs) plus the speculative block
(rounds, drafted, accepted, tokens_per_round). token_sha is the output contract; tokens_per_round is
the quantity the draft-window rule optimises.

Method note (measured 2026-11-03, and it changed the design): with a random nonce per trial the
1-stream spread was 12.2%, because acceptance is content-dependent (0.517 / 0.625 / 0.659 over three
trials) and throughput tracks it exactly. Rounds per second, meanwhile, was stable to under 0.5%
(17.60 / 17.56 / 17.65). Greedy decode on a FIXED prompt is deterministic, so identical prompts give
identical replies, identical token counts and identical acceptance - leaving only machine timing.
The prompts are therefore fixed and shared by every leg, and the reported quantities are:
  decode_tps = completion / (seconds - prefill_seconds)     per request, prefill removed
  round_ms   = (seconds - prefill_seconds) / rounds * 1000   the quantity the cost tables price
  tokens_per_round                                           the quantity the window rule optimises
An exact draft-policy change keeps the reply identical, so it must show up as FEWER ROUNDS for the
same tokens. Prefill is measured by pf2.py, which does need fresh content; decode does not.
"""
import json, os, random, statistics, sys, threading, time, urllib.request

BASE = os.environ.get("U", "http://127.0.0.1:8888")
MODEL = os.environ.get("M", "GLM-5.3-Flash")
TRIALS = int(os.environ.get("TRIALS", "3"))
MAXTOK = int(os.environ.get("MAXTOK", "250"))
LOADS = [int(x) for x in os.environ.get("LOADS", "1,2,4").split(",")]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed")

# CTXWORDS>0 prepends a deterministic filler so decode runs with a long context already in the
# window - the regime where a shared-window cost table matters most, since round cost grows with
# context. The filler is seeded, so it is byte-identical across legs and stays comparable.
CTXWORDS = int(os.environ.get("CTXWORDS", "0"))

LONG = ["Write five detailed paragraphs about the history of ocean navigation.",
        "Explain how a modern operating system schedules threads. Be thorough and use several paragraphs.",
        "Describe the engineering of suspension bridges in five paragraphs.",
        "Write five paragraphs on the development of the printing press and its consequences."]

# The frozen contract corpus: greedy, short, and diverse in kind. Never edit between legs - the whole
# point is that every leg answers the identical prompts and we diff token_sha.
CONTRACT = {
 "prose":    "In exactly three sentences, explain why the sky appears blue.",
 "code":     "Write a Python function that merges two sorted lists into one sorted list. Code only.",
 "spelling": "Spell the word necessary backwards.",
 "math":     "What is 847 times 23? Show the steps.",
 "zig":      "Write a Zig function that returns the sum of a slice of u32. Code only.",
 "json":     "Return a JSON object with keys name, age, city for a fictional person. JSON only.",
 "boundary": None,   # filled below: a prompt that lands near a 4096-row chunk boundary
 "longctx":  None,   # filled below: well past one chunk
}
def filler(nwords, seed):
    random.seed(seed)
    return " ".join(f"w{random.randint(0, 99999)}" for _ in range(nwords))
CONTRACT["boundary"] = "Reply with the single word ok. Ignore this text:\n" + filler(2800, 11)
CONTRACT["longctx"]  = "Reply with the single word ok. Ignore this text:\n" + filler(12000, 12)

def with_context(prompt):
    if CTXWORDS <= 0:
        return prompt
    random.seed(4242)                     # fixed: identical bytes in every leg
    filler = " ".join(f"w{random.randint(0, 99999)}" for _ in range(CTXWORDS))
    return "Reference text (ignore it, answer the question after it):\n" + filler + "\n\n" + prompt

def chat(text, maxtok, timeout=1800):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}],
                       "temperature": 0.0, "max_tokens": maxtok,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    el = time.time() - t0
    tf, sp, u = r.get("tensorfold", {}), r.get("speculative", {}), r.get("usage", {})
    return {"wall": el, "completion": u.get("completion_tokens"), "prompt": u.get("prompt_tokens"),
            "cached": u.get("prompt_tokens_details", {}).get("cached_tokens"),
            "tps": tf.get("tokens_per_second"), "seconds": tf.get("seconds"),
            "prefill_s": tf.get("prefill_seconds"), "ttft": tf.get("time_to_first_token"),
            "token_sha": tf.get("token_sha"), "min_rows": tf.get("min_rows"),
            "drafts": tf.get("drafts"), "spec": sp,
            "text": (r["choices"][0]["message"].get("content") or "")}

def contract(label):
    out = {}
    print(f"# contract corpus, greedy, {MODEL} at {BASE}")
    for name, prompt in CONTRACT.items():
        try:
            d = chat(prompt, 120)
            out[name] = {"token_sha": d["token_sha"], "completion": d["completion"],
                         "prompt": d["prompt"], "text": d["text"]}
            print(f"  {name:9s} {d['prompt']:6d} tok -> sha {d['token_sha']}  "
                  f"{d['completion']:3d} out  {d['text'][:46]!r}")
        except Exception as e:
            out[name] = {"error": repr(e)[:160]}
            print(f"  {name:9s} ERROR {repr(e)[:120]}")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"contract-{label}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"# wrote {p}")
    return out

def sweep(label):
    res = {}
    print(f"# decode sweep, {MODEL} at {BASE}, {TRIALS} trials, max_tokens {MAXTOK}"
          + (f", context filler {CTXWORDS} words" if CTXWORDS else ""))
    for n in LOADS:
        trials = []
        for t in range(TRIALS):
            # Fixed prompts, deliberately NOT randomised: see the method note above.
            outs, errs = [None] * n, []
            def run(i):
                try:
                    outs[i] = chat(with_context(LONG[i % len(LONG)]), MAXTOK)
                except Exception as e:
                    errs.append(repr(e)[:160])
            th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
            t0 = time.time()
            for x in th: x.start()
            for x in th: x.join()
            el = time.time() - t0
            good = [o for o in outs if o]
            if errs or not good:
                print(f"  {n} stream(s) trial {t+1}: ERROR {errs[:1]}")
                continue
            tok = sum(o["completion"] for o in good)
            tot = tok / el
            dtps, rms = [], []
            for o in good:
                ds = (o["seconds"] or 0) - (o["prefill_s"] or 0)
                if ds > 0 and o["completion"]:
                    dtps.append(o["completion"] / ds)
                    rr = (o.get("spec") or {}).get("rounds")
                    if rr: rms.append(ds / rr * 1000)
            tpr = [o["spec"].get("tokens_per_round") for o in good if o.get("spec")]
            acc = [o["spec"].get("acceptance_rate") for o in good if o.get("spec")]
            dr  = [o["spec"].get("drafted") for o in good if o.get("spec")]
            rd  = [o["spec"].get("rounds") for o in good if o.get("spec")]
            trials.append({"total_tps": tot, "per_stream_tps": tot / n, "tokens": tok, "elapsed": el,
                           "tokens_per_round": tpr, "acceptance": acc, "drafted": dr, "rounds": rd,
                           "decode_tps": dtps, "round_ms": rms,
                           "sum_rounds": sum(x for x in rd if x) if rd else None,
                           "shas": [o["token_sha"] for o in good],
                           "cached": [o["cached"] for o in good]})
            m = lambda v: (statistics.mean([x for x in v if x is not None]) if any(x is not None for x in v) else None)
            print(f"  {n} stream(s) trial {t+1}: {tot:6.1f} tok/s total, {tot/n:5.1f}/stream  "
                  f"tok {tok:4d} rounds {sum(x for x in rd if x):4d}  "
                  f"round {statistics.mean(rms) if rms else float('nan'):5.1f} ms  "
                  f"tok/round {m(tpr):.3f}  accept {m(acc):.3f}  drafted/round "
                  f"{(sum(dr)/sum(rd)) if rd and sum(rd) else float('nan'):.2f}")
        if trials:
            tps = [x["total_tps"] for x in trials]
            med, spread = statistics.median(tps), (max(tps) - min(tps)) / statistics.median(tps) * 100
            rmall = [v for x in trials for v in x["round_ms"]]
            rmed = statistics.median(rmall) if rmall else None
            rspread = ((max(rmall) - min(rmall)) / rmed * 100) if rmall else None
            # Tail, not just the mean: a scheduler that trades drafts for round time can hide a worse
            # tail, and mean tok/s masks it. Per-request round_ms averages over that request's rounds,
            # so this is a tail across requests, not across individual rounds.
            rs = sorted(rmall)
            rp95 = rs[min(len(rs) - 1, int(0.95 * len(rs)))] if rs else None
            rmax = rs[-1] if rs else None
            tpsall = [x["total_tps"] for x in trials]
            tmean = statistics.mean(tpsall)
            tsd = statistics.stdev(tpsall) if len(tpsall) > 1 else 0.0
            tse = tsd / (len(tpsall) ** 0.5) if tpsall else 0.0
            toks = {x["tokens"] for x in trials}
            rnds = {x["sum_rounds"] for x in trials}
            res[str(n)] = {"trials": trials, "median_total_tps": med, "spread_pct": spread,
                           "mean_total_tps": tmean, "sd_total_tps": tsd, "se_total_tps": tse,
                           "n_trials": len(tpsall),
                           "median_round_ms": rmed, "round_ms_spread_pct": rspread,
                           "p95_round_ms": rp95, "max_round_ms": rmax,
                           "tokens_identical": len(toks) == 1, "rounds_identical": len(rnds) == 1,
                           "tokens": sorted(toks), "rounds": sorted(x for x in rnds if x is not None)}
            fmt = lambda v: f"{v:.2f}" if v is not None else "n/a"  # a load may have no usable round ms
            print(f"  -> {n} stream(s): median {med:.1f} tok/s (mean {tmean:.2f} +- se {tse:.2f}, "
                  f"n={len(tpsall)}), spread {spread:.2f}%  |  "
                  f"round p50 {fmt(rmed)} p95 {fmt(rp95)} max {fmt(rmax)} ms, spread {fmt(rspread)}%  |  "
                  f"tokens {'identical' if len(toks)==1 else sorted(toks)}  "
                  f"rounds {'identical' if len(rnds)==1 else sorted(x for x in rnds if x is not None)}")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"dec-{label}.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"# wrote {p}")
    for n, r in res.items():
        rmsp = r["round_ms_spread_pct"]
        tail = (f"round_ms spread {rmsp:.2f}% -> {2*rmsp:.2f}%" if rmsp is not None
                else "no round ms data")
        print(f"# NOISE FLOOR {n} stream(s): tok/s spread {r['spread_pct']:.2f}% "
              f"-> a win must exceed {2*r['spread_pct']:.2f}%; {tail}")
    return res

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    if "--contract" in sys.argv: contract(label)
    else: sweep(label)
