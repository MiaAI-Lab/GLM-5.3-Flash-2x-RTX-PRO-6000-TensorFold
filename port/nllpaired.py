#!/usr/bin/env python3
"""Per-token logprobs over a fixed, varied corpus, saved for a paired comparison between configs.

Usage: nllpaired.py <label>   -> writes nll-<label>.json
Paired beats aggregate: both configs score the same token sequences, so the per-token difference has a
standard error and a sign test, where two separate means of ~0.75 would hide a 0.005 effect in noise.
"""
import json, os, sys, urllib.request

BASE=os.environ.get("U","http://127.0.0.1:8888"); MODEL=os.environ.get("M","GLM-5.3-Flash")

CORPUS = {
 "en_tech": "A memory management unit translates virtual addresses to physical addresses using a page table. "
   "When a translation is absent from the table the processor raises a page fault, and the operating system either "
   "maps a new frame, fetches the page from backing store, or terminates the offending process. Translation "
   "lookaside buffers cache recent translations because walking a multi-level page table on every access would be "
   "prohibitively slow. Modern processors use several levels of translation cache and tag entries with an address "
   "space identifier so that a context switch need not flush them all.",
 "en_narrative": "The harbour emptied slowly through the afternoon. Boats that had come in heavy with mackerel rode "
   "high again, and the gulls, having learned the rhythm of the place, waited on the bollards rather than the water. "
   "An old man mended a net he no longer used, less from need than from the habit of keeping his hands occupied, and "
   "the smell of diesel and salt settled over the stones as the light went orange and then grey.",
 "python": "import heapq\n\ndef dijkstra(graph, source):\n    dist = {source: 0}\n    seen = set()\n"
   "    queue = [(0, source)]\n    while queue:\n        d, node = heapq.heappop(queue)\n"
   "        if node in seen:\n            continue\n        seen.add(node)\n"
   "        for neighbour, weight in graph[node].items():\n            nd = d + weight\n"
   "            if neighbour not in dist or nd < dist[neighbour]:\n                dist[neighbour] = nd\n"
   "                heapq.heappush(queue, (nd, neighbour))\n    return dist\n",
 "zig": "const std = @import(\"std\");\n\npub fn Ring(comptime T: type, comptime n: usize) type {\n    return struct {\n"
   "        items: [n]T = undefined,\n        head: usize = 0,\n        len: usize = 0,\n\n"
   "        pub fn push(self: *@This(), value: T) !void {\n            if (self.len == n) return error.Full;\n"
   "            self.items[(self.head + self.len) % n] = value;\n            self.len += 1;\n        }\n    };\n}\n",
 "math": "Let f be continuous on the closed interval [a, b] and differentiable on the open interval (a, b). "
   "The mean value theorem asserts that there exists a point c in (a, b) such that the derivative of f at c equals "
   "the quotient of f(b) minus f(a) by b minus a. Geometrically, some tangent to the graph is parallel to the secant "
   "joining the endpoints. The theorem follows from Rolle's theorem applied to f minus that secant line.",
 "legal": "The indemnifying party shall defend, indemnify and hold harmless the indemnified party against any claim "
   "arising out of or relating to a breach of the representations set forth in Section 4, provided that the "
   "indemnified party gives prompt written notice of the claim and does not settle it without consent. Nothing in "
   "this Section limits either party's liability for fraud or for death or personal injury caused by negligence.",
 "dialogue": "\"You told them it was finished.\"\n\"I told them it would be finished.\"\n\"That is not the same "
   "thing, and you knew it when you said it.\"\nShe set the cup down without drinking from it. \"I knew it was the "
   "only sentence that would get us another week. If you have a better one I will use it on Thursday.\"\n",
 "chinese": "深度学习模型的推理速度取决于内存带宽和计算能力的平衡。对于大型语言模型，解码阶段通常受内存带宽限制，"
   "因为每生成一个词元都需要读取全部权重；而预填充阶段则主要受计算能力限制，因为可以同时处理大量词元。"
   "因此，量化权重对解码的加速效果通常比对预填充更明显。",
 "json": "{\"name\": \"glm53-spark\", \"ranks\": 2, \"window\": 1048576, \"kv\": \"fp8\", \"pages\": true, "
   "\"streams\": 4, \"drafter\": {\"kind\": \"dflash2\", \"bits\": 4}, \"reserve_gib\": 10.0, "
   "\"link\": {\"transport\": \"roce\", \"hcas\": [\"rocep1s0f1\", \"roceP2p1s0f1\"], \"gid\": 3}}",
 "medical": "Atrial fibrillation is characterised by disorganised electrical activity in the atria, producing an "
   "irregularly irregular pulse and loss of effective atrial contraction. Stasis in the left atrial appendage "
   "predisposes to thrombus formation, and anticoagulation is therefore considered according to an assessment of "
   "stroke risk weighed against the risk of bleeding.",
 "history": "The Treaty of Westphalia ended three decades of war in the Holy Roman Empire and is conventionally, if "
   "imprecisely, taken to mark the emergence of the sovereign territorial state as the basic unit of European "
   "politics. Its settlements on religious confession within the Empire mattered more to contemporaries than the "
   "abstractions later read into it.",
 "instructions": "To replace the cabin filter, open the glovebox and remove the two retaining clips at its hinge. "
   "Lower the box fully, then slide the filter housing cover to the left until it clears the tab. Withdraw the old "
   "filter noting the airflow arrow, insert the new one with the arrow pointing down, and refit the cover and clips.",
}

def logprobs(text):
    body=json.dumps({"model":MODEL,"prompt":text,"max_tokens":1,"temperature":0,"prompt_logprobs":0}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/completions",body,
        {"Content-Type":"application/json"}),timeout=900))
    out=[]
    for slot in r["choices"][0].get("prompt_logprobs") or []:
        if not slot: continue
        if len(slot)!=1: raise SystemExit(f"expected 1 logprob a slot, got {len(slot)}")
        tid,v=next(iter(slot.items()))
        out.append((int(tid), v["logprob"] if isinstance(v,dict) else float(v)))
    return out

label=sys.argv[1] if len(sys.argv)>1 else "config"
res={}
for name,text in CORPUS.items():
    res[name]=logprobs(text)
n=sum(len(v) for v in res.values())
tot=sum(lp for v in res.values() for _,lp in v)
json.dump({"label":label,"model":MODEL,"per_passage":{k:[[t,l] for t,l in v] for k,v in res.items()}},
          open(f"nll-{label}.json","w"))
print(f"  {label}: {n} tokens, NLL {-tot/n:.5f}")
