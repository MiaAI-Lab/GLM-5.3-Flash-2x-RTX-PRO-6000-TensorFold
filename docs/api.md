# The HTTP API for scoring runs (sparkDash)

TensorFold speaks the OpenAI HTTP API. This page documents the parts the Fidelity benchmark uses:
per-token logprobs on `POST /v1/chat/completions`' sibling `POST /v1/completions`, the build
identity on `GET /v1/models`, and the flags that make runs reproducible. Everything here is
verified against the live RTX PRO 6000 server (`GLM-5.3-Flash`, TensorFold `f350cd3e` + the
recipe's patch stack) on 2026-11-05.

## Prompt logprobs (`prompt_logprobs`)

`POST /v1/completions` with `prompt_logprobs: K` returns, in `choices[0].prompt_logprobs`, one
entry per prompt token: position 0 is `null` (no conditioning precedes it), every later position is
a map of token id to `{logprob, rank, decoded_token}` holding the top-K tokens plus the position's
own token with its exact logprob and rank, even when that token sits outside the top-K.

```sh
curl -s http://127.0.0.1:8888/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "The quick brown fox jumps over the lazy dog.",
  "max_tokens": 1, "temperature": 0, "prompt_logprobs": 2
}'
```

```json
{
  "id": "cmpl-...", "object": "text_completion", "created": 1793880729, "model": "GLM-5.3-Flash",
  "choices": [{
    "index": 0, "text": " The", "finish_reason": "length", "logprobs": null,
    "prompt_token_ids": [153963, 3974, ...],
    "prompt_logprobs": [
      null,
      {"154822": {"logprob": -0.3166, "rank": 1, "decoded_token": "[gMASK]"},
       "1815":    {"logprob": -7.5978, "rank": 2, "decoded_token": " path"},
       "3974":    {"logprob": -10.3478, "rank": 660, "decoded_token": " quick"}},
      ...
    ]
  }],
  "usage": {"prompt_tokens": 11, ...},
  "prompt_logprobs_missing": [],
  "exact_mode": "exact",
  "tensorfold": { "...the runtime block, as before..." }
}
```

The fields, one by one:

- `prompt_logprobs` — an integer, 0 through the server's cap (128 by default, `--prompt-logprobs-cap`
  at launch, reported by `GET /v1/models`). The cap's ceiling is the row layout's own 128.
- `prompt_token_ids` — present whenever `prompt_logprobs` was asked: the exact ids the model scored,
`  `usage.prompt_tokens` of them. Compare it between the two models of a comparison before trusting
  any logprob: different tokenisation invalidates the pairing.
- `prompt_logprobs_missing` — top-level, present whenever `prompt_logprobs` was asked. The positions
  whose token has no score (an empty list normally). A missing position is `null` in the entry array
  and named here; the server never drops or invents entries.
- `logprobs` (generated-token logprobs) and `echo` combine freely with `prompt_logprobs`:
  `{"prompt_logprobs": 50, "logprobs": 5, "echo": true}` is one 200. `echo: true` prepends the prompt
  to `choices[0].text` (a text prompt verbatim; token-id prompts decoded), and in streaming it is the
  first chunk's text. On a backend without reply-row support, `logprobs` alone is refused with a 400
  that names it, but beside a `prompt_logprobs` request it is ignored so the call survives.

### Requested ids: `prompt_logprobs_token_ids`

Alongside the top-K, the server scores caller-chosen ids at every position (vLLM's
`token_ids_logprob`). Two shapes:

```jsonc
{"prompt_logprobs": 5, "prompt_logprobs_token_ids": [3974, 154822]}       // the same ids everywhere
{"prompt_logprobs": 5, "prompt_logprobs_token_ids": [[], [3974], [154822]]} // per position, one list
                                                                          // per prompt token
```

Each id's `{logprob, rank, decoded_token}` lands in that position's map (the ranks use the same rule
as the token's own). A flat list may carry at most 128 ids; a per-position list at most 128 per row,
and there must be exactly one list per prompt position (row 0 is accepted and unused — nothing
conditions position 0). The ids must be inside the vocabulary. This is what scores two models on a
common support: send the reference's per-position top-K, and the bucketed KL bound follows.

### Compact entries: `prompt_logprobs_format`

`"prompt_logprobs_format": "compact"` replaces each map with
`{"ids": [...], "logprobs": [...], "actual": {"id", "logprob", "rank"}}` — the token first (when it
has a score), then the best, then the requested ids. The compact rows carry no per-id ranks; the map
form (the default, `"vllm"`) does. On a 32k-token prompt the compact form is roughly five times
smaller.

```json
{"ids": [154822, 1815, 3974], "logprobs": [-0.3166, -7.5978, -10.3478],
 "actual": {"id": 3974, "logprob": -10.3478, "rank": 660}}
```

### Guarantees

- Every position of the prompt is scored in the same pass that prefills it. A request with
  `prompt_logprobs` never serves entries from the prefix cache: the engine re-runs the prompt from
  row 0, so a cached prefix cannot stale or skip a row. The same prompt sent twice returns identical
  entries, cached or not.
- The rows are exact: fp32 log-probabilities under the raw bf16 logits, the rank the count of tokens
  at or above the value, computed by a fixed reduction tree (no run-to-run or batch-composition
  variation).
- Prompts of 8k and 32k tokens were verified in one request: one entry per position,
  `prompt_token_ids.len == usage.prompt_tokens`, `prompt_logprobs_missing` empty.

## Build and tokenizer identity: `GET /v1/models`

Each model carries a `tensorfold` object beside the standard fields (all the previous keys remain):

```json
{"object": "list", "data": [{
  "id": "GLM-5.3-Flash", "object": "model", "created": 1793880351, "owned_by": "tensorfold",
  "context_length": 1001472, "max_model_len": 1001472,
  "reasoning": {"default_enabled": false, "default_effort": "high", "supported_efforts": ["max", "high", "low"]},
  "tensorfold": {
    "quantization": {"bits": 4, "...": "the checkpoint's own quantization_config, verbatim"},
    "compute_dtype": "bfloat16",
    "rope_theta": 1000000, "rope_scaling": { "...as the checkpoint states, or absent..." },
    "checkpoint": "/model", "checkpoint_sha256": "6e69...",
    "kv_cache_dtype": "fp8", "tensor_parallel": 2, "context": 1001472,
    "speculative": {"enabled": true, "kind": "DFlash2", "name": "incoai/GLM-5.3-Flash-DFlash2"},
    "tokenizer_sha256": "aa13...",
    "prompt_bos": true,
    "prompt_logprobs_cap": 128
  }
}]}
```

`checkpoint_sha256` is TensorFold's full checkpoint identity (config, index and every weight byte,
sha256); `tokenizer_sha256` is sha256 of `tokenizer.json`. `prompt_bos` says whether a raw
completions text prompt gets a token prepended. The engine's version and commit are on `/health`
(`version`, and `commit`/`patches` when the image recorded them). Pick `prompt_logprobs` values at
or below `prompt_logprobs_cap` without probing.

## Reproducible greedy runs: `exact_mode`

Every completion reply carries `"exact_mode": "exact"`. The request side that guarantees
batch-invariant greedy decoding is **`"temperature": 0`** (or its absence with a greedy server
default): the request then runs the engine's exact lane — no sampling, no seed, the same tokens in
any batch composition, verified by the recipe's ABBA reply-hash runs. Drafted verification is greedy
and exact-match, so drafts do not change the sequence. Set `temperature: 0` on every comparison run;
the dashboard records `exact_mode` and the reply's `tensorfold.token_sha256` and refuses to compare
hashes across different `tensorfold` build facts.

## Constraints and compatibility

- No existing field changed shape; everything above is additive. Chat completions need no logprob
  support and have none of these fields.
- `prompt_logprobs` is accepted on `POST /v1/completions` without streaming only, as before.
- Streamed requests, `chat_template_kwargs.enable_thinking` / `thinking_mode`, and
  `usage.completion_tokens_details.reasoning_tokens` are unchanged.
