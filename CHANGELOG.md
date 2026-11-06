# Changelog

- **2026-11-06** -- `WITH_VIDEO=1`, opt-in, ported from the 2x DGX Spark recipe (its issue 8): `docker/Dockerfile`
  builds `libtfvideo.so` over upstream's static FFmpeg 9.0.2 into its own image tag (the default image and its tag are
  unchanged), so `VISION=1` serves `video_url` parts instead of HTTP 500. README: the Ablit weights drop a digit when
  copying an id into schema-constrained JSON where the published weights do not (that recipe's issue 9), with the
  workaround (`ABLIT=0`, or thinking on for those requests).
- **2026-11-04** -- CREDITS and NOTICE, the Aevonix entry corrected. The four serving scripts are that recipe's rework, which
  stands - but the entry now records where Aevonix's own scripts came from: their NOTICE takes `scripts/config.sh` and
  `start-tp4.sh` of this lab's GLM-5.3-Flash-EXL3-4x-DGX-Sparks-TensorFold (created 2026-10-01, the day before Aevonix's
  repository) as "the basis" of their `start.sh` and `scripts/config.sh`, and their `stop.sh` and `scripts/prepare.sh` as
  "adapted from Mia's scripts". `scripts/banner.sh` was never theirs: the two-Spark and four-Spark recipes carry the same
  MIA AI LAB banner file and this recipe recolours it, while Aevonix's banner is their own artwork. And patch 0073, whose K12
  loop patch 0078 follows, is `patches/aevonix/0073` in Aevonix's repository - the old cite to "MiaAI-Lab/Aevonix" names a
  repository that does not exist.
- **2026-11-04** -- Engine patch `0098`: a tool parameter whose schema names no type (`{}`, a description alone, a `$ref`
  that resolves nowhere) takes the JSON object or array the model wrote instead of coming back as a JSON string of it;
  scalars, plain words and quoted strings stay text, and a tool declared without a parameter schema keeps streaming its
  arguments as before. The rule is the one agreed on the 2x DGX Sparks recipe's issue #180 (upstream keeps untyped
  values as text; noted at the gate). Ported from the GB10 pair's recipe
  (MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104, its `0105`), where the bug was reproduced live and the patch host-tested;
  on this stack `zig build test-tool-parse-host test-tool-visibility-http-host test-json-answer-host
  check-native-cuda-host` pass (79 tests). Takes effect at the next `./start.sh restart` (new image).
- **2026-11-04** -- Engine patch `0097`, warm turns for GLM: after a reply, the server prefills the next turn's
  prompt in the background (`warm.zig`, a prompt-only pass), so a user message after an agent loop resumes past the
  loop. Before, GLM's template dropped the loop's reasoning once that message came, the prompt changed at the loop's
  first turn, and the whole previous turn was read again (measured: 4,096 of 9,335 tokens cached on a small loop;
  10-30 s on a 100K turn). The engine had the mechanism for the Nemotron family only; GLM now declares `warm_quiet`,
  its chunk gate yields a background pass to real work, the host takes the family's value, a tool-call reply warms
  the tool turn's opening instead of a user turn's, and the warm message carries the reply's reasoning. Codex found
  that a request admitted into the fill queue ahead of a warm pass was invisible to its yield checks, fixed in the
  shared lane host with a regression test. Analysed by a council of three (Claude, Codex, GLM on this pair; records in
  `docs/council-2026-11-04/`). Each pass's outcome is logged in rank0.log (`next turn: background prefill of N
  tokens done` / `stepped aside: other work came first` / `stepped aside` mid-pass / `failed: ...`). Verified live
  2026-11-04 on image `e3e480df52c9`: `scripts/cache-replay.py` reports the next user message cached 8,192 of 9,335
  (was 4,096), rank0.log shows the pass after the answer (`9322 tokens, resumed at 4096, kept 1`), and a request
  sent 0.3 s after an answer waits 0.3 s for the pass's chunk, one sent 2 s after waits nothing. A first replay that
  overlapped a 138K-token agent request showed no pass at all: a decoding or filling stream blocks the pass by
  design, so measure on an idle engine. Image `c2588984442f` adds only the log line. Expect more kept states per chat: raise
  `TF_GLM_HOST_CACHE_GIB` (export it in `scripts/local.sh`) if the hit rate drops.

- **2026-11-04** -- Engine patch `0096`: a tool call that follows prose quoting tool-call markup ("Qwen's
  `<tool_call>{}`") is parsed as the call. Before, the parser took the earliest `<tool_call>` and the first
  `</tool_call>` after it, so the quoted opener began the envelope, the real call did not parse as its payload, and the
  reply came back as content with no `tool_calls`: the agent's turn ended with the raw call on screen (rank log:
  `warning: kept 1 malformed or unoffered tool envelopes as content`; 1 of 203 requests on 11-04, 7 of 834 on 11-03).
  The parser now retries a rejected envelope from the last opener inside it (up to four back, for a value that quotes
  the opener itself); the quoting prose stays content, and a call that still does not parse, or names an unoffered
  tool, stays content as before. Host tests: `zig build test-tool-parse-host` (41 pass, two new),
  `test-tool-visibility-http-host`, `test-json-answer-host`. Not a template bug: the call the model wrote matched the
  checkpoint's `chat_template.jinja` exactly. Takes effect at the next `./start.sh restart` (new image).

- **2026-11-03** -- `PREFILL_SHARE` (default `0.68`, passed to the ranks as `TENSORFOLD_PREFILL_SHARE`): a stream
  decoding beside another request's long prefill no longer crawls. A long prompt is read in atomic 2,048-row chunks,
  and at the engine's default share of 1 the other streams got one decode round per chunk, about 2-5 tok/s with a
  pause of ~390 ms before every token. At `0.68` each chunk owes the decoders about ten rounds first. Measured here with
  one streamed chat running when a cold 175K-token prompt arrives: the chat's rate during that prefill 2.3 -> 27.2
  events/s (median gap 389 -> 11 ms; the worst gap stays one chunk, ~440 ms), the long prompt's first token 33.4 -> 49.8
  s. Replies are byte-identical at any value: the share only moves when the next chunk runs. `PREFILL_BUDGET` (rows
  filled per visit while a stream waits; empty) is passed as `TENSORFOLD_PREFILL_BUDGET` when set. The same symptom
  on the DeepSeek 4x Sparks recipe needed its own engine patches (`0031`/`0032`, `TF_DS_PREFILL_YIELD=1`); this
  engine pin already interleaves chunks, so no patch here ([README](README.md#prefill-beside-decoding)).
