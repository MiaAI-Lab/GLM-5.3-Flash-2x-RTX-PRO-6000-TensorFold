# Changelog

- **2026-11-12** -- The stack on main now matches what this host's deployment has served since 2026-11-05: engine
  patches 0094 (a prompt chunk takes 4096 rows) and 0095 (the drafter ingests a 4096-row chunk) are out — the running
  images dropped both (98 patches at `de97bf59db4d`, 100 at today's tree), so prefill stays at 2,048-row chunks.
  Patch 0107 (32-row decode windows) is rebased onto the stack without them: its `cuda_state.zig` hunk sat on 0094's
  `prefill_rows = 4096` context line, and nothing else in the stack needed a change (107 of the remaining patches
  apply untouched). Host suite green on the rebased engine tree (`.src/tf`, branch `deploy113`): 3,883 of 3,885, the
  two fails predate all of this (the draft-sources head check and the exl3 scan-fault nondeterminism).
- **2026-11-12** -- Engine patch 0113: a long chat no longer dies at its first turn with `PoolTooSmall`. The KV pool's
  pages are what the memory budget leaves, often fewer rows than the context window names, and a request whose prompt
  plus reply needed more pages than the whole pool holds was admitted only to be refused at its opening
  (`error.PoolTooSmall: shorten the prompt or lower max_tokens`) — with the server clamping the reply to the context
  window, no shorter `max_tokens` saved it, so every turn of such a chat failed the same way. Such a request now
  opens with the longest reply the pool holds for its prompt (its verify window and run included): the stream's reply
  is capped to match, one `kv pool: ... opens with N` line says so, and the reply ends at the length limit like any
  other long reply. Sharded (`TF_GLM_KV_SHARD=1`) the budget is one part's pages, the cap the same.
  `error.PoolTooSmall` remains for a prompt that can never fit the pool even alone, and its words now name
  `TENSORFOLD_MEMORY_RESERVE_GIB` and `--context` as the ways to give the pool more room. The patch also corrects two
  test constants patch 0107 (32-row decode windows) left stale, which had kept `test-glm-kv-pages-host` red:
  `capacity(330529)` is 330624, and the rank fingerprint's layout is 16. Built and tested without devices on the
  engine tree at `.src/tf` (branch `fidelity113`: `f350cd3e` plus all 110 patches), host suite 973 of 977 — the three
  fails predate this patch (the draft-sources head check and the exl3 scan-fault nondeterminism from the old tree,
  and the FP8 pages store's resume miss, present on the bare 0112 stack).
- **2026-11-12** -- Ported from the 2x DGX Spark recipe's tool-call-guard round. Not yet built on this host: all 109
  patches apply to `f350cd3e` with the Dockerfile's loop, and `scripts/tests/run.sh` passes.
  - Engine patch 0101, `TF_TOOL_CALL_GUARD=1` (opt-in, off by default; that recipe's 0107, ported here 2026-11-05
    and live on this host's deployment since 2026-11-07, image `5a91310bda8a`): a chat request that offers no tools
    although its history called them gets a closing system note and its call markup hidden; a request that offers
    tools has the envelopes the parser keeps as text hidden, streamed or whole, each logged.
  - Engine patch 0112 (that recipe's 0112, verbatim; its base is 0101's result here byte for byte): each rejected
    tool envelope is logged one by one — `tool call rejected from ID: REASON ['NAME'] (N tools offered)
    thinking= effort=; FATE`, reasons `unknown_tool_name`, `malformed` and `unclosed`, and `TF_TOOL_CALL_DEBUG=1`
    adds the envelope's text. With tools offered the guard hides only well-formed calls to unoffered tools: a
    malformed call stays as text the client can see, and a reply that hiding would leave with no text and no call
    keeps its calls — an agent never retries a blank reply blind until its timeout (the Hermes report that recipe's
    entry records, 25 replies of 436 with calls hidden, 11 left empty). The no-tools path is unchanged; guard off,
    replies are byte-identical apart from the log.
  - Engine patch 0111 (that recipe's 0111, verbatim): the clean image build compiles without a compiler warning —
    two fp8 prompt kernels' unread `RW`, prefill_attention's readerless `SMEM_MAX`, glm_mm's unused `Tile::SMEM`,
    and the flashnext_fact/flashnext_attn unroll loops moved behind an inverted `if constexpr`. On that recipe all
    six files' fatbins compiled byte-identical to the unpatched build.
  - Checked against that recipe's final tree: after this round, openai.zig, tool_guard.zig, tool_stream.zig,
    late_system_test.zig and five of the six kernel files are byte-identical to it; glm_mm.cu differs only by its
    GB10 stage kernels (that recipe's 0100), which this recipe does not carry. Noted on the way: that recipe's 0103
    (its re-port of this recipe's 0096) adds a guard against an opener inside a still-open value (`insideValue`,
    with a host test) that has not flowed back this way — a candidate for a later port.

- **2026-11-11** -- Ported from the 2x DGX Spark recipe's 2026-11-10/11 round (its main 74cf93c). Not yet run on
  this host: the patches apply to `f350cd3e` with the Dockerfile's loop and the scripts pass `scripts/tests/run.sh`.
  - Engine patches 0105-0107 from upstream TensorFold main, regenerated against this recipe's 0001-0104:
    `L2PF=1` (`TF_GLM_L2PF`, opt-in; on the Spark pair +2% at one stream, flat under load), a JSON `<tool_call>`
    that `max_tokens` cuts is returned as the reply's cut call (with two local fixes), and 32-row decode windows
    (upstream #2257, adapted: on the Spark pair a one-stream file rewrite decoded 151 to 201-205 tok/s, prose
    unchanged, token hashes identical). Upstream #1779's shared draft reservation is not taken: upstream turned it
    off for DFlash2 itself (cc2fa8d23).
  - Engine patch 0108, `EXTRA_EOS` (default `<|assistant|>`): a reply ends at `<|assistant|>`, which GLM-5.3's
    `generation_config.json` leaves out (thanks @wagzee). Engine patch 0109, `THINK_OFF_SPLIT`, **opt-in** (off by
    default; 1024 is the measured hold): a thinking-off draft closed by `</think>` goes to `reasoning_content`
    (thanks @wagzee).
  - `start.sh`: a one-stream startup line is read by the `PROMPT_CACHE_MIN_GIB` fallback, and `E_CACHE` names the
    log archive `stop.sh` wrote rather than the run directory it removed (thanks @laidick); a `CONTEXT=auto` start
    with KV windows no longer exits 1 after LIVE, and `W_STREAMS` is quiet for `PARALLEL=auto` and matches
    "1 stream at once"; `VISION=0` with `EFFORT_LINE=thinking` writes the template after the first download
    (thanks @travellingsoldier85).
  - `validate_config`: `KV_SHARD=1` also needs `PARALLEL` above 1, which patch 0104 refuses only after the serving
    container is stopped (thanks @travellingsoldier85).
  - `scripts/graft.py`, a partial Ablit overlay (thanks @mjjjjaazing), with a guard that refuses an output that
    resolves to either input.
  - Tests: `cache-min.sh` gains the one-stream and log-archive cases, `kv-shard-config.sh` (10 cases) and
    `graft.sh` are new; each fails without its fix.

- **2026-11-07** -- Ported from the 2x DGX Spark recipe's 2026-11-07 round, checked on this host (2x RTX PRO 6000,
  `PARALLEL=4`, `CONTEXT=auto`, DFlash2, `DENSE=q8 HEAD=q8`, a 2.8 GiB reserve, the published weights):
  - Engine patch `0103` (that recipe's 0109, upstream #1411 backported by Robert Wojciechowski): `LEARN_GIB`
    (`--learn-conversations`, off by default) keeps the prompt states on disk, every rank's part. After a restart an
    18.5k-token conversation resumed 16,384 tokens cached (first token 0.75 s) and a 101k-token one 98,304 (1.16 s),
    replies byte-identical to the in-memory resume.
  - Engine patch `0104` (that recipe's 0110, toward upstream #1431, one hunk moved by hand: this stack's
    shared-window forward has no TF_GLM_PROF bracket): `KV_SHARD=1` (`TF_GLM_KV_SHARD`, off by default) keeps each
    stream's latent rows on one GPU. The full 1,048,576-token window fits beside four streams (2,052 pages) where the
    same image unsharded fitted 702,464 (1,396 pages); greedy hashes identical; decode 169-170 tok/s at one stream
    against 171-172, 293-295 summed at four against 294-296; `tf-cuda-test glm-kv-shard-check` PASS on both GPUs
    (10 prompt paths, 510 window rows byte-identical to the unsharded twin).
  - What `KV_SHARD=1` buys on this host, against the deployment it replaced (the image before these patches, the same
    settings): the pool went from about 973K tokens to the full 1,048,576 (+8%), for about 3.5% decode at one stream
    (~176 to ~170 tok/s, both builds' costs together). The gain stops there because `CONTEXT=auto` sizes the pool for
    one window at most: the start line reads 2,052 pages (3.47 GiB) of the 5.0 GiB left after the 2.8 GiB reserve,
    3 x 417 MiB fixed for the streams beyond the engine's own, and the 0.3 GiB that remains goes to the prompt cache.
    A sharded page costs about 1.73 MiB a GPU (3,542 B a token), so each GiB freed would hold about 300K more
    tokens, which only helps several long conversations at once; one request still stops at 1,048,576. The 2x DGX
    Spark recipe, with 128 GB a GPU, goes from 2.1M to 3.15M tokens with the same patch.
  - With both off the image gives the same greedy hashes as the one before (prose `ed074e68e492`, code
    `1388e42283e7`, thinking `650453259cef`); decode 171-172 tok/s at one stream and 294-296 summed at four against
    174-179 and 300-314 (two runs each, another client's requests reached the server during the earlier ones).
  - `EFFORT_LINE=thinking` (default; that recipe's #23): the template's `Reasoning Effort: Max` line with thinking on
    only. Before, a thinking-off "binary of 217, digits only" at 64 tokens worked the division out and stopped on
    `length` here too. `EFFORT_LINE=always` keeps the old template.
  - `PROMPT_CACHE_MIN_GIB` (that recipe's issue 16): `start.sh` stops the server before LIVE when rank 0's prompt cache
    (device pages plus host tier, a rank) comes out under it.
  - An exported `NCCL_SOCKET_IFNAME` exclusion list (`^lo,docker,...`, that recipe's issue 17) is replaced by `lo` with
    a warning, since it would keep NCCL off the ranks' loopback link.
  - `scripts/tests/` (`run.sh`): script syntax, the patch-number check, and tests for the cache floor and the NCCL value.
  - README: the `SERVE_EXTRA_ARGS='--top-k 20'` sampler default for requests that send no `top_k`.
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
