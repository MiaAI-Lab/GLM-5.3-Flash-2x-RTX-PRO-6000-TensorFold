# Council of three, 2026-11-04: a user message after an agent loop re-reads the loop

Question: why does the next user message after a tool loop prefill the whole loop again, and what is the exact fix?
Members: Claude (Fable 5.1, wrote the brief and the patch), GLM-5.3-Flash served on this pair (`glm.md`, 43,879
prompt tokens of brief plus engine source, 95 s), Codex (`codex.md`, read-only on the patched engine tree).

Outcome: engine patch `0097`. The engine already had warm turns (`server/warm.zig`: a background prompt-only pass
of the next turn's rendering after each reply) for the Nemotron family; GLM declared no `warm_quiet`, so they were
off. GLM confirmed the TP2, snapshot, drafter and page paths need no new protocol and that the marks path (not
`keepBorn`) keeps the pass's states. Codex found that a foreground request admitted into the fill queue was invisible
to the pass's yield checks (`lane_host.zig`), fixed with a regression test. Both rejected template-level tricks and
inexact KV edits.

Verified live 2026-11-04 after the restart (image `e3e480df52c9`): `scripts/cache-replay.py` reports the next user
message cached 8,192 of 9,335 (was 4,096); rank0.log shows the pass after the answer (`9322 tokens, resumed at 4096,
kept 1`); a request sent 0.3 s after an answer waited 0.3 s (one chunk) and one sent 2 s after waited nothing, on
both ranks. A first replay that overlapped a 138K-token agent request showed no pass: another stream decoding or
filling blocks the pass by design. Remaining follow-ups: the warm prompt being shorter than the reply's prompt
(`born` is eviction metadata only, harmless), a pass that runs as a fill (a chunk a round) beside a decoding stream
instead of waiting for a quiet engine (matters with two agents on the pair), and keeping the cleared rendering warm
during the loop instead of after the final answer.
