# Changelog

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
