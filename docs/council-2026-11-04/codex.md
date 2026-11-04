Enabling GLM warm turns is the right exact mechanism, **after fixing a scheduling hole in the shared host**. The TP, snapshot and page-release machinery already exists.

I inspected `8190a0ea`, labelled `f350cd3e` plus patches 0001–0093, including the existing uncommitted changes. Those changes already add GLM’s `warm_quiet` and yield check; the brief describes their earlier state.

**1. What GLM actually lacks**

The relevant paths are:

| Concern | What this tree does |
|---|---|
| Background prefill at TP2 | `Lanes.prefillFn` calls `Leader.prefill`, which broadcasts `.prefill`, IDs and resume metadata before both ranks execute `Engine.prefill`. The background flag stays in the host; the follower needs no separate background protocol. See [cuda_lanes.zig:218](zig/src/families/glm/cuda_lanes.zig:218), [cuda_follow.zig:243](zig/src/families/glm/cuda_follow.zig:243). |
| Yield and cancellation | The dirty `Lanes.cancelled` now checks `s.isCancelled() or s.yields()`. `Engine.gate` collectively reduces that boolean, so **both ranks return `Cancelled`**, rather than GLM returning `Yielded`. The follower’s `apply` explicitly accepts `Cancelled` and continues receiving steps. `warmPass` accepts either error. See [cuda_lanes.zig:209](zig/src/families/glm/cuda_lanes.zig:209), [cuda_engine.zig:521](zig/src/families/glm/cuda_engine.zig:521), [cuda_follow.zig:400](zig/src/families/glm/cuda_follow.zig:400). |
| Retaining partial progress | On cancellation at a chunk boundary, `prefillChunk` calls `snapshots.interrupted`. Rank 0 invokes `keepInterruptedPrefill(at)`; the additional snapshot gate publishes the saved state and evictions to followers **before releasing the sequence**. This is already implemented. See [cuda_engine.zig:471](zig/src/families/glm/cuda_engine.zig:471), [cuda_snapshots.zig:204](zig/src/families/glm/cuda_snapshots.zig:204). |
| DFlash2 context | Snapshots optionally include the bound slot’s valid context ring. Restore uses `resumeAt` when that ring exists, otherwise explicitly restarts the drafter at the mark. `prefillChunk` also performs ordinary prompt ingestion during warming. No new drafter lifecycle is required. See [cuda_snapshots.zig:87](zig/src/families/glm/cuda_snapshots.zig:87), [cuda_snapshots.zig:143](zig/src/families/glm/cuda_snapshots.zig:143). |
| Live KV ownership | Opening reserves pages for the whole prompt, reply allowance and verify window. Warming therefore still reserves substantial pages with `max_tokens=0`. On completion/cancellation, `releaseFn → Leader.release → Engine.release → Pages.close` returns them on both ranks. Snapshot copies live in separate pools and retain no live KV pages. See [cuda_pages.zig:176](zig/src/families/glm/cuda_pages.zig:176), [cuda_lanes.zig:702](zig/src/families/glm/cuda_lanes.zig:702). |

Three details need particular care.

First, **this tree’s chunk grid is 2,048 rows**, defined in [cuda_state.zig:19](zig/src/families/glm/cuda_state.zig:19). Foreground `Leader.chunk` can combine two chunks into a 4,096-row paired step. The warm path calls whole `Engine.prefill`, whose loop advances one 2,048-row chunk at a time. A measured 4,096-row step does not establish a 4,096-row snapshot grid.

Second, retain the existing grid and one-token lookahead. `cuda_snaps.cacheRules` defaults to `lookahead=1`; declaring `warm_quiet` additionally selects `.warm=.waits`. Warm marks target the grid boundary before `W−1`, where `W` is the warm prompt length. Thus the expected final retained mark is:

```text
K = floor((W − 1) / 2048) × 2048
```

provided the mark fits and survives eviction. The next request still processes the remaining 1–2,048 shared tokens plus its new content. The current `warmPass` deliberately skips its terminal `keepBorn` when lookahead is nonzero. Don’t introduce an arbitrary end snapshot: row arithmetic and follower publication both require attention. See [cuda_snaps.zig:6](zig/src/native/cuda_snaps.zig:6), [prompt_cache.zig:423](zig/src/core/prompt_cache.zig:423), [lane_host.zig:434](zig/src/core/lane_host.zig:434).

Third, `born` is eviction metadata, not the snapshot position or a substitute cache identity. Carrying the original request’s prompt length prevents its warm extension from automatically superseding that request’s own states. It cannot guarantee those states survive capacity eviction. Clearing reasoning may make `born > warm_prompt.len`; that case deserves coverage. See [prompt_cache.zig:459](zig/src/core/prompt_cache.zig:459), [prompt_cache_policy.zig:5](zig/src/core/prompt_cache_policy.zig:5).

The remaining scheduling hole is concrete:

- `quiet` and `yieldsTo` check queued foreground work and `core.activeCount()`.
- `activeCount()` counts `core.live`, which contains decoding streams.
- A foreground request waiting in `h.filling` has already left the queue and has not joined `core.live`.
- With a spare lane, admission can launch a warm pass before `fill.run`, letting the entire warm prompt block that foreground prefill.

The relevant functions are [lane_host.zig:211](zig/src/core/lane_host.zig:211), `launch:342`, `run:530`, and [lanes/engine.zig:89](zig/src/core/lanes/engine.zig:89). This is a source-level finding; I have not run its reproducer.

**2. Is there a better exact fix?**

Warming the cleared rendering preserves the intended semantics. Removing reasoning changes the causal inputs to every following layer and the KDA recurrence. Deleting KV rows, shifting positions or retaining reasoning-influenced suffix states cannot produce an exact cleared execution.

The current dirty server policy deserves a separate decision. [warm.zig:65](zig/src/server/warm.zig:65) now appends:

- A dummy **tool result** after a tool-call reply, retaining the current loop’s reasoning.
- A dummy **user message** after a plain reply, producing the cleared branch.

That helps the immediate continuation. **It prepares the cleared branch only after the final answer**, leaving the large replay to be hidden during the human’s pause.

An exact second-stage improvement is to maintain the cleared “next user” branch incrementally during tool-loop pauses. Each later cleared rendering can extend an earlier cleared snapshot, avoiding one large final replay. Keep the normal tool-continuation branch separately if useful. This requires bounded background scheduling and coalescing obsolete work; it is larger than the minimal enablement patch.

There is no guarantee that arbitrary templates make two probes a universal prefix. Existing token-prefix validation makes a mismatching prediction miss safely. Actual checkpoint-template tests should establish the useful prefix.

Also, this repository loads GLM’s template from checkpoint assets. Its existing `hf_text` test uses an older synthetic clearing rule, so it does **not** independently verify the brief’s last-user-specific Jinja behavior: [hf_text.zig:483](zig/src/server/hf_text.zig:483).

**3. Minimal patch and qualification**

The current uncommitted diff already supplies these pieces:

- GLM `warm_quiet=150 ms`: `cuda_native.zig:31`.
- GLM cancellation checks yielding: `cuda_lanes.zig:211`.
- Family-specific quiet duration passed through `native/cuda.zig:134,439`.
- Reply `reasoning_content` included in warming: `server/openai.zig:403`.
- Tool/user continuation selection: `server/warm.zig:65`.

I would add one shared, lock-held predicate used by both `quiet` and `yieldsTo`:

```zig
fn warmBlocked(h: *const LaneHost, warm: *const Job) bool {
    if (h.foregroundWaiting() or h.core.activeCount() > 0) return true;
    for (h.admitted.items) |other| {
        if (other != warm and !other.request.background) return true;
    }
    return false;
}
```

This covers admitted foreground prefills while excluding the warm job itself.

The first regression belongs in `lane_host_warm_test.zig`: use `Fake.stepped()`, enqueue a foreground request followed by a background warm request **before starting the host**, and allow two lanes. Assert that the warm pass runs zero chunks, finishes cancelled, and the foreground request completes correctly. Existing warm tests use `Fake.backend()` and therefore miss this condition: [lane_host_warm_test.zig:47](zig/src/core/lane_host_warm_test.zig:47).

Add two complementary checks:

- A warm interruption with a real store and GLM-style grid/lookahead rules: retain a completed boundary, resume from it, compare output with cache-off/plain execution, and verify release. The existing arrival test establishes delivery and cancellation, not retained-state correctness.
- An actual GLM-template tool-loop fixture: reasoning retained before a tool continuation, cleared before the next user, and warmed IDs matching the corresponding real request. The new `late_system_test.zig:553` uses a renderer that ignores reasoning and has thinking disabled.

Existing host coverage already checks cancellation publication, DFlash2 ring copies, FP8 layouts and host-tier restoration in `zig/glm_snapshot_host.zig`, particularly tests at lines 248, 657, 779 and 911. Those are useful foundations; they do not replace the TP2 served test.

For the serving pair, the material risks are:

- **Memory:** a warm pass reserves its entire live prompt’s pages. For the default full model, FP8 pages cost 6,380 bytes/token without the target MTP head, or 6,960 with it, plus fixed state. Around 100K tokens therefore occupies roughly 0.6–0.65 GiB/rank of live KV. Saved states consume the already allocated snapshot pools and may evict useful entries. See [cuda_state.zig:77](zig/src/families/glm/cuda_state.zig:77).
- **Stalls:** four decoding streams prevent warming; continuous load can prevent it indefinitely. With the admitted-prefill hole fixed, foreground arrival interrupts at a chunk gate. Delay still includes the current chunk, collective control, and potentially synchronous snapshot copies. It is not instant preemption.
- **Early user reply:** completed matching progress should survive cancellation. The foreground request then prefills the remainder and boundary tail. Cache refusal, eviction or a different rendered prefix can force repeated work; warming does not guarantee a hit.
- **Throughput:** whole warm prefill does not use paired prompt steps, so the reported foreground 4–5K tok/s should not automatically be assigned to it.

Before deploying, exercise completion and interruption on **TP2 + EXL3 + DFlash2 + paged FP8 + host snapshots**, compare warmed/resumed output against fresh plain output, and verify both ranks’ snapshot IDs and the final idle KV census. Include simultaneous foreground prefills as well as four decoding streams.

No files changed and no device tests ran. ATC and Zig are unavailable in this environment; the live checkpoint assets and serving pair were not inspected.