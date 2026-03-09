# Episode #5

## 1. Context for This Episode
- This episode focused on the live transcription runtime behavior after the notes-first pipeline work was already in place.
- The goal was to make shutdown and terminal UX reliable and predictable: on stop/error, finish ASR first (including tail work), clearly communicate progress, and keep status output clean and consistent.

## 2. Main Problems We Faced

### Issue: Tail Pass Could Be Missed on Stop
- Symptoms seen:
  - User report: tail pass did not transcribe after `Ctrl+C`.
  - In-chat finding: shutdown ordering could signal ASR stop too early, risking skipped pending tasks/tail.
- Why it mattered:
  - Notes completeness was at risk exactly at meeting end, which is the most critical moment for trust.

### Issue: Live CLI Output Was Noisy and Hard to Read
- Symptoms seen:
  - Repeated `REC ...` status lines.
  - ASR backlog warning printed as extra lines instead of updating in place.
  - Request for a clean break after `drop=#`.
- Why it mattered:
  - Terminal readability degraded during long sessions and made warnings feel like clutter instead of actionable status.

### Issue: Perceived “ASR Stall” at Small Backlog Values
- Symptoms seen:
  - User observed ASR appearing stuck around `2.9s`.
  - Question about whether there is a minimum amount before transcription runs.
- Why it mattered:
  - It looked like a failure, but it was actually interval scheduling behavior; this gap can confuse operators.

### Issue: Stop Controls and Background Operation Expectations
- Symptoms seen:
  - User asked for stop options besides `Ctrl+C`.
  - User asked whether the tool can run in background and how to stop it.
- Why it mattered:
  - Operational usability depends on clear control surfaces, especially for unattended/long-running sessions.

### Issue: Runtime Metric Meanings Were Not Obvious
- Symptoms seen:
  - User asked what `rtf`, `commit`, `asr`, `cap`, and `drop` mean.
- Why it mattered:
  - Without clear semantics, users cannot correctly diagnose whether latency is capture-bound, ASR-bound, or data-loss related.

## 3. Debugging Path & Options Considered

### Tail Pass Could Be Missed on Stop
- Debugging path:
  - Reviewed live path in `src/narada/cli.py` (`_run_tty_notes_first` and finalization logic).
  - Identified ordering concern around stop signaling vs. queued/final tasks.
  - Validated with integration-focused tests around interrupt/final tail behavior.
- Options discussed:
  - `Option A`: stop immediately on `Ctrl+C` (rejected for notes completeness).
  - `Option B`: stop capture but keep ASR draining until pending work is done (chosen).
- Dead end vs. helpful discovery:
  - Dead end: treating `Ctrl+C` as immediate process stop.
  - Helpful: sentinel-driven worker drain is safer for final task completion.

### Live CLI Output Was Noisy and Hard to Read
- Debugging path:
  - Reviewed existing status printing cadence and warning emission.
  - Mapped where backlog warnings were emitted as separate lines.
- Options discussed:
  - `Option A`: keep one-line carriage-return status plus separate warnings (rejected).
  - `Option B`: render a fixed status block and refresh it in place, including warning state (chosen).
- Dead end vs. helpful discovery:
  - Dead end: continuing separate warning lines in high-frequency loops.
  - Helpful: a small status renderer gives consistent UX without changing ASR logic.

### Perceived “ASR Stall” at Small Backlog Values
- Debugging path:
  - Confirmed interval planner defaults and behavior (windowing + tail pass).
  - Explained that sub-window tail audio can wait until enough audio arrives or finalization occurs.
- Options discussed:
  - `Option A`: keep defaults and explain behavior (used in this chat).
  - `Option B`: lower interval for faster apparent responsiveness (discussed, not implemented here).
- Dead end vs. helpful discovery:
  - Dead end: interpreting small residual backlog as hard stall.
  - Helpful: understanding interval threshold + final tail semantics.

### Stop Controls and Background Operation Expectations
- Debugging path:
  - Checked current control flow and documented that TTY stop path is `Ctrl+C`-centric.
- Options discussed:
  - `Option A`: rely on OS-level background process management for now (current behavior).
  - `Option B`: add explicit managed background start/stop commands (not implemented in this chat).
- Dead end vs. helpful discovery:
  - Dead end: assuming there is already an alternate in-app stop button.
  - Helpful: clearly separating current capability vs. future managed-control feature.

### Runtime Metric Meanings Were Not Obvious
- Debugging path:
  - Read metric collection/formatting paths and explained each status field with current semantics.
- Options discussed:
  - `Option A`: leave as internal-only metrics (rejected in practice; user needs operational clarity).
  - `Option B`: provide explicit definitions tied to runtime behavior (used).
- Dead end vs. helpful discovery:
  - Dead end: shorthand labels without explanation.
  - Helpful: mapping each metric to its computation source improved diagnosis.

## 4. Final Solution Used (For This Chat)

### Issue: Tail Pass Could Be Missed on Stop
- Actual fix/decision:
  - Implemented ASR-first shutdown: on `Ctrl+C`/stop phase, capture stops first, then ASR pending work and tail tasks drain before exit.
  - Added shutdown progress messaging with ETA based on `asr_backlog * rtf`.
- Files/layers involved:
  - `src/narada/cli.py`
  - Test coverage in `tests/test_start_integration.py`
- Conceptual change:
  - Stop became a graceful draining phase instead of immediate tear-down.

### Issue: Live CLI Output Was Noisy and Hard to Read
- Actual fix/decision:
  - Implemented in-place refreshed status block after startup/QR output.
  - Moved ASR backlog warning into status updates instead of printing extra warning lines.
  - Status block now naturally places `drop=#` on its own line context (multi-line block).
- Files/layers involved:
  - `src/narada/cli.py`
  - Test coverage in `tests/test_cli_live_runtime.py` and `tests/test_start_integration.py`
- Conceptual change:
  - Transition from append-style terminal logging to persistent status rendering.

### Issue: Perceived “ASR Stall” at Small Backlog Values
- Actual fix/decision:
  - Clarified behavior: interval scheduling can defer short tails until enough audio or shutdown tail pass.
  - No performance/window-size code change was made in this episode.
- Files/layers involved:
  - Operational explanation based on current planner/runtime behavior.

### Issue: Stop Controls and Background Operation Expectations
- Actual fix/decision:
  - No new background command/control feature implemented in this episode.
  - Current behavior documented in chat: can run as background OS process, but graceful stop semantics are still centered on managed shutdown flow.
- Files/layers involved:
  - No code added specifically for background command lifecycle in this episode.

### Issue: Runtime Metric Meanings Were Not Obvious
- Actual fix/decision:
  - Provided explicit operator-facing definitions for `rtf`, `commit`, `asr`, `cap`, `drop`.
  - Kept metric logic unchanged.
- Files/layers involved:
  - Explanation based on runtime/performance paths already present.

## 5. Tools, APIs, and Concepts Used
- `Typer CLI`: command flow and live terminal output handling.
- `threading` + `queue`: separate capture and ASR work, and safe shutdown draining.
- `SessionSpool` / interval tasks: persisted audio ranges and deferred/tail transcription coverage.
- `RuntimePerformance`: source of `rtf`, commit latency, capture backlog, ASR backlog, drops.
- `StabilizedConfidenceGate`: commit gating with holdback/dedupe/final drain behavior.
- `pytest` integration/unit tests: validated shutdown behavior, warning semantics, and status helpers.
- `ruff` + `compileall`: style/lint and syntax validation.
- `whisper-cpp` / `faster-whisper` runtime context: discussed consistency across engines and operator expectations.

## 6. Lessons Learned (For This Episode)
- Graceful shutdown must prioritize data completion (tail drain) over immediate process exit.
- Real-time UX should use a stable status surface; append-style warning spam hides signal in noise.
- Backlog metrics are only useful if users understand what they measure and when they advance.
- Small residual ASR backlog is not always a stall; windowing strategy can intentionally defer short tails.
- Stop semantics should be explicit and user-visible during drain, including an ETA expectation.
- Cross-mode consistency matters: operator behavior should feel the same regardless of engine/mode.
- Tests should assert shutdown guarantees, not just steady-state transcription.
