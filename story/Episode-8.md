# Episode 8

## 1. Context for This Episode
- This episode focused on stabilizing live `narada start` behavior on Windows, especially shutdown reliability under Ctrl+C, backlog drain semantics, and user-facing runtime status clarity.
- The rough goal was to make `mic` and `system` live sessions predictable: first Ctrl+C should enter drain mode cleanly, pending ASR should commit before exit, second signal should still force-exit, and noisy native abort behavior should be minimized.

## 2. Main Problems We Faced

### 1) Intermittent non-deterministic Ctrl+C shutdown
- Symptoms: user saw runs where one Ctrl+C behaved correctly, but other runs exited quickly with little/no new commits despite visible backlog (`asr` often >10s).
- Symptoms: drain notice was sometimes missing entirely, and shell returned immediately.
- Why this was a problem: users could not trust that spoken audio would be flushed to transcript before exit.

### 2) Native Windows abort path (`forrtl` / access violation) bypassing graceful drain
- Symptoms: `forrtl: error (200): program aborting due to control-C event`.
- Symptoms: `$LASTEXITCODE` reported as `-1073741819` in one failing run.
- Why this was a problem: process could terminate before Python-level drain/commit finalization, causing data-loss perception and unstable UX.

### 3) Live status UI instability and line corruption during shutdown
- Symptoms: status printed many lines instead of one updating line.
- Symptoms: drain suffix text got concatenated/repeated on the same output stream.
- Symptoms: backlog warnings appeared interleaved with status text.
- Why this was a problem: users could not clearly read runtime state transitions or trust what stage shutdown was in.

### 4) Capture elapsed and shutdown-state communication confusion
- Symptoms: recording timer kept moving after first Ctrl+C in earlier behavior.
- Symptoms: no explicit one-time "entered drain mode" indicator in some paths.
- Why this was a problem: users could not tell whether capture had truly paused and backlog draining had started.

### 5) Mixed-mode quality gap
- Symptoms: user reported `--mode mixed` output mostly reflected system audio and missed mic contribution.
- Why this was a problem: mixed mode behavior did not match user expectations for reliable dual-source capture quality.

### 6) Product-path confusion around model distribution
- Symptoms: user asked whether models can be used without manual local downloads and where files are stored.
- Why this was a problem: deployment/UX decisions (local-first vs hosted inference vs auto-download) affect onboarding and trust boundaries.

## 3. Debugging Path & Options Considered

### Issue 1: Intermittent Ctrl+C shutdown behavior
- Debugging steps: traced `_ShutdownSignalController`, `_install_start_signal_handlers`, and `_run_tty_notes_first` interrupt/finalization ordering in `src/narada/cli.py`.
- Debugging steps: compared first-signal transition path against finalization joins and queue drains.
- Helpful discovery: duplicate-signal accounting had already been improved, but abrupt exits were still possible due to native-path interruption.
- Options considered: signal dedupe windows, first-signal ordering changes, broader retry wrappers, and deterministic one-time transition guards.
- Dead end: treating this only as signal-counting logic was insufficient for the remaining crash-like exits.

### Issue 2: Native abort path on Windows
- Debugging steps: inspected capture read/close interaction in `src/narada/audio/capture.py` and `src/narada/audio/backends/windows.py`.
- Debugging steps: correlated immediate exits and `-1073741819` with potential close/read races during first interrupt.
- Helpful discovery: immediate `close()` on first Ctrl+C could race with blocking `read_frame()` in capture worker threads.
- Options considered: stronger worker Ctrl suppression, stderr suppression only, immediate close on interrupt, delayed close after joins.
- Dead end: worker-only hardening could not fully explain main-process capture crash signatures.

### Issue 3: UI instability
- Debugging steps: reviewed `_LiveStatusRenderer` behavior and warning emission sites in `cli.py`.
- Helpful discovery: long status text and non-status warning lines could splice with active inline rendering.
- Options considered: strict single-line rewrite path, explicit line breaks before warnings, compact drain `eta` field, non-ANSI newline fallback.
- Dead end: relying on ANSI clear alone did not guarantee stable rendering in all console situations.

### Issue 4: Capture/drain state clarity
- Debugging steps: reviewed first-interrupt handling and capture clock freeze behavior in notes-first runtime.
- Helpful discovery: explicit one-time drain message and pause semantics needed to happen before any possible forced-exit checks for clarity.
- Options considered: emit transition notice at first signal, freeze elapsed clock, include paused/draining state marker.
- Dead end: passive status-only indication was not enough for user confidence.

### Issue 5: Mixed-mode gap
- Debugging steps: inspected mixed ingestion in `cli.py` and blend logic in `audio/mixer.py`.
- Helpful discovery: mixed mode is a software blend/paired-ingest path, not dual-track independent transcription.
- Options considered: keep mixed as-is, deprecate/remove mixed, prefer separate `mic` and `system` runs.
- Dead end: expecting blend mode to behave like independent dual-speaker diarized capture.

### Issue 6: Model download UX
- Debugging steps: reviewed README paths and model discovery behavior.
- Helpful discovery: current model files are local cache/app-data paths, not repo paths.
- Options considered: hosted/cloud inference, LAN inference node, bundled model, auto-download on first run.
- Dead end: "no local model download at all" is impossible without moving inference off-device.

## 4. Final Solution Used (For This Chat)

### Issue 1 + Issue 2: Deterministic shutdown hardening and native-race avoidance
- Actual fix: decoupled capture stop from native capture close in notes-first runtime.
- Conceptual change: first interrupt now requests capture stop immediately but defers native handle close until capture threads have been joined.
- Conceptual change: if capture workers are still alive after join timeout, code warns and skips immediate close to avoid unsafe close/read race.
- Files/layers involved: `src/narada/cli.py` (`_run_tty_notes_first`, interrupt/finalization helpers).

### Issue 1 + Issue 4: First-signal transition consistency
- Actual fix: first Ctrl+C path now consistently performs transition actions before forced-exit checks: note interrupt, mark drain, request capture stop, emit one-time drain notice.
- Conceptual change: transition became idempotent and explicit for user visibility.
- Files/layers involved: `src/narada/cli.py` (interrupt handling in notes-first path).

### Issue 2 + Windows parity: SIGBREAK handling
- Actual fix: start signal handler registration now includes `SIGBREAK` when available and maps it through Ctrl+C semantics.
- Conceptual change: Windows console interrupt handling parity improved in main runtime signal path.
- Files/layers involved: `src/narada/cli.py` (`_install_start_signal_handlers`).

### Issue 3: Regression coverage and behavior validation
- Actual fix: added tests to lock in shutdown and signal behavior rather than relying on ad-hoc manual runs.
- Tests added:
- `tests/test_start_integration.py`: race-aware capture fixture and regression for "single Ctrl+C during active read should not close during read."
- `tests/test_cli_live_runtime.py`: SIGBREAK registration and mapping tests.

### Validation outcome in this chat
- Quality gates were run and passed in order: `compileall`, `ruff format --check`, `ruff check`, `mypy src`, `pytest -q`.
- Commits were created and pushed.
- User confirmed: `mic` and `system` now work and drain behavior is fine.
- Remaining unresolved product decision: mixed mode behavior and whether to remove it.

## 5. Tools, APIs, and Concepts Used
- `Typer` CLI signal flow: used to reason about `KeyboardInterrupt` and `typer.Exit` semantics for first/second signal behavior.
- Python `signal` API (`SIGINT`, `SIGTERM`, `SIGBREAK`): used to normalize interrupt handling across platforms, especially Windows.
- Threaded capture pipeline (`threading`, `queue`): key to diagnosing stop/close ordering and race risk.
- Notes-first runtime architecture: capture -> planner/spool -> ASR queue -> result drain -> commit; central to shutdown correctness.
- Windows audio backend behavior: important for understanding blocking reads and native close race characteristics.
- Integration tests (`pytest`): used to encode regressions for signal races and shutdown transitions.
- Static quality gates (`ruff`, `mypy`, `compileall`): ensured style/type/syntax integrity while changing runtime shutdown paths.
- Runtime diagnostics (`asr`, `rtf`, `taskq`, `pend`, drain summary): used to interpret whether backlog was truly being processed.

## 6. Lessons Learned (For This Episode)
- Graceful shutdown in threaded native-I/O pipelines is mostly an ordering problem, not just a signal-counting problem.
- On Windows, one Ctrl+C can still hit native behavior; design shutdown so critical safety does not depend on immediate native handle closure.
- "Stop capture" and "close capture resources" should be separate phases when reads may be in-flight.
- One-time explicit transition logs dramatically improve operator trust during shutdown.
- Regression tests should model timing races directly; deterministic test fixtures are better than relying only on manual repro.
- A mode that conceptually blends streams is not equivalent to dual-track transcription; user expectation needs product-level alignment.
- Cross-platform signal parity (`SIGBREAK` included) prevents hidden platform-specific behavior gaps.
- Passing full quality gates after runtime fixes is essential because shutdown fixes can subtly affect many code paths.
