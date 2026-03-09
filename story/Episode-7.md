# Episode 7

## 1. Context for This Episode
- This episode focused on Narada runtime reliability in real usage: long-call ASR behavior, backlog/commit latency, and shutdown behavior under signals.
- The rough goal was to make sure `narada start` behaves safely and predictably: no dropped work, graceful drain on stop, and practical performance/reliability across CPU/GPU paths.

## 2. Main Problems We Faced

### 1) Long-window GPU instability in certain presets
- Symptoms:
  - Reported behavior: ~5 minute windows could pass on CUDA `fast`, but `balanced`/`accurate` hit the worker timeout and disabled GPU for the session.
  - Existing timeout context discussed: `_GPU_TRANSCRIBE_TIMEOUT_S = 12.0`.
- Why this mattered:
  - Throughput and stability regressed mid-session, with unexpected fallback to CPU and lag growth.
  - It conflicted with the expectation that compute `auto` + CUDA should stay robust.

### 2) Late/no visible commits during heavy backlog
- Symptoms:
  - Runtime showed large ASR lag (`asr=146.0s`) and very high `rtf`, while transcript commits were delayed.
  - User saw runs where little/no text was committed for long periods.
- Why this mattered:
  - User experience looked broken (“recording but not producing notes”), especially in long lectures/calls.
  - It made runtime status harder to trust.

### 3) Ctrl+C did not always produce graceful drain behavior
- Symptoms:
  - Expected behavior: stop capture, drain ASR backlog to zero, commit remaining text, then exit.
  - Observed behavior in one run: native abort output (`forrtl: error (200): program aborting due to control-C event`) instead of clear graceful progression.
  - Also no explicit one-shot log saying first Ctrl+C entered “drain mode.”
- Why this mattered:
  - Violated safety expectations for transcript completeness on shutdown.
  - Users could not confidently know whether pending ASR work would be committed.

### 4) Mixed-mode shutdown could behave worse / appear stuck
- Symptoms:
  - Reported as “worse on mixed mode that it doesn’t kill.”
  - Discussion identified risk around paired ingestion semantics in mixed mode during drain.
- Why this mattered:
  - Mixed mode is core workflow; shutdown uncertainty undermines trust in the notes-first architecture.

### 5) Misleading faster-whisper worker identity/noise on CPU runs
- Symptoms:
  - Log/trace references like `Process narada-faster-whisper-gpu` appeared even with `--compute cpu`.
- Why this mattered:
  - Misleading diagnostics made triage confusing and suggested wrong backend path.

### 6) CI failure due to missing test dependency
- Symptoms:
  - `pytest` collection failed on all CI OSes with `ModuleNotFoundError: No module named 'numpy'` from `tests/test_faster_whisper_engine.py`.
- Why this mattered:
  - Pipeline blocked before tests could even run, masking real regressions and stopping delivery.

## 3. Debugging Path & Options Considered

### Issue 1: Long-window GPU instability
- Debugging path:
  - Traced lifecycle in `faster_whisper_engine.py`: request creation, worker timeout boundary, and GPU-disable path.
  - Compared preset complexity (`fast` vs `balanced`/`accurate`) against fixed timeout behavior.
- Options considered:
  - Dynamic timeout by audio duration + preset.
  - Optional internal chunking only for long GPU windows.
  - Bounded retry + circuit-breaker before disabling GPU.
  - Explicit macOS fallback policy for unsupported faster-whisper GPU modes.
- Helpful discoveries:
  - Fixed timeout + heavier decode settings explained asymmetric failures.
- Dead ends:
  - Keeping a single static short timeout for all durations/presets was not viable.

### Issue 2: Late commits / large ASR backlog
- Debugging path:
  - Walked through notes-first flow in `cli.py`: capture -> spool/planner -> ASR task queue -> result drain -> confidence gate -> writer.
  - Mapped when writes happen: only after `_drain_asr_results()` and gate emission.
  - Confirmed holdback behavior via `StabilizedConfidenceGate`.
- Options considered:
  - Runtime fairness improvements (bounded per-cycle draining/ingest to avoid starvation).
  - Keep no-drop policy by retaining unbounded capture queues but improve scheduling fairness.
- Helpful discoveries:
  - `cap=0` with large `asr` can legitimately happen (capture keeps up while ASR falls behind).
  - `rtf` is processing/audio ratio, so `>1` means ASR slower than real-time.

### Issue 3: Ctrl+C graceful drain and signal semantics
- Debugging path:
  - Located signal handling and shutdown in `cli.py`: `_install_start_signal_handlers`, `_run_tty_notes_first` `try/except/finally`, sentinel + `queue.join`.
  - Compared intended flow with observed native abort log.
- Options considered:
  - First signal strict drain (safe default), second signal force exit.
  - SIGTERM parity with Ctrl+C.
  - Add explicit user-facing message when entering drain mode.
- Helpful discoveries:
  - Python graceful path exists, but native/runtime-level abort can preempt full Python cleanup.
- Dead ends:
  - Assuming every Ctrl+C reaches Python `KeyboardInterrupt` cleanup was incorrect in all environments.

### Issue 4: Mixed-mode shutdown progress
- Debugging path:
  - Reviewed mixed ingestion condition (`while mic_frames and system_frames`) and remaining-frame accounting during shutdown.
- Options considered:
  - Batched/progressive shutdown loops with frequent status + drain checks.
  - Preserve architecture (no mode redesign), but ensure shutdown loop can continue making progress.
- Helpful discoveries:
  - Unbalanced mic/system leftovers can block ingestion progress while “remaining capture” still appears non-zero.

### Issue 5: Worker naming/noise on CPU path
- Debugging path:
  - Reviewed guarded worker startup/name/signal behavior in `faster_whisper_engine.py`.
- Options considered:
  - Device-neutral or device-correct worker naming.
  - Child signal handling improvements so Ctrl+C does not produce misleading noise.
- Helpful discoveries:
  - Diagnostic clarity needed to match actual runtime device path.

### Issue 6: CI numpy import error
- Debugging path:
  - Error was immediate at test import time; root cause was dependency provisioning in CI job.
- Options considered:
  - Install correct extras/deps for test environment (preferred).
  - Make tests conditional/skipped without numpy (fallback option, less ideal).
- Helpful discoveries:
  - This was infra/dependency setup, not a runtime ASR logic bug.

## 4. Final Solution Used (For This Chat)

### 1) Reliability/shutdown direction was converged into concrete implementation plans
- The chat converged on a concrete patch direction:
  - GPU reliability hardening (dynamic timeout, long-window chunking guardrails, bounded retries/circuit-breaker, macOS explicit fallback).
  - Drain-first shutdown semantics with explicit stop-producing + drain queues/tasks before exit.
  - SIGTERM parity and second-signal force-exit policy.
- Main layers/files discussed: `src/narada/asr/faster_whisper_engine.py`, `src/narada/cli.py`, `src/narada/live_notes.py`, plus test modules.

### 2) Backlog/commit behavior was explained with exact runtime conditions
- This episode produced a code-grounded explanation of:
  - when capture stops,
  - when ASR backlog is reduced,
  - and when transcript lines are actually committed.
- Conceptually, the final explanation clarified that commits depend on ASR result drain + gate emission, not raw capture completion.

### 3) CI dependency issue was identified as packaging/workflow, not ASR logic
- In this chat timeline, the failing symptom (`numpy` missing during test collection) was treated as CI environment dependency setup.
- The discussion also referenced that related fixes had been pushed in prior steps of this episode’s workstream.

### 4) End-state of this specific turn
- No new code modifications were made in the last exchanges.
- Final deliverables were diagnostic explanations and a handoff prompt for a new Codex instance to continue implementation cleanly.

## 5. Tools, APIs, and Concepts Used
- **Typer CLI (`narada start`)**: central runtime entrypoint and signal-handling behavior surface.
- **Python signals (`SIGINT`, `SIGTERM`)**: first-signal graceful drain vs second-signal forced stop policy.
- **`queue.Queue`, sentinel, `task_done`, `join`**: backbone of safe producer/consumer drain semantics.
- **Faster-whisper guarded worker**: process-guarded ASR execution with timeout/retry/fallback strategy.
- **Whisper-cpp subprocess path**: timeout/retry concerns for external process execution.
- **Notes-first pipeline**: capture -> spool -> interval planner -> ASR -> confidence gate -> writer.
- **`StabilizedConfidenceGate` holdback**: intentional commit delay mechanism (`holdback_windows`) affecting perceived latency.
- **Runtime metrics (`rtf`, `cap`, `asr`)**: key to interpreting whether lag is in capture or transcription stages.
- **CI test dependency management**: import-time failures can be infra issues, not code logic defects.
- **Cross-OS constraints**: Windows/Linux/macOS behavior parity and graceful fallbacks were explicit requirements.

## 6. Lessons Learned (For This Episode)
- Make shutdown behavior explicit to users; silent transitions feel like failure.
- Distinguish capture backlog from ASR backlog; they can diverge significantly.
- Commit timing in streamed ASR depends on result drain and gating, not just incoming audio.
- Fixed timeouts across all audio lengths/presets are fragile for real workloads.
- Mixed-mode paired ingestion needs shutdown logic that still progresses with imbalanced buffers.
- Second-signal force-exit is useful, but first-signal defaults should stay “drain safely.”
- Native/runtime-level interrupts can bypass ideal Python cleanup paths; design for that reality.
- CI import failures often indicate missing test dependencies rather than functional regressions.
- Keep architecture/contracts stable while hardening internals to avoid user-facing churn.
