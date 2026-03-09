# Episode 9

## 1. Context for This Episode
- This episode focused on two closely related parts of the Windows live-runtime experience: recovering a broken `--mode system` preflight path, and then cleaning up terminal output so live `REC` status stayed readable during normal use and `--serve`.
- The rough goal was to restore trustworthy Windows system capture without relaxing backend requirements, then make the interactive CLI print only the statements users actually need.

## 2. Main Problems We Faced

### 1) Windows system-capture readiness regression
- Symptoms: `narada start --mode system --system 20 ...` began failing preflight with: `Windows system capture requires a WASAPI output device. Selected 'Headphones ...' uses host API 'unknown'.`
- Symptoms: `narada devices` still showed the same device ID as an output device, and the user reported that this path had worked before.
- Why this was a problem: the preflight gate rejected a valid Windows system-capture setup, so live transcription could not even start.

### 2) Device metadata mismatch between selection UX and backend validation
- Symptoms: device IDs shown to the user did not line up cleanly with readiness validation, especially for logical `input/output` devices and curated device lists.
- Symptoms: the selected device could be valid in runtime terms but fail setup checks because host API data had become `None` or `"unknown"`.
- Why this was a problem: the user-facing selector and the backend truth source had drifted apart, which made system capture feel unreliable and confusing.

### 3) Confusion around WASAPI vs PyAudioWPatch on Windows
- Symptoms: the user asked why Narada was using WASAPI at all, since PyAudioWPatch had been installed specifically to avoid this class of issue.
- Why this was a problem: without a clear explanation, the failure looked like the wrong backend was being used instead of a false-negative validation bug.

### 4) Terminal noise breaking the one-line live UI
- Symptoms: terminal output still included lines like `2026-03-03 ... INFO narada.asr.whisper_cpp: whisper.cpp compute resolved...`.
- Symptoms: `--serve` printed full request-thread tracebacks such as `ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host`.
- Why this was a problem: these lines broke the single-line `REC` display, filled the terminal with low-signal runtime chatter, and made expected client disconnects look like application failures.

### 5) Too many runtime statements competing with the actual UX
- Symptoms: backlog warnings, engine diagnostics, and retry chatter were all candidates to print during live operation.
- Why this was a problem: the terminal was trying to serve as both a user UI and a debug log stream, which degraded both roles.

## 3. Debugging Path & Options Considered

### Issue 1: Windows system-capture readiness regression
- Debugging steps: inspected `src/narada/devices.py` and `src/narada/setup/assistant.py` to compare curated device enumeration against raw backend enumeration.
- Debugging steps: traced the preflight path through `prepare_start_setup -> ensure_system_capture_prereqs -> _check_system_capture_readiness`.
- Helpful discovery: the readiness check was validating against the curated list, while runtime capture later resolved against raw/include-all devices.
- Helpful discovery: `hostapi` metadata was being dropped when logical devices were built or selections were materialized.
- Options considered: relax the WASAPI requirement, keep validating only against curated devices, or validate against raw backend devices while preserving the curated selector UX.
- Dead end: relaxing WASAPI was rejected because the requirement itself was not the bug.
- Dead end: keeping curated-only validation would preserve the false-negative behavior.

### Issue 2: Selector/backend mismatch
- Debugging steps: looked at logical `input/output` device construction and how `system_device_id` and `hostapi` flowed into materialized selections.
- Helpful discovery: the curated layer was useful for UX, but not authoritative enough for backend readiness decisions on Windows.
- Options considered: making curated IDs the canonical backend source, copying backend metadata through the curated layer, and resolving final readiness against raw endpoints.
- Dead end: treating the curated view as the sole source of truth was what caused the mismatch in the first place.

### Issue 3: WASAPI vs PyAudioWPatch confusion
- Debugging steps: revisited the Windows backend assumptions and clarified what PyAudioWPatch actually exposes.
- Helpful discovery: PyAudioWPatch does not replace the WASAPI requirement; it depends on WASAPI loopback endpoints on Windows.
- Options considered: bypassing WASAPI checks or accepting non-WASAPI outputs.
- Dead end: bypassing WASAPI would have turned a false-negative fix into a policy change and risked breaking valid backend assumptions.

### Issue 4: Terminal noise and broken single-line status
- Debugging steps: scanned `src/narada/cli.py`, `src/narada/server.py`, `src/narada/asr/whisper_cpp_engine.py`, and `src/narada/asr/faster_whisper_engine.py` for `logger.*`, `_safe_echo`, `typer.echo`, and server traceback paths.
- Helpful discovery: the `ConnectionResetError` traceback was coming from Python's `socketserver`/`http.server` error handling, not from ASR itself.
- Helpful discovery: single-line status rendering only controlled its own writes; stderr logging and server-thread tracebacks could still break the UI.
- Options considered: keep verbose output, keep warning lines but separate them cleanly, move backlog pressure into the status line, and suppress expected disconnect tracebacks entirely.
- Dead end: continuing to treat expected HTTP disconnects as full terminal errors would keep breaking the live UI.

### Issue 5: Deciding what the user should actually see
- Debugging steps: separated startup/shutdown/actionable errors from routine diagnostics and retry chatter.
- Helpful discovery: the terminal needed a minimal product-facing policy, not another round of ad hoc special cases.
- Options considered: minimal terminal output, balanced output, and verbose output.
- Final preference chosen in-chat: minimal terminal output, status-only backlog warnings, and silent handling of normal serve disconnects.

## 4. Final Solution Used (For This Chat)

### Issue 1 + Issue 2: Restore trustworthy Windows readiness checks
- Actual fix: preserved `hostapi` metadata through curated and materialized device representations.
- Actual fix: changed Windows readiness validation to use raw/include-all devices for the final WASAPI check, while keeping curated device enumeration for selector UX.
- Actual fix: added a safe fallback so if raw validation could not resolve the selected endpoint, readiness could still fall back to the curated set instead of failing in a new way.
- Files/layers involved: `src/narada/devices.py`, `src/narada/setup/assistant.py`.
- Conceptual change: the curated list remains the user-facing selection layer, but raw endpoint enumeration is now the authoritative backend validation source on Windows.

### Issue 3: Clarify backend policy without changing it
- Actual decision: keep WASAPI strict on Windows and do not accept non-WASAPI outputs.
- Files/layers involved: no policy relaxation was introduced; the fix stayed in metadata propagation and setup resolution.
- Conceptual change: PyAudioWPatch remained a dependency for Windows loopback capture, but no longer got blamed for a metadata propagation bug.

### Issue 4 + Issue 5: Minimal-terminal output hardening
- Actual fix: converted notes-first backlog warnings from standalone warning lines into compact status indicators (`warn=asr`, `warn=cap`, or `warn=asr+cap`) in the live `REC` line.
- Actual fix: suppressed expected `--serve` client disconnect/reset tracebacks by overriding server error handling and treating normal socket reset/pipe-close conditions as expected churn.
- Actual fix: demoted non-essential engine diagnostics to debug, including whisper.cpp compute-resolution info and retry chatter.
- Files/layers involved: `src/narada/cli.py`, `src/narada/server.py`, `src/narada/asr/whisper_cpp_engine.py`, `src/narada/asr/faster_whisper_engine.py`.
- Conceptual change: the terminal was narrowed to essential user-facing state, while debug-style details stayed in logging rather than competing with the live UI.

### Validation and delivery in this chat
- Actual work completed: both change sets were implemented, tested, committed, and pushed.
- Quality gates run and passing:
- `python -m compileall src tests`
- `ruff format --check .`
- `ruff check .`
- `mypy src`
- `pytest -q`

## 5. Tools, APIs, and Concepts Used
- Windows WASAPI: remained the required host API for system loopback capture on Windows.
- PyAudioWPatch: clarified as the Windows loopback mechanism that still depends on WASAPI endpoints.
- Curated vs raw device enumeration: the key distinction that explained why `narada devices` could look correct while preflight still failed.
- `AudioDevice.hostapi`: the missing metadata field that had to survive logical-device construction and selection materialization.
- Setup assistant preflight flow: `prepare_start_setup`, `ensure_system_capture_prereqs`, and `_check_system_capture_readiness` were the critical setup path.
- `_LiveStatusRenderer`: the live single-line CLI renderer whose output was being broken by unrelated terminal writes.
- Python `socketserver` / `http.server`: the source of request-thread tracebacks for normal client disconnects during `--serve`.
- `handle_error()` override: the mechanism used to suppress expected disconnect noise while preserving unexpected error reporting.
- whisper.cpp and faster-whisper logging: used to separate product-facing runtime output from debug-only diagnostics.
- `pytest`, `ruff`, `mypy`, `compileall`: used to lock the fixes down and prove they did not regress style, typing, syntax, or behavior.

## 6. Lessons Learned (For This Episode)
- A curated device list is a UX layer, not automatically a reliable backend validation source.
- Metadata loss is enough to create a false platform regression even when the actual runtime backend still works.
- On Windows loopback capture, keeping WASAPI strict is fine if the validation path is actually using correct endpoint metadata.
- A single-line terminal UI only works if every other runtime output path is explicitly treated as part of the same UX design.
- Expected network disconnects in a live server should not surface as terminal failures by default.
- Debug diagnostics and user-facing runtime state should not compete for the same terminal channel.
- When a regression appears right after a broader feature change, the safest fix is often a narrow plumbing correction rather than a policy change.
