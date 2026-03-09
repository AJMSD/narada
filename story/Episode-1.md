# Episode 1

## 1. Context for This Episode
This episode covered the transition from PRD decisions to real implementation and stabilization of Narada v1.

The rough goal was to convert product decisions into an executable checklist, implement major workstreams (capture, mix, ASR, near-live pipeline, tests, model readiness UX), improve installation/device UX, and keep quality high with no regressions.

## 2. Main Problems We Faced

### Issue A: PRD Decisions Were Open / Not Locked
- Symptoms:
  - The PRD had unresolved questions (transcript style, mixed mode approach, language handling, redaction scope, near-live target).
- Why it mattered:
  - Without locked decisions, implementation risked churn and inconsistent behavior.

### Issue B: Need for a Single Execution Tracker + Quality Discipline
- Symptoms:
  - Repeated asks to create/maintain `checklist.md`, track what is left, ensure no syntax/style/type regressions, and update README professionally.
- Why it mattered:
  - The project needed repeatable delivery discipline and clear progress tracking.

### Issue C: Model Setup/Availability UX Was Incomplete
- Symptoms:
  - Questions about whether Whisper models were incorporated, where to get them, and whether setup can be automatic.
  - Need to guide users when models are missing and explain fallback behavior.
- Why it mattered:
  - First-run usability was weak without explicit model diagnostics and setup instructions.

### Issue D: Packaging Dependency Break (`whispercpp>=0.0.18`)
- Symptoms:
  - `pip install -e ".[asr,audio,noise]"` failed with:
    - `No matching distribution found for whispercpp>=0.0.18`
- Why it mattered:
  - Optional ASR installation path was broken for users.

### Issue E: Device Enumeration Returned Nothing
- Symptoms:
  - `narada devices --type input` and `narada devices` returned `No audio devices found.` even with connected hardware.
- Why it mattered:
  - Core UX blocker: users cannot select capture devices.

### Issue F: Windows Device List Had Heavy Duplicates/Noise
- Symptoms:
  - Massive repeated entries across APIs (MME/DirectSound/WASAPI/WDM-KS), alias endpoints (`Sound Mapper`, `Primary Sound Driver`), noisy names, multiline output.
- Why it mattered:
  - High confusion and poor accessibility in device selection.

### Issue G: Device Table Formatting/Columns Hurt Readability
- Symptoms:
  - Header misalignment and extra columns (`Host API`, `Default`) made output harder to read.
- Why it mattered:
  - Device selection should be simple and scannable.

### Issue H: Device Type Selection Was Too Literal
- Symptoms:
  - User wanted logical device IDs where combo devices auto-resolve to input/output by context, without exposing separate IDs.
- Why it mattered:
  - Better mental model for users and fewer selection mistakes.

### Issue I: `start` and `serve` Were Separate Flows
- Symptoms:
  - Need to serve live transcript directly when starting capture, if user opts in.
- Why it mattered:
  - Two-terminal workflow is friction; one-command flow improves usability.

## 3. Debugging Path & Options Considered

### A/B (PRD + Checklist + Quality)
- Steps:
  - Converted open PRD questions into explicit decisions.
  - Built/updated detailed implementation checklist and tracked status.
  - Enforced verify gates repeatedly (`compileall`, `ruff format/check`, `mypy`, `pytest`).
- Options considered:
  - Lightweight notes vs. strict exhaustive checklist with quality gates.
- Helpful discovery:
  - Strict gates caught formatting, typing, and test regressions early.

### C (Model UX)
- Steps:
  - Added model discovery logic and startup preflight messaging.
  - Added doctor model checks and README model setup guidance.
  - Added tests for no-model/partial-model/selected-missing scenarios.
- Options considered:
  - Hard fail when selected model missing vs. guided fallback to available engine.
- Helpful discovery:
  - Explicit preflight outcomes with links gave clearer first-run behavior.

### D (Dependency Break)
- Steps:
  - Diagnosed pip failure from user logs.
  - Added optional `requirements.txt` path with compatible whispercpp pin.
- Options considered:
  - Keep extras-only install vs. provide an alternate requirements-based install route.
- Dead end:
  - Relying only on broken extras spec caused install failure.

### E/F/G/H (Devices)
- Steps:
  - Inspected enumeration code and found silent empty list when `sounddevice` import fails.
  - Implemented OS-aware dedupe and alias filtering.
  - Added raw mode (`--all`) for troubleshooting.
  - Refined output to `ID | Name | Type` only.
  - Introduced logical devices and contextual endpoint resolution (input vs system output).
  - Added tests for dedupe, alias filtering, ambiguity, logical ID routing, and table formatting.
- Options considered:
  - Raw endpoint list only vs. curated default + raw override.
  - Expose host API/default columns vs. minimal columns.
- Helpful discoveries:
  - Curated logical IDs significantly reduce confusion.
  - Keeping `--all` preserves power-user troubleshooting.

### I (`start --serve`)
- Steps:
  - Reused existing server internals by adding a startable/stoppable server helper.
  - Added `--serve`, `--bind`, `--port`, `--qr` to `start` with validation.
  - Added integration tests for server lifecycle and flag misuse.
- Options considered:
  - Full refactor vs. minimal background-thread reuse of existing server.
- Helpful discovery:
  - Minimal server-lifecycle abstraction enabled one-command flow without heavy changes.

## 4. Final Solution Used (For This Chat)

### A/B: Locked Decisions + Execution Framework
- Decision/fix:
  - Locked PRD answers (no timestamps, software mixing, default auto language + multilingual flag, broad PII redaction except names, accuracy-first commits).
  - Used `checklist.md` as execution tracker and enforced verify gates.
- Layers/files involved:
  - Planning/docs and repo-wide quality workflow (`scripts/verify.ps1`, tests, CI, README updates).

### C: Model Discovery + Guidance
- Decision/fix:
  - Added model discovery, startup preflight outcomes, doctor model reporting, model dir config/flags, and setup links.
- Layers/files involved:
  - ASR + CLI + doctor + config + docs:
    - `src/narada/asr/model_discovery.py`
    - `src/narada/cli.py`
    - `src/narada/doctor.py`
    - `src/narada/config.py`
    - `README.md`, `.env.example`

### D: Installation Unblock
- Decision/fix:
  - Added `requirements.txt` optional install path with compatible dependency versions.
- Layers/files involved:
  - `requirements.txt`, `README.md`

### E/F/G/H: Device UX Overhaul
- Decision/fix:
  - Default device list is curated/deduped and cleaner.
  - Added `--all` for raw endpoints.
  - Table simplified to ID/Name/Type.
  - Implemented logical IDs with automatic input/output endpoint selection based on command context.
- Layers/files involved:
  - `src/narada/devices.py`
  - `src/narada/cli.py`
  - `tests/test_devices.py`
  - `README.md`

### I: Start + Serve in One Command
- Decision/fix:
  - Added opt-in embedded serving in `narada start` via `--serve` plus networking flags.
  - Reused server code with explicit lifecycle object.
- Layers/files involved:
  - `src/narada/server.py`
  - `src/narada/cli.py`
  - `tests/test_start_integration.py`
  - `tests/test_server_integration.py`
  - `README.md`

## 5. Tools, APIs, and Concepts Used
- **Typer CLI**: Command surface and flag validation for `devices`, `start`, `serve`, `doctor`.
- **sounddevice / PortAudio**: Enumerated audio endpoints; exposed multi-host-API duplication behavior.
- **Windows host APIs (WASAPI/MME/DirectSound/WDM-KS)**: Core source of duplicate endpoint entries.
- **faster-whisper & whisper.cpp adapters**: Dual-engine ASR support and fallback behavior.
- **Model discovery/preflight**: Runtime checks for local model availability and setup guidance.
- **SSE (Server-Sent Events)**: Live LAN transcript streaming via `/events`.
- **Append-only transcript writer**: File as source of truth for both local output and LAN view.
- **Ruff + mypy + pytest + compileall**: Quality gate stack that prevented regressions.
- **README + `.env.example` + `requirements.txt`**: Installation, configuration, and operational clarity.
- **Logical device abstraction**: User-facing device IDs decoupled from raw backend endpoint IDs.

## 6. Lessons Learned (For This Episode)
- Lock product decisions early, or implementation effort fragments quickly.
- A strict local verify pipeline is worth it; it catches subtle breakage before push.
- First-run UX (clear diagnostics + setup links) is as important as core functionality.
- On Windows audio stacks, raw endpoint enumeration is not user-facing UX; curation is required.
- Keep a curated default and a raw escape hatch (`--all`) to satisfy both normal users and debuggers.
- Logical IDs with contextual endpoint resolution reduce user errors and cognitive load.
- Add features by reusing existing layers (server lifecycle abstraction) before considering large refactors.
- Installation paths should have a resilient fallback when optional dependency ecosystems shift.
- Documentation should evolve with behavior changes immediately, not as an afterthought.
- Repeated, test-backed small commits keep large refactors safe.
