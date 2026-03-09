# Episode 4

## 1. Context for This Episode
This episode focused on stabilizing Narada live transcription on Windows while moving from a strict "real-time captions" mindset toward a "useful notes as fast as possible after meeting end" workflow.

The rough goal was to fix reliability/performance blockers (faster-whisper live behavior, GPU failures, dashboard duplication concerns), verify runtime dependencies end-to-end, and decide what architecture would best serve post-call notes.

## 2. Main Problems We Faced

### Issue 1: Faster-whisper live stall on GPU after VAD
- Key symptoms:
  - Logs repeatedly showed `Processing audio with duration 00:06.000` and then `VAD filter removed ...` with no commits for minutes.
  - CPU mode worked, but CUDA/auto path stalled.
- Why it mattered:
  - Live output became unusable and fallback behavior was inconsistent when stalls occurred during lazy segment iteration.

### Issue 2: Sample-rate handling mismatch in live NumPy path
- Key symptoms:
  - Captured audio could be opened at rates like 48k while model expects 16k waveform input.
  - Reported effect: duration/time distortion (for example, 6s interpreted like ~18s), lag, and late/missing commits.
- Why it mattered:
  - Timing drift directly degrades both live UX and transcript correctness.

### Issue 3: SSE dashboard duplicate lines on reconnect
- Key symptoms:
  - Transcript file contents were correct, but browser log repeated chunks.
  - `/events` cursor state reset per connection and frontend appended blindly.
- Why it mattered:
  - Dashboard trust breaks when displayed transcript diverges from source file.

### Issue 4: CUDA runtime dependencies missing / inconsistent
- Key symptoms:
  - Direct faster-whisper CUDA probe failed with `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`.
  - DLL checks showed missing `cublas64_12.dll`, `cudart64_12.dll`, and related runtime files in active PATH context.
- Why it mattered:
  - GPU acceleration could not run reliably, forcing slower CPU paths.

### Issue 5: Whisper-cpp runtime/GPU ambiguity
- Key symptoms:
  - Initially, `whisper-cli` was not on PATH, so whisper-cpp engine availability was false.
  - Later, command behavior showed `-ng` handling mismatch (`error: input file not found '99'`) and `use gpu = 0` in stderr for that invocation style.
- Why it mattered:
  - User expected GPU usage, but runtime behavior could silently fall back to CPU.

### Issue 6: Model-availability confusion ("only whisper-cpp model files available")
- Key symptoms:
  - Running `--model tiny` produced: only whisper-cpp model files available.
  - `narada doctor --model tiny` confirmed faster-whisper tiny missing; `--model small` showed both engines available.
- Why it mattered:
  - User saw contradictory expectations ("both engines installed") versus per-model local file availability.

### Issue 7: Real-time versus notes-first outcome
- Key symptoms:
  - User-observed throughput/latency: high RTF and long commit delays (`rtf=1.39`, ~24s commit on tiny whisper-cpp run; worse in other cases).
- Why it mattered:
  - If end-of-meeting output is still delayed too long, value over native meeting transcribers is reduced.

## 3. Debugging Path & Options Considered

### Issue 1: Faster-whisper GPU stall
- Debugging steps:
  - Traced that `faster-whisper` segment consumption is lazy, so hangs can occur during iteration.
  - Confirmed fallback around setup call alone is insufficient for iteration-time stalls.
- Options considered:
  - **A:** Cover generator consumption with timeout/watchdog and fallback.
  - **B:** Avoid GPU language detection path for `--language auto`.
  - **C:** Preflight CUDA runtime and downgrade early.
- Helpful discovery:
  - Hard recovery requires guarding the full transcribe+materialization path, not just call setup.

### Issue 2: Sample-rate mismatch
- Debugging steps:
  - Reviewed capture sample-rate behavior and faster-whisper NumPy waveform expectations.
- Options considered:
  - Resample in capture layer vs resample in faster-whisper adapter before `model.transcribe(...)`.
- Helpful discovery:
  - Keeping capture/request contract unchanged and normalizing at adapter boundary minimizes blast radius.

### Issue 3: SSE duplication
- Debugging steps:
  - Reviewed `/events` behavior and frontend append logic.
- Options considered:
  - Add SSE `id` + `Last-Event-ID` resume semantics.
  - Add client-side idempotency guard on `lastEventId`.
- Helpful discovery:
  - Duplication was transport/reconnect semantics, not writer/file corruption.

### Issue 4: CUDA dependency failures
- Debugging steps:
  - Ran `nvidia-smi`, package checks, direct faster-whisper CUDA probe, and DLL load checks.
- Options considered:
  - Stay CPU-only vs install CUDA toolkit/runtime and verify all binaries in PATH.
- Dead end:
  - Assuming driver presence alone was sufficient.
- Helpful discovery:
  - CTranslate2 device visibility can be true while runtime still fails at first real GPU op.

### Issue 5: Whisper-cpp GPU behavior
- Debugging steps:
  - Verified Narada whisper-cpp command construction and tested actual CLI behavior.
  - Compared stderr from runs with/without current `-ng` usage.
- Options considered:
  - Trust current invocation vs align invocation to installed `whisper-cli` flag semantics.
- Helpful discovery:
  - Current invocation pattern on this installed CLI variant can disable/mis-handle GPU intent.

### Issue 6: Model availability message
- Debugging steps:
  - Ran `narada doctor --model tiny` and `--model small`.
  - Resolved actual model paths for both engines.
- Helpful discovery:
  - Availability is model-specific, not just engine-package-specific.

### Issue 7: Notes-first strategy
- Debugging steps:
  - Compared chunked live commit behavior vs post-call processing latency tradeoff.
- Options considered:
  - Full post-call transcription only.
  - Confidence-gated delayed commits.
  - Interval/background transcription while capture continues, with tail finalize at end.
- Helpful discovery:
  - Capture-first + background transcription best balances fast post-call notes with low drop risk.

## 4. Final Solution Used (For This Chat)

### Issue 1: Faster-whisper GPU stall recovery
- Actual fix/decision:
  - Implemented guarded GPU execution with timeout and session-level GPU disable/fallback to CPU when GPU path stalls/fails.
- Files/layers involved:
  - `src/narada/asr/faster_whisper_engine.py`
  - Regression tests in `tests/test_faster_whisper_engine.py`

### Issue 2: Live latency profile defaults
- Actual fix/decision:
  - Implemented lower-latency live defaults (`2.0s` chunk / `0.5s` overlap) to reduce end-to-commit delay.
- Files/layers involved:
  - `src/narada/cli.py`
  - Coverage in `tests/test_start_integration.py`

### Issue 3: Verification and non-regression
- Actual fix/decision:
  - Ran targeted tests + full suite + lint successfully (as reported during the session).
- Files/layers involved:
  - `tests/test_faster_whisper_engine.py`
  - `tests/test_start_integration.py`
  - `README.md` (behavior notes)

### Issue 4: Runtime environment readiness for GPU
- Actual fix/decision:
  - Installed CUDA toolkit, installed CUDA-enabled `whisper-cli`, persisted User PATH, and re-verified engine availability and probes.
- Layers involved:
  - Local Windows runtime/tooling setup (no repo code changes for this step).

### Issue 5: Model availability confusion
- Actual fix/decision:
  - Confirmed root cause as missing faster-whisper **tiny** model specifically, not missing faster-whisper engine package.
- Layers involved:
  - `narada doctor` checks and model discovery behavior.

### Issue 6: Whisper-cpp GPU command mismatch
- Actual fix/decision:
  - Diagnosed and confirmed mismatch behavior in current invocation for this installed CLI variant.
  - Not fixed in code during this chat; left as explicit follow-up implementation item.
- Files/layers involved:
  - `src/narada/asr/whisper_cpp_engine.py` (identified, not edited in this step)

### Issue 7: Notes-first architecture direction
- Actual decision:
  - Chosen direction was to prioritize capture-first + interval/background transcription + fast end-of-meeting finalize, rather than strict live captioning guarantees.
  - Planning/docs prompts were produced; no new code implemented for this architecture in this final step.

## 5. Tools, APIs, and Concepts Used
- **Narada CLI (`start`, `doctor`)**: Used to validate engine/model/runtime state and reproduce user-facing behavior.
- **faster-whisper + CTranslate2**: Used for GPU/CPU transcription path validation and failure classification.
- **whisper.cpp (`whisper-cli`)**: Used to verify runtime availability and inspect actual GPU flag semantics.
- **CUDA runtime + `nvidia-smi`**: Used to confirm GPU hardware visibility versus actual inference readiness.
- **Windows PATH/User environment**: Used to make CUDA and `whisper-cli` discoverable persistently.
- **Overlap chunking + commit latency/RTF metrics**: Used to reason about real-time usability and notes-readiness.
- **SSE reconnect model (`/events`)**: Used to isolate frontend duplication as stream-resume behavior.
- **Test/lint gates (`pytest`, `ruff`)**: Used to validate implemented fixes without regressing existing behavior.

## 6. Lessons Learned (For This Episode)
- Treat GPU availability as a runtime execution check, not just package/device detection.
- If inference APIs are lazy generators, fallback logic must guard iteration/materialization too.
- Keep sample-rate normalization at engine adapter boundaries to preserve higher-level contracts.
- Model availability is per model size/version; engine install alone does not guarantee readiness.
- Reconnect-safe live views need explicit event resume/idempotency semantics.
- Real-time captioning and fast post-call notes are different optimization targets; design explicitly for one.
- Capture reliability should be independent of ASR speed to avoid dropped audio under load.
- Verify CLI flag semantics against the actual installed binary version, not assumptions from prior builds.
