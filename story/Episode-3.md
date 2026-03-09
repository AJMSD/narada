# Episode 3

## 1. Context for This Episode
This episode focused on stabilizing Windows live transcription for `narada start --mode system` and making runtime behavior match what users see in `narada devices`.

The rough goal was to get real-time system capture/transcription working reliably on Windows, reduce dropped/late output, and explain remaining gaps clearly.

## 2. Main Problems We Faced

### Issue 1: Windows system capture failed with sounddevice loopback
- Key symptoms:
  - `PortAudioError ... Invalid number of channels [PaErrorCode -9998]`
  - Later: `Installed sounddevice (0.5.5) does not support WasapiSettings(loopback=True)`
- Why it was a problem:
  - `--mode system` could not open streams reliably, so core functionality failed.

### Issue 2: Device IDs were confusing/unusable on Windows
- Key symptoms:
  - Device lists included mixed host APIs, truncated names, duplicate-looking devices, and IDs that did not map cleanly to working loopback capture.
- Why it was a problem:
  - Users selected IDs that looked valid but failed at runtime, creating repeated setup failures.

### Issue 3: Live mode dropped or delayed effective transcription output
- Key symptoms:
  - Long periods with no transcript commits.
  - User observed extremely small captured audio windows (example discussion around `0.033s`) and wanted wall-clock commits.
- Why it was a problem:
  - Real calls require continuous capture and predictable commit cadence; 1 Hz polling-style behavior risked starvation/backlog.

### Issue 4: Faster-whisper runtime crash after live-loop refactor
- Key symptoms:
  - `ValueError: File object has no read() method, or readable() returned False.`
  - Wrapped as `EngineUnavailableError` in Narada.
- Why it was a problem:
  - Live transcription crashed before writing text, leaving output files empty.

### Issue 5: Whisper-cpp engine unavailable despite local model file
- Key symptoms:
  - `Selected engine 'whisper-cpp' is unavailable...`
  - `narada doctor` showed model present but engine unavailable.
- Why it was a problem:
  - Users assumed model presence was sufficient; runtime dependency (`whisper-cli`) was missing.

### Issue 6: Whisper-cpp live throughput lag and apparent dropped audio
- Key symptoms:
  - Status like `rtf=2.74 | commit=44633ms`.
  - Perception of stall and repeated wording.
- Why it was a problem:
  - Processing slower than real time causes large lag and makes call transcription feel unreliable.

### Issue 7: QR code did not render
- Key symptoms:
  - `Install qrcode[pil] to enable ASCII QR rendering.`
- Why it was a problem:
  - Expected UX feature was absent unless optional dependency was installed.

### Issue 8: Dashboard duplicated content even when file content was clean
- Key symptoms:
  - Web log showed repeated chunks while transcript file had single appended lines.
- Why it was a problem:
  - Live viewer looked incorrect/noisy, reducing trust in realtime output.

## 3. Debugging Path & Options Considered

### Issue 1: Windows loopback backend failure
- Debugging steps:
  - Traced stack from `cli.py` into capture open path.
  - Confirmed environment-level loopback limitation with installed sounddevice/PortAudio build.
- Options considered:
  - `PyAudioWPatch` vs `SoundCard` for Windows loopback.
  - User requested unified Windows backend for mic + system to avoid mixed-library mismatch.
- Dead ends:
  - Trying to keep sounddevice loopback on that Windows build.
- Helpful discoveries:
  - Loopback support was absent in the installed sounddevice stack, so migration was required.

### Issue 2: Device ID mismatch/noisy enumeration
- Debugging steps:
  - Reviewed Windows enumeration and runtime selection flow.
  - Compared listed IDs against capture backend behavior.
- Options considered:
  - Keep dedupe and try to reconcile truncated names.
  - Filter MME/DirectSound before dedupe so unusable endpoints never appear in default listing.
- Helpful discoveries:
  - Runtime and enumeration must use the same Windows backend for stable ID semantics.

### Issue 3: Dropped/delayed commits in live mode
- Debugging steps:
  - Reviewed main loop behavior and chunk commit triggers.
  - Identified need for continuous reads and wall-clock flush boundaries.
- Options considered:
  - Keep existing cadence with minor tuning.
  - Move to producer/consumer capture threads + queue + forced wall-clock flush (chosen).
- Helpful discoveries:
  - No-drop intent needs continuous capture ingestion, not periodic polling.

### Issue 4: Faster-whisper crash
- Debugging steps:
  - Traced call path to `faster_whisper_engine.py`.
  - Verified installed faster-whisper API expected `numpy.ndarray`/file-like input, not Python list.
- Options considered:
  - Temporary fallback to whisper-cpp.
  - Fix adapter input type and warmup type (chosen).
- Helpful discoveries:
  - Tests used fake engines/models, so type mismatch escaped coverage.

### Issue 5: Whisper-cpp unavailable
- Debugging steps:
  - Checked `where whisper-cli`, `narada doctor`, environment PATH, and engine availability probe.
  - Located that model file existed but CLI binary was absent from PATH.
- Options considered:
  - Use faster-whisper only.
  - Install whisper.cpp CLI and expose it on PATH (chosen).
- Dead ends:
  - Assuming `whispercpp` Python package alone was enough for current Narada whisper-cpp runtime path.

### Issue 6: Whisper-cpp performance lag
- Debugging steps:
  - Interpreted runtime metrics (`rtf`, commit latency).
  - Related lag to process-per-chunk whisper-cli path and overlap behavior.
- Options considered:
  - Use smaller model (`tiny`) for speed.
  - Move to GPU-capable binaries for whisper.cpp, or prefer faster-whisper CUDA path.
- Helpful discoveries:
  - Throughput mismatch, not only “hardware bad,” explained the user-visible lag.

### Issue 7: QR output
- Debugging steps:
  - Checked server QR helper behavior.
- Options considered:
  - Keep optional dependency and show install hint (current behavior).
  - Make QR dependency always installed.
- Helpful discoveries:
  - Message was expected behavior, not a capture/transcription failure.

### Issue 8: Dashboard duplication
- Debugging steps:
  - Read SSE handler and frontend append logic.
  - Noted reconnect resets server cursor (`last_size`) and frontend appends blindly.
- Options considered:
  - Keep current simple append behavior.
  - Add resume/dedupe strategy for SSE reconnects (planned next, not implemented in this episode).
- Helpful discoveries:
  - File output and viewer rendering can diverge because of transport/reconnect semantics.

## 4. Final Solution Used (For This Chat)

### Issue 1 + Issue 2: Windows backend unification (implemented earlier in this chat flow)
- Actual decision/fix:
  - Route Windows mic/system capture through `PyAudioWPatch`.
  - Align Windows device enumeration with same backend.
  - Keep CLI contract unchanged (ID/name/contains matching).
- Layers involved:
  - `src/narada/audio/backends/windows.py`
  - `src/narada/audio/capture.py`
  - `src/narada/devices.py`
  - Related tests/docs

### Issue 3: Continuous capture + wall-clock commit (implemented)
- Actual decision/fix:
  - Replaced live-loop behavior with capture worker threads + queues.
  - Added wall-clock forced flush and queue backlog warnings.
  - Preserved existing chunker/gate/writer contracts.
- Layers involved:
  - `src/narada/cli.py`
  - `src/narada/config.py`
  - Integration/runtime tests

### Issue 4: Faster-whisper type crash (implemented)
- Actual decision/fix:
  - Changed faster-whisper adapter to pass `numpy.float32` arrays instead of Python lists.
  - Updated warmup path accordingly.
  - Added regression tests asserting ndarray usage.
- Layers involved:
  - `src/narada/asr/faster_whisper_engine.py`
  - `tests/test_faster_whisper_engine.py`

### Issue 5: Whisper-cpp availability correctness (implemented)
- Actual decision/fix:
  - Hardened availability to match actual runtime requirement (`whisper-cli` on PATH).
  - Added availability tests and aligned dependency pins.
  - Installed `whisper-cli.exe` locally and set PATH guidance.
- Layers involved:
  - `src/narada/asr/whisper_cpp_engine.py`
  - `tests/test_whisper_cpp_engine.py`
  - `requirements.txt`, `README.md`

### Issue 6: Performance mitigation (partially resolved operationally)
- Actual decision/fix:
  - Downloaded `ggml-tiny.bin` for whisper.cpp to enable faster runs.
  - Confirmed GPU exists; clarified current whisper.cpp binary was CPU build and GPU needs cuBLAS release binary.
  - Confirmed faster-whisper CUDA viability in environment.
- Layers involved:
  - Runtime environment setup (binary/model/PATH), not code changes in this step.

### Issue 7: QR behavior (decision only)
- Actual decision/fix:
  - Kept current optional dependency behavior and installation hint.
  - No code change in this chat for auto-install/default inclusion.

### Issue 8: Dashboard duplication (diagnosed, not fixed yet)
- Actual decision/fix:
  - Root cause identified in SSE reconnect behavior and frontend append model.
  - A dedicated next-step plan prompt was prepared to fix this alongside remaining faster-whisper runtime correctness.
  - No code patch for dashboard dedupe in this chat.

## 5. Tools, APIs, and Concepts Used
- **PyAudioWPatch**: Used as Windows capture/enumeration backend to align device IDs with runtime behavior.
- **sounddevice / PortAudio**: Initial backend that exposed Windows loopback limitations in this environment.
- **faster-whisper**: Main ASR path; learned strict input-type expectations and runtime behavior implications.
- **whisper.cpp / whisper-cli**: Alternate ASR engine requiring CLI presence on PATH for current Narada adapter path.
- **Typer CLI**: Command surface for `start`, `devices`, `doctor`; surfaced actionable runtime diagnostics.
- **SSE (`/events`) + EventSource**: Live dashboard transport; reconnect semantics explained duplicate UI appends.
- **OverlapChunker + ConfidenceGate + TranscriptWriter**: Core pipeline components reused while refactoring live runtime.
- **Threaded producer/consumer queues**: Used to move from periodic polling to continuous capture ingestion.
- **Runtime metrics (`rtf`, commit latency)**: Used to identify throughput mismatch and explain perceived stalls/drops.
- **NVIDIA/CUDA checks (`nvidia-smi`, ctranslate2/faster-whisper probe)**: Confirmed GPU viability for selected engine paths.

## 6. Lessons Learned (For This Episode)
- Keep device enumeration and capture runtime on the same backend to avoid “valid-looking but unusable” IDs.
- A no-drop real-time goal requires continuous ingestion architecture, not periodic capture loops.
- Adapter type mismatches can stay hidden if tests only use permissive fakes; include type-contract assertions in tests.
- Model presence and engine availability are different checks; runtime dependencies (like CLI binaries) must be explicit.
- Throughput metrics (`rtf`, commit latency) are crucial for explaining user-perceived stalls and dropped speech.
- Optional UX features (like QR rendering) should fail clearly and independently from core capture/ASR paths.
- Reconnect-safe streaming UIs need resume/dedupe semantics; append-only clients alone are not sufficient.
- Fixes can unmask deeper issues; treat each newly surfaced error as progress in layered debugging.
