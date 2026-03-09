# Episode 10: Diagnosing a Failed Run and Implementing Performance Improvements

---

## 1. Context for This Episode

This episode covers a complete performance investigation cycle on the Narada project. It started with reading and understanding the full codebase from scratch, then moved into diagnosing a specific failed real-world session, then into identifying five concrete performance improvement opportunities, and finally into implementing four of the five across three files on a dedicated branch.

The rough goal was: understand why a multilingual system-audio transcription session produced repeated ASR worker timeouts, frame drops, and a massive post-Ctrl+C drain backlog, then make the code measurably more efficient without changing any existing logic or behavior.

---

## 2. Main Problems We Faced

### Problem 1: Real-Time Factor Exceeding 1.0 (RTF > 1)

**Symptoms:**
- Status line showed `rtf=1.05` climbing to `rtf=3.62`
- RTF > 1.0 means transcription is slower than real-time
- At RTF=3.62, the system needed 3.62 seconds of CPU time to process 1 second of audio

**Why it mattered:** Once RTF exceeds 1.0, the ASR backlog can never shrink while audio is still being captured. It grows continuously, causing cascading failures.

---

### Problem 2: Spurious ASR Worker Timeouts in Multilingual Mode

**Symptoms:**
- Five out of thirteen tasks failed with: `faster-whisper CPU worker failed after retry: ASR worker exceeded timeout (51.6s).`
- The timeout of exactly 51.6s matched the formula: `30 + 12 × 1.80 = 51.6s` (CPU balanced preset on a 12-second audio window)

**Why it mattered:** The timeout budget was calibrated for single-language operation. Multilingual mode runs per-segment language detection on top of beam decoding, adding roughly 30-50% overhead. Tasks that would have succeeded were being killed because the timeout constant assumed single-language cost.

---

### Problem 3: Redundant Memory Allocations in the PCM-to-Float Conversion

**Symptoms:** Not a runtime crash, but a code-level inefficiency discovered during a deep read of `faster_whisper_engine.py`.

```python
return (samples.astype(np.float32) / 32768.0).copy()
```

**Why it mattered:** `astype` already allocates array #1, `/` allocates array #2, `.copy()` allocates array #3. The final `.copy()` is entirely redundant because the division already returned a new array. This happens on every single transcription call.

---

### Problem 4: Redundant `bytes()` Wrap Before `np.frombuffer` in the GPU Worker

**Symptoms:** Code-level inefficiency in `_gpu_transcribe_worker_main`:

```python
audio = np.frombuffer(bytes(request.get("audio", b"")), dtype=np.float32).copy()
```

**Why it mattered:** The value arriving from the multiprocessing queue is already `bytes`. Wrapping it in `bytes()` creates a full second copy of the raw audio payload before `np.frombuffer` even starts. For a 12-second audio window at 16kHz float32, that is roughly 768KB being copied for no reason on every worker request.

---

### Problem 5: `SessionSpool.read_range` Opening a New File Descriptor per ASR Task

**Symptoms:** Code-level inefficiency in `live_notes.py`:

```python
with self.data_path.open("rb") as handle:
    handle.seek(start_byte)
    return handle.read(end_byte - start_byte)
```

**Why it mattered:** Every interval task calls `read_range` once. Each call triggered a full `open` → `seek` → `read` → `close` syscall sequence. With the default 12-second interval this happens constantly throughout a session.

---

### Problem 6: O(n) Front-Deletion on Every Emitted Window in `OverlapChunker`

**Symptoms:** Code-level inefficiency in `pipeline.py`:

```python
del self._buffer[:stride]
```

**Why it mattered:** `del bytearray[:n]` shifts all remaining bytes toward index 0 — a full `memmove` of everything after the deleted region. On a 6-second audio chunk at 16kHz mono PCM16, that is ~192KB per emission. This runs on every emitted chunk on the non-TTY (legacy stdin) capture path.

---

## 3. Debugging Path & Options Considered

### For the failed run (Problems 1 and 2)

The log output was examined line by line. Key observations:

- `asr=126.9s` in the status line with only 2m26s elapsed meant ASR had fallen 126 seconds behind before the user even hit Ctrl+C.
- The timeout value of `51.6s` was not arbitrary — it matched exactly the formula `_CPU_TRANSCRIBE_TIMEOUT_BASE_S + (audio_seconds × factor)` = `30 + 12 × 1.80`.
- The `cap=0.0s` and `drop=1` values indicated a brief capture queue backup had occurred.
- The drain after Ctrl+C took approximately 14 minutes (`eta=~369.7s` at one point) because the RTF during drain was still 2-3x.

**Options considered for the user's situation:**
- `--asr-preset fast` (lower beam size, no VAD) — identified as the fastest user-level lever
- `--model tiny` instead of `small` — trades accuracy for speed
- `--engine whisper-cpp` — potentially faster on macOS ARM
- `--language auto` instead of `hindi,english` — removes multilingual processing overhead
- `--notes-interval-seconds 20` — reduces ASR task frequency

None of these were implemented in code; they were surfaced as user options.

**For the code fixes** the approach was: read every relevant file completely before touching anything, identify every call site for each method being changed, check test coverage, make the minimum possible diff, and run the test suite before each commit.

### Dead ends
None — all five issues were confirmed as real inefficiencies. No proposed fix was rolled back.

### Helpful discoveries
- The exact timeout value `51.6s` made it immediately obvious where to look in the source.
- The `_CPU_MULTILINGUAL_TIMEOUT_FACTOR` was absent entirely — the system had no concept of multilingual cost in its timeout calculation.
- `OverlapChunker` is not used in TTY notes-first mode (which routes through `SessionSpool`/`IntervalPlanner`) but is used in the legacy stdin path, making it lower-priority for the specific failing session but still worth fixing.

---

## 4. Final Solution Used (For This Chat)

### Fix 1: Eliminate redundant array allocation in `_pcm16le_to_float_array`

**File:** `src/narada/asr/faster_whisper_engine.py`

**Change:** Replaced `(samples.astype(np.float32) / 32768.0).copy()` with `astype` into a named variable followed by in-place `/=`, then return. Reduced from 4 NumPy allocations to 2 per call.

---

### Fix 2: Remove `bytes()` wrapper in worker audio deserialization

**File:** `src/narada/asr/faster_whisper_engine.py`

**Change:** Removed the `bytes(...)` wrap around `request.get("audio", b"")` before passing to `np.frombuffer`. The `.copy()` after `frombuffer` is retained because `frombuffer` returns a read-only view and the worker needs a writable array.

---

### Fix 3: Persistent read handle in `SessionSpool`

**File:** `src/narada/live_notes.py`

**Change:** Added `self._read_handle = self.data_path.open("rb")` in `__init__` (file is guaranteed to exist because the write handle creates it). `read_range` now calls `self._read_handle.seek(start_byte)` and reads from the persistent handle. `close()` closes it alongside the other handles.

---

### Fix 4: Multilingual timeout factor in CPU ASR timeout calculation

**File:** `src/narada/asr/faster_whisper_engine.py`

**Change:** Added module-level constant `_CPU_MULTILINGUAL_TIMEOUT_FACTOR: float = 1.4`. Added `multilingual: bool = False` parameter to `_compute_worker_transcribe_timeout_s`. When `multilingual=True` on the CPU path, the per-audio-second factor is multiplied by 1.4 before computing the timeout. Both call sites in `_transcribe_gpu_guarded` now pass `multilingual=multilingual`. For the exact session that failed (12s balanced CPU), the timeout budget changes from `51.6s` to `60.2s`.

---

### Fix 5: Offset cursor in `OverlapChunker` to avoid O(n) front-deletion

**File:** `src/narada/pipeline.py`

**Change:** Added `_read_offset: int = 0` to `OverlapChunker`. `_drain_ready_windows` now advances `_read_offset` by `stride` per window rather than calling `del self._buffer[:stride]`. A single compaction (`del self._buffer[:self._read_offset]` + reset) is performed once at the end of the method, only when `_read_offset >= chunk_bytes`. `pending_duration_s` uses `len(self._buffer) - self._read_offset`. `flush` reads `bytes(self._buffer[self._read_offset:])` and resets both buffer and offset.

---

## 5. Tools, APIs, and Concepts Used

| Technology / Concept | How it appeared in this chat |
|---|---|
| **faster-whisper** | The ASR engine in use. Its CPU worker runs in a subprocess via `multiprocessing.Process`. Timeout and RTF behavior were central to the diagnosis. |
| **multiprocessing / spawn context** | Worker process for faster-whisper is spawned via `mp.get_context("spawn")`. Audio is serialized as `bytes` through a `Queue`. The redundant `bytes()` wrap was rooted here. |
| **NumPy** | Used for PCM-to-float conversion and audio resampling. The redundant `.copy()` and in-place `/=` optimization were NumPy-specific idioms. |
| **bytearray cursor pattern** | Replacing front-deletion with an offset cursor is a standard technique for ring-buffer and sliding-window implementations. Applied to `OverlapChunker`. |
| **RTF (Real-Time Factor)** | `total_processing_seconds / total_audio_seconds`. RTF > 1 means the system is falling behind real-time. Values seen: 1.05 to 3.62. |
| **SessionSpool** | A temp-directory-backed audio spool for the TTY notes-first pipeline. The `read_range` pattern of opening a new file descriptor per read was the inefficiency addressed. |
| **IntervalPlanner** | Slices the spool into fixed-interval ASR tasks. Feeds into the single-threaded ASR worker loop in the CLI. |
| **Git branching** | All changes were made on `performance-tweaks` branch, committed in logical groups, and pushed to origin. |
| **pytest** | Test suite used to verify zero regressions after each change. Pre-existing failure in `test_devices.py` was confirmed pre-existing before any changes were made. |

---

## 6. Lessons Learned (For This Episode)

- **RTF > 1 is a cascade trigger, not just a lag indicator.** Once transcription falls behind real-time, the backlog only grows during active capture. Any task timeout set relative to audio duration will start firing, making the situation exponentially worse.

- **Timeout formulas need to account for the mode they're running in.** The CPU timeout was calibrated for single-language operation. Multilingual mode adds a fixed per-segment overhead that invalidates a linear audio-duration budget. Timeouts should be parameterized by processing mode, not just audio length.

- **Read the exact timeout value in the error message before guessing the cause.** `51.6s` directly reproduced the formula `30 + 12 × 1.80`, pinpointing the constant and the missing multiplier without any guesswork.

- **`.copy()` after `.astype()` with a different dtype is always redundant.** `astype` with a new dtype allocates a fresh array unconditionally. The copy buys nothing and costs a full array allocation at every call site.

- **`bytes(x)` before `np.frombuffer(x)` is always redundant if `x` is already bytes-like.** `np.frombuffer` accepts any object exporting the buffer protocol. The outer `bytes()` call creates a throwaway copy of the entire payload.

- **File handles that are opened and closed repeatedly on a hot path should be opened once.** The pattern `with open(path) as f: f.seek(...); return f.read(...)` inside a locked method that is called at a regular interval is a straightforward candidate for a persistent handle.

- **Front-deletion from a `bytearray` is O(n).** Any sliding-window or overlap-chunking implementation that deletes from the front of a mutable buffer on each stride step is doing O(n) work per step. An offset cursor defers compaction and makes the common case O(1).

- **Confirm that a test failure is pre-existing before declaring a regression.** The `test_devices.py` failure existed on `main` before any changes were made. Stashing and retesting on the base confirmed this, preventing a false alarm.

- **Separate commits by concern, not by file.** Fixes 1 and 2 both touched `faster_whisper_engine.py` but were logically about different systems (host-side conversion vs. worker-side deserialization). Fix 4 touched the same file later for a third concern. Grouping them by concern — copy efficiency, spool I/O, timeout correctness, chunker algorithm — produced a cleaner history than grouping by file.
