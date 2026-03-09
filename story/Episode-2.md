# Episode 2: System Audio Capture — A Three-Layer Bug Hunt

---

## 1. Context for This Episode

This chat was about getting `narada start --mode system` to actually work end-to-end on Windows. The user had a working CLI and wanted to capture audio playing through their speakers (system/loopback audio) and transcribe it in near real-time, with the transcript served live over HTTP on the LAN.

The rough goal: run a single command, point it at an output device, and have Narada record + transcribe what the machine is playing — without a microphone involved.

What actually happened was three separate bugs, each masking the next, that had to be peeled back one at a time before the user could even get a stream open.

---

## 2. Main Problems We Faced

### Problem 1 — Hardcoded `channels=1` breaks WASAPI loopback on stereo devices

**Symptoms:**
```
PortAudioError: Error opening RawInputStream: Invalid number of channels [PaErrorCode -9998]
```

**Why it was a problem:**  
`open_system_capture` always passed `channels=1` to PortAudio. Most Windows WASAPI output devices are stereo-only and reject mono capture requests. The user's device (DitooPro Bluetooth headphones, device 21) reported this immediately. There was no channel introspection before the stream was opened — it just crashed.

---

### Problem 2 — WASAPI `WasapiSettings(loopback=True)` applied to an MME device ID

**Symptoms:**
```
PortAudioError: Error opening RawInputStream: Incompatible host API specific stream info [PaErrorCode -9984]
```

**Why it was a problem:**  
Even after fixing Problem 1, all output devices (Realtek headphones, Realtek speakers) still failed. The error code changed (-9998 → -9984), revealing a different root cause: the device IDs shown by `narada devices` were MME IDs (e.g. ID 7, 8), not WASAPI IDs. Narada was attaching `WasapiSettings(loopback=True)` — a WASAPI-specific option — to an MME stream index, which PortAudio flatly rejected.

---

### Problem 3 — MME device IDs winning deduplication over WASAPI IDs

**Symptoms:**  
`narada devices` showed IDs 1, 4, 5, 7, 8 at the top of the list — all MME entries — instead of their WASAPI equivalents (20, 21, 22, 25, 26, 27).

**Why it was a problem:**  
The deduplication was designed to pick one canonical ID per device. It was broken because `_device_preference_key` sorted by `(is_default, hostapi_rank, id)` — making `is_default` the primary criterion. The Windows system default happened to be an MME device, so MME always won. Users were then given IDs that could never be used with WASAPI loopback.

---

### Problem 4 — MME names are truncated at 32 characters (discovered but not fixed this episode)

**Symptoms:**  
ID 1 showed as `"Microphone (2- Fifine Microphon"` (no closing paren), while ID 26 showed as `"Microphone (2- Fifine Microphone)"` (full name). These survived deduplication as two separate entries because their canonical names didn't match.

**Why it was a problem:**  
Even after fixing the sorting, the deduplication couldn't collapse truncated MME names onto their full WASAPI counterparts. This left junk MME entries in the device list confusing users. The root cause is a Windows OS limitation (`WAVEOUTCAPS.szPname` is a 32-char fixed buffer) — not in Narada's code.

---

### Problem 5 — Bluetooth HFP devices give no useful error

**Symptoms:**  
Device 21 (DitooPro Bluetooth headphones via `bthhfenum.sys`) failed with the same channel count error as a regular device would, giving the user no indication that BT HFP is architecturally unsupported for loopback.

**Why it was a problem:**  
The user would try the command, get a PortAudio error, have no idea it was a Bluetooth protocol limitation rather than a fixable configuration issue.

---

## 3. Debugging Path & Options Considered

### For Problem 1 (channels=1)

**Debugging steps:**
- Read the traceback: `Pa_OpenStream` failing with `-9998` = `paInvalidNumChannels`
- Traced back through `open_system_capture` → `_open_raw_input_stream` → hardcoded `channels=1` default
- Confirmed WASAPI stereo devices reject mono open requests

**Options considered:**
- **Option A (chosen):** Query `sd.query_devices()` before opening, read `max_output_channels` (for loopback, which exposes an output endpoint), use that as the actual channel count. Downmix stereo → mono in `CaptureHandle.read_frame()` before returning to callers — so all downstream code always sees mono.
- **Option B (rejected):** Try to open with 1 and let the user figure out what went wrong. Clearly bad UX.

**Key discovery:** For WASAPI loopback, the relevant field is `max_output_channels`, not `max_input_channels`, because the endpoint being captured is an output endpoint.

---

### For Problem 1 follow-up — channel count query can lie

**Debugging steps:**
- After implementing the query fix, Bluetooth device 21 still failed with the same error
- Realised `sd.query_devices()` returns the device's maximum capability, not its current shared-mode mix format
- For BT HFP specifically, the device switches to mono/8000Hz when HFP mode activates, but `max_output_channels` still returns 2

**Options considered:**
- **Option A (chosen):** Retry with fallback channel counts `[detected, 2, 1]` on `-9998` only. Any other PortAudio error surfaces immediately.
- **Option B:** Query the actual WASAPI shared-mode mix format directly. Would require calling into Windows COM APIs, far too complex.
- **Dead end:** Trusting `max_output_channels` unconditionally — doesn't work for BT HFP or devices where the mix format differs from the hardware max.

---

### For Problem 2 (MME IDs in device list)

**Debugging steps:**
- User tried Realtek headphones, Realtek speakers, BT speaker — all failed with `-9984`
- Read the error: "Incompatible host API specific stream info" — this is specifically what PortAudio returns when you attach WASAPI-specific options to a non-WASAPI stream
- Ran a Python script to dump all raw PortAudio device entries with their host API indices
- Confirmed that "Speakers (2- Realtek(R) Audio)" exists as ID 9 (MME), ID 19 (DirectSound), ID 22 (WASAPI)
- Traced back to `_device_preference_key`: sorting tuple was `(is_default, hostapi_rank, id)`
- The Windows system default was pointing to ID 7 (MME) → `is_default=True` → scored 0 → won deduplication over WASAPI ID 22 (`is_default=False` → scored 1)

**Options considered:**
- **Option A (chosen):** Swap sort key order to `(hostapi_rank, is_default, id)`. One-line change. Host API quality always primary; `is_default` becomes a tiebreaker within the same API tier.
- **Option B:** Filter MME/DirectSound out entirely before deduplication. Cleaner, but needs the truncated-name issue to also be solved to avoid any remaining MME stragglers.
- **Option C explored later:** The truncation issue means Option A alone still leaves some MME entries (can't dedupe what doesn't match). Option B is the correct long-term path but was identified, not implemented, this episode.

---

### For Problem 5 (BT HFP error quality)

**Debugging steps:**
- Noticed BT devices use `bthhfenum.sys` in their name
- BT HFP devices will always fail regardless of channel count — the driver doesn't expose a PCM loopback endpoint

**Options considered:**
- **Option A (chosen):** Detect `bthhfenum` substring in device name and raise a specific, actionable error message
- **Option B:** Let it fall through with the generic PortAudio error message. Poor UX.

---

## 4. Final Solution Used (For This Chat)

### Fix 1 — Auto-detect native channel count and downmix (`capture.py`)

Added `_query_native_channels(device_id, loopback=bool)` which calls `sd.query_devices()` and reads `max_output_channels` for loopback or `max_input_channels` for mic. Falls back to safe defaults (2 for loopback, 1 for mic) if the query fails.

Added `_downmix_pcm16le_to_mono(pcm_bytes, channels)` which averages N interleaved PCM-16 LE channels per frame into a single mono track, working directly on bytes with no float round-trip.

Updated `CaptureHandle` to accept `native_channels` — the hardware count used to open the stream. `read_frame()` downmixes when `native_channels > 1`. All callers always see `channels=1`.

Updated `open_system_capture` to remove the `channels` parameter entirely, auto-detect the hardware count, open the stream at that count, and always expose `channels=1` upward.

**Files:** `src/narada/audio/capture.py`

---

### Fix 2 — Loopback stream retry with fallback channel counts (`capture.py`)

Added `_open_loopback_stream(...)` which tries `[native_channels, 2, 1]` (deduplicated) in order. Retries only on `-9998` (invalid channel count). Any other PortAudio error surfaces immediately via `_raise_loopback_error`.

Added `_raise_loopback_error(device_name, cause)` which detects BT HFP devices by `bthhfenum` substring and raises a targeted, actionable `CaptureError` message instead of exposing raw PortAudio codes.

`open_system_capture` now delegates to `_open_loopback_stream` and uses `opened_channels` (the count that actually worked) for the `CaptureHandle`, not the queried native count.

**Files:** `src/narada/audio/capture.py`

---

### Fix 3 — Host API rank as primary deduplication key (`devices.py`)

Changed `_device_preference_key` from:
```python
return (0 if device.is_default else 1, _hostapi_rank(...), device.id)
```
to:
```python
return (_hostapi_rank(...), 0 if device.is_default else 1, device.id)
```

WASAPI always wins over DirectSound and MME, regardless of which one the OS has set as the current default. `is_default` remains a tiebreaker within the same host API.

**Files:** `src/narada/devices.py`

---

### Remaining open issue — MME name truncation

Not fixed this episode. MME entries whose names are truncated at 32 chars survive deduplication because they don't match their full-name WASAPI counterparts. The correct fix (filtering MME/DirectSound out before deduplication on Windows) was identified and agreed on, but left for the next episode.

---

## 5. Tools, APIs, and Concepts Used

| Technology / Concept | How it was used / what we learned |
|---|---|
| **WASAPI loopback** | The only Windows mechanism for capturing system audio. Requires `WasapiSettings(loopback=True)` passed as `extra_settings` to `sd.RawInputStream`. Only works with WASAPI device IDs. |
| **PortAudio error codes** | `-9998` = `paInvalidNumChannels` (retryable with different channel count). `-9984` = `paIncompatibleHostApiSpecificStreamInfo` (wrong host API settings on wrong device). |
| **sounddevice** | Python wrapper over PortAudio. `sd.query_devices(id)` returns a dict with `max_input_channels`, `max_output_channels`, `hostapi` index. `sd.query_hostapis()` returns names indexed by that integer. |
| **WASAPI shared-mode mix format** | Distinct from `max_output_channels`. The format the Windows audio engine uses for mixing. A device can report `max_output_channels=2` but operate in mono HFP mode. |
| **Windows MME `WAVEOUTCAPS.szPname`** | A 32-character fixed buffer in the Windows MME API. Device names longer than 32 chars are silently truncated. WASAPI uses `IPropertyStore` which has no such limit. |
| **Bluetooth HFP** | Hands-Free Profile routes audio through `bthhfenum.sys`. Switches device to mono/8000Hz when active. Does not expose a PCM loopback endpoint. Detectable by driver name substring. |
| **`_device_preference_key` sort tuple** | Controls which host-API duplicate wins deduplication. Key lesson: tuple element order determines priority. |
| **`_downmix_pcm16le_to_mono`** | Averages interleaved N-channel PCM-16 LE into mono by integer arithmetic on raw bytes, avoiding a float conversion round-trip. |

---

## 6. Lessons Learned (For This Episode)

- **The deduplication sort key element order is priority order.** Putting `is_default` first made OS defaults always win, regardless of API quality. The fix was one character swap that changed which factor dominated.

- **Verify what IDs you're actually passing before blaming the capture logic.** Two of the three bugs showed up as capture failures, but the real cause was the wrong device ID being selected upstream.

- **`max_output_channels` ≠ current mix format.** Querying capabilities before opening a stream improves reliability but is not a guarantee. The actual channel count a device accepts depends on its current shared-mode format, which can change dynamically (especially with BT HFP).

- **Retry loops with constrained fallbacks are the right pattern for hardware negotiation.** Don't fail hard on the first rejected channel count — try the detected value, then common values, then give up with a clear message.

- **Attach host-API-specific settings only to matching host-API device IDs.** `WasapiSettings(loopback=True)` on an MME ID is guaranteed to fail with `-9984`, no matter what else you do.

- **Filter at the source, not at the surface.** The right place to exclude legacy APIs (MME, DirectSound) on Windows is during enumeration, before deduplication — not inside deduplication logic that tries to match truncated names to full ones.

- **OS limitations propagate silently.** The MME 32-char name truncation happens in Windows before Narada sees any data. There is no code path in Narada that could prevent it. The correct response is to not use that API at all.

- **BT HFP is a protocol mismatch, not a configuration problem.** No amount of channel-count tuning will make a Hands-Free Profile endpoint work as a loopback source. Detecting it early and explaining it clearly is the only productive path.

- **When error codes change between runs (−9998 → −9984), you've fixed one bug and revealed the next.** A new error code after a change is usually progress, not a regression — it means you peeled a layer.
