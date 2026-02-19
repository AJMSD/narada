from narada.asr.base import TranscriptSegment
from narada.pipeline import ConfidenceGate, OverlapChunker


def test_confidence_gate_commits_confident_segments() -> None:
    gate = ConfidenceGate(0.65)
    committed = gate.ingest(
        [
            TranscriptSegment(
                text="low confidence", confidence=0.5, start_s=0.0, end_s=1.0, is_final=False
            ),
            TranscriptSegment(
                text="high confidence", confidence=0.9, start_s=1.0, end_s=2.0, is_final=False
            ),
        ]
    )
    assert [item.text for item in committed] == ["high confidence"]
    pending = gate.drain_pending()
    assert [item.text for item in pending] == ["low confidence"]


def test_overlap_chunker_emits_with_overlap_stride() -> None:
    chunker = OverlapChunker(chunk_duration_s=2.0, overlap_duration_s=0.5)
    # 4 seconds of mono PCM16 at 4 Hz => 16 samples => 32 bytes
    frame = b"\x01\x00" * 16
    windows = chunker.ingest(frame, sample_rate_hz=4, channels=1)
    assert len(windows) == 2
    # chunk bytes = 2s * 4 * 2 = 16 bytes
    assert len(windows[0].pcm_bytes) == 16
    assert len(windows[1].pcm_bytes) == 16
    # Remaining buffer should hold overlap tail.
    assert chunker.pending_duration_s() == 1.0


def test_overlap_chunker_flush_respects_min_duration() -> None:
    chunker = OverlapChunker(chunk_duration_s=2.0, overlap_duration_s=0.5, min_flush_duration_s=1.0)
    _ = chunker.ingest(b"\x01\x00" * 3, sample_rate_hz=4, channels=1)  # 0.75 seconds
    assert chunker.flush(force=False) == []
    forced = chunker.flush(force=True)
    assert len(forced) == 1


def test_overlap_chunker_on_format_change_flushes_existing_buffer() -> None:
    chunker = OverlapChunker(chunk_duration_s=2.0, overlap_duration_s=0.5, min_flush_duration_s=0.0)
    _ = chunker.ingest(b"\x01\x00" * 4, sample_rate_hz=4, channels=1)
    emitted = chunker.ingest(b"\x01\x00" * 4, sample_rate_hz=8, channels=1)
    assert len(emitted) == 1
    assert emitted[0].sample_rate_hz == 4
