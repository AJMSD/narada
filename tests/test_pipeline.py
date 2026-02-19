from narada.asr.base import TranscriptSegment
from narada.pipeline import ConfidenceGate


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
