import pytest

from narada.start_runtime import mono_frame_to_pcm16le, parse_input_line


def test_parse_input_line_plain_text() -> None:
    parsed = parse_input_line("hello world", mode="mic")
    assert parsed is not None
    assert parsed.text == "hello world"
    assert parsed.audio is None
    assert parsed.confidence == 1.0


def test_parse_input_line_json_text_with_confidence() -> None:
    parsed = parse_input_line('{"text":"hello","confidence":0.72}', mode="system")
    assert parsed is not None
    assert parsed.text == "hello"
    assert parsed.confidence == 0.72


def test_parse_input_line_legacy_mixed_json_is_rejected() -> None:
    payload = (
        '{"mic":{"samples":[0.1,0.1,0.3,0.3],"sample_rate_hz":16000,"channels":2},'
        '"system":{"samples":[0.2,-0.2],"sample_rate_hz":8000,"channels":1}}'
    )
    with pytest.raises(ValueError, match="Mixed JSON payloads are no longer supported"):
        parse_input_line(payload, mode="mic")


def test_parse_input_line_audio_payload_for_single_mode() -> None:
    parsed = parse_input_line(
        '{"audio":{"samples":[0.1,-0.1],"sample_rate_hz":16000,"channels":1}}',
        mode="mic",
    )
    assert parsed is not None
    assert parsed.audio is not None
    assert parsed.audio.sample_rate_hz == 16000
    assert parsed.text is None


def test_mono_frame_to_pcm16le_clips_and_encodes() -> None:
    parsed = parse_input_line(
        '{"audio":{"samples":[2.0,0.0,-2.0],"sample_rate_hz":16000,"channels":1}}',
        mode="system",
    )
    assert parsed is not None and parsed.audio is not None
    pcm = mono_frame_to_pcm16le(parsed.audio)
    assert len(pcm) == 6
