"""Tests for sanji.transcription — split refinement with faster-whisper mocked.

``faster_whisper.WhisperModel`` and the ffmpeg-backed ``extract_gap_audio`` are
mocked so the pattern-matching logic runs without a model download or GPU.
"""

from unittest import mock

from sanji.transcription import _find_pattern_timestamp, refine_splits_with_transcription


class _FakeWord:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word


class _FakeSegment:
    def __init__(self, words):
        self.words = words


def _model_returning(words):
    model = mock.Mock()
    model.transcribe = mock.Mock(return_value=([_FakeSegment(words)], None))
    return model


class TestFindPatternTimestamp:
    def test_returns_word_start_at_match(self):
        words = [(100.0, 100.5, "okay"), (100.6, 101.0, "next"), (101.1, 101.5, "up")]
        result = _find_pattern_timestamp(words, "okay next up", r"\bnext up\b")
        assert result == (100.6, "next up")

    def test_returns_none_when_no_match(self):
        words = [(0.0, 0.5, "hello")]
        assert _find_pattern_timestamp(words, "hello", r"\bworld\b") is None


class TestRefineSplitsWithTranscription:
    def test_moves_split_to_detected_intro_phrase(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        song_regions = [(0.0, 100.0), (200.0, 300.0)]
        words = [
            _FakeWord(0.0, 0.5, "okay"),
            _FakeWord(0.6, 1.0, "next"),
            _FakeWord(1.1, 1.5, "up"),
            _FakeWord(1.6, 2.0, "we"),
            _FakeWord(2.1, 2.5, "have"),
        ]
        with mock.patch("faster_whisper.WhisperModel", return_value=_model_returning(words)), \
                mock.patch("sanji.transcription.extract_gap_audio"):
            refined = refine_splits_with_transcription([180.0], song_regions, audio)

        # "next" abs time = gap_start(100) + 0.6 = 100.6; new split = max(100+2, 100.6-3) = 102.0
        assert refined == [102.0]

    def test_keeps_split_when_no_intro_found(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        song_regions = [(0.0, 100.0), (200.0, 300.0)]
        words = [_FakeWord(0.0, 0.5, "hello"), _FakeWord(0.6, 1.0, "world")]
        with mock.patch("faster_whisper.WhisperModel", return_value=_model_returning(words)), \
                mock.patch("sanji.transcription.extract_gap_audio"):
            refined = refine_splits_with_transcription([180.0], song_regions, audio)

        assert refined == [180.0]
