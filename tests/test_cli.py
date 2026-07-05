"""Tests for sanji.cli — argument parsing and the ML-mocked pipeline.

The inaSpeechSegmenter (``classify_audio``) and faster-whisper
(``refine_splits_with_transcription``) calls are mocked so the pipeline runs
without downloading models or requiring a GPU.
"""

import sys
from unittest import mock

import pytest

from sanji.cli import build_parser, main


@pytest.fixture
def local_video(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video")
    return video


def _fake_segments():
    """Two music regions (0-100s, 200-300s) separated by a speech gap."""
    return [("music", 0, 100), ("speech", 100, 200), ("music", 200, 300)]


class TestArgumentParsing:
    def test_defaults(self):
        args = build_parser().parse_args(["video.mp4"])
        assert args.input == "video.mp4"
        assert args.output == "./output"
        assert args.validate is False
        assert args.no_transcribe is False
        assert args.dry_run is False
        assert args.expect_songs is None

    def test_output_flag(self):
        args = build_parser().parse_args(["video.mp4", "-o", "clips"])
        assert args.output == "clips"

    def test_missing_required_input_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args([])
        assert exc.value.code == 2

    def test_invalid_float_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["video.mp4", "--min-gap", "not-a-number"])
        assert exc.value.code == 2

    def test_help_exits_0(self, capsys):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--help"])
        assert exc.value.code == 0
        assert "Split music reaction videos" in capsys.readouterr().out


class TestMain:
    def test_help_exits_0(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sanji", "--help"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_missing_local_file_exits_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["sanji", str(tmp_path / "missing.mp4")])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "file not found" in capsys.readouterr().out

    def test_dry_run_pipeline_with_mocked_classifier(
        self, local_video, tmp_path, monkeypatch, capsys
    ):
        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sanji",
                str(local_video),
                "-o",
                str(out_dir),
                "--dry-run",
                "--no-transcribe",
                "--threshold",
                "0.15",
                "--min-segment",
                "30",
                "--min-song",
                "30",
            ],
        )
        with (
            mock.patch(
                "sanji.pipeline.get_video_duration", return_value=300.0
            ) as m_dur,
            mock.patch("sanji.pipeline.extract_audio") as m_extract,
            mock.patch(
                "sanji.pipeline.classify_audio", return_value=_fake_segments()
            ) as m_classify,
        ):
            main()

        m_dur.assert_called_once()
        m_extract.assert_called_once()
        m_classify.assert_called_once()
        out = capsys.readouterr().out
        assert "song_regions_detected" in out
        assert "dry_run_skipping_split" in out

    def test_transcription_step_invoked_when_not_skipped(
        self, local_video, tmp_path, monkeypatch
    ):
        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sanji",
                str(local_video),
                "-o",
                str(out_dir),
                "--dry-run",
                "--threshold",
                "0.15",
                "--min-segment",
                "30",
                "--min-song",
                "30",
            ],
        )
        with (
            mock.patch("sanji.pipeline.get_video_duration", return_value=300.0),
            mock.patch("sanji.pipeline.extract_audio"),
            mock.patch("sanji.pipeline.classify_audio", return_value=_fake_segments()),
            mock.patch(
                "sanji.pipeline.refine_splits_with_transcription",
                side_effect=lambda split_points, *a, **k: split_points,
            ) as m_refine,
        ):
            main()

        m_refine.assert_called_once()
