"""Tests for the extracted run_pipeline orchestration and the CLI adapter.

The heavy ML/ffmpeg steps are mocked, so these run without TensorFlow,
faster-whisper, inaSpeechSegmenter, or ffmpeg installed.
"""

import sys
from unittest import mock

import pytest

from sanji import cli
from sanji.pipeline import DurationExceededError, PipelineParams, PipelineResult, run_pipeline


class TestCliAdapter:
    """cli.main() should only parse args and delegate to run_pipeline."""

    def test_builds_params_from_argv(self, monkeypatch, tmp_path):
        argv = [
            "sanji",
            "https://youtu.be/abc",
            "-o",
            str(tmp_path),
            "--expect-songs",
            "3",
            "--no-transcribe",
            "--threshold",
            "0.5",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with mock.patch.object(cli, "run_pipeline") as run:
            cli.main()

        assert run.call_count == 1
        params = run.call_args.args[0]
        assert isinstance(params, PipelineParams)
        assert params.input == "https://youtu.be/abc"
        assert params.output_dir == tmp_path
        assert params.expect_songs == 3
        assert params.no_transcribe is True
        assert params.threshold == 0.5
        assert params.dry_run is False

    def test_missing_local_file_exits_1(self, monkeypatch, tmp_path):
        missing = tmp_path / "nope.mp4"
        monkeypatch.setattr(sys, "argv", ["sanji", str(missing)])

        with mock.patch.object(
            cli,
            "run_pipeline",
            side_effect=FileNotFoundError(f"file not found: {missing}"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()

        assert exc_info.value.code == 1


class TestRunPipeline:
    def test_missing_local_file_raises(self, tmp_path):
        params = PipelineParams(
            input=str(tmp_path / "missing.mp4"), output_dir=tmp_path / "out"
        )
        with pytest.raises(FileNotFoundError):
            run_pipeline(params)

    @mock.patch("sanji.pipeline.split_video")
    @mock.patch("sanji.pipeline.write_manifest")
    @mock.patch("sanji.pipeline.refine_splits_with_transcription")
    @mock.patch("sanji.pipeline.find_split_points")
    @mock.patch("sanji.pipeline.merge_short_regions")
    @mock.patch("sanji.pipeline.find_song_regions")
    @mock.patch("sanji.pipeline.compute_music_density")
    @mock.patch("sanji.pipeline.classify_audio")
    @mock.patch("sanji.pipeline.extract_audio")
    @mock.patch("sanji.pipeline.get_video_duration")
    def test_local_file_orchestration(
        self,
        mock_duration,
        mock_extract,
        mock_classify,
        mock_density,
        mock_regions,
        mock_merge,
        mock_splits,
        mock_refine,
        mock_manifest,
        mock_split,
        tmp_path,
    ):
        video = tmp_path / "source.mp4"
        video.write_bytes(b"x")
        out = tmp_path / "out"
        out.mkdir()
        # split_video really creates these files; the pipeline stats them for the
        # summary, so the mock must point at files that exist.
        segment = out / "reaction_01.mp4"
        segment.write_bytes(b"video-bytes")

        mock_duration.return_value = 600.0
        mock_classify.return_value = []
        mock_density.return_value = ([], [])
        mock_regions.return_value = [(0.0, 100.0), (200.0, 300.0)]
        mock_merge.return_value = [(0.0, 100.0), (200.0, 300.0)]
        mock_splits.return_value = [150.0]
        mock_refine.return_value = [150.0]
        mock_split.return_value = [segment]
        mock_manifest.return_value = out / "reaction_MANIFEST.csv"

        params = PipelineParams(input=str(video), output_dir=out)
        result = run_pipeline(params)

        # Pipeline steps invoked in order.
        mock_duration.assert_called_once()
        mock_extract.assert_called_once()
        mock_classify.assert_called_once()
        mock_refine.assert_called_once()
        mock_split.assert_called_once()
        mock_manifest.assert_called_once()

        assert isinstance(result, PipelineResult)
        assert result.duration == 600.0
        assert result.split_points == [150.0]
        assert result.manifest_path == out / "reaction_MANIFEST.csv"
        assert result.segment_files == [segment]

    @mock.patch("sanji.pipeline.refine_splits_with_transcription")
    @mock.patch("sanji.pipeline.find_split_points")
    @mock.patch("sanji.pipeline.merge_short_regions")
    @mock.patch("sanji.pipeline.find_song_regions")
    @mock.patch("sanji.pipeline.compute_music_density")
    @mock.patch("sanji.pipeline.classify_audio")
    @mock.patch("sanji.pipeline.extract_audio")
    @mock.patch("sanji.pipeline.get_video_duration")
    def test_local_file_creates_missing_work_dir(
        self,
        mock_duration,
        mock_extract,
        mock_classify,
        mock_density,
        mock_regions,
        mock_merge,
        mock_splits,
        mock_refine,
        tmp_path,
    ):
        """A non-existent work_dir must exist before ffmpeg writes audio.wav (#73).

        The worker passes work_dir=<scratch>/work without creating it; only the
        URL path created it implicitly, so direct-upload (local file) jobs failed
        at extract_audio with ffmpeg exit 1.
        """
        video = tmp_path / "source.mp4"
        video.write_bytes(b"x")
        work_dir = tmp_path / "job" / "work"  # deliberately not created

        mock_duration.return_value = 600.0
        mock_classify.return_value = []
        mock_density.return_value = ([], [])
        mock_regions.return_value = []
        mock_merge.return_value = []
        mock_splits.return_value = []
        mock_refine.return_value = []

        params = PipelineParams(
            input=str(video), output_dir=tmp_path / "out", dry_run=True
        )
        run_pipeline(params, work_dir=work_dir)

        assert work_dir.is_dir()
        (audio_path,) = mock_extract.call_args.args[1:]
        assert audio_path.parent == work_dir

    @mock.patch("sanji.pipeline.get_video_duration")
    def test_duration_cap_raises_before_processing(self, mock_duration, tmp_path):
        """Enforcement B: DurationExceededError raised before expensive steps."""
        video = tmp_path / "long.mp4"
        video.write_bytes(b"x")
        mock_duration.return_value = 3 * 60 * 60  # 3 hours

        params = PipelineParams(
            input=str(video),
            output_dir=tmp_path / "out",
            max_duration_seconds=2 * 60 * 60,  # 2-hour cap
        )
        with pytest.raises(DurationExceededError):
            run_pipeline(params)

    @mock.patch("sanji.pipeline.split_video")
    @mock.patch("sanji.pipeline.write_manifest")
    @mock.patch("sanji.pipeline.refine_splits_with_transcription")
    @mock.patch("sanji.pipeline.find_split_points")
    @mock.patch("sanji.pipeline.merge_short_regions")
    @mock.patch("sanji.pipeline.find_song_regions")
    @mock.patch("sanji.pipeline.compute_music_density")
    @mock.patch("sanji.pipeline.classify_audio")
    @mock.patch("sanji.pipeline.extract_audio")
    @mock.patch("sanji.pipeline.get_video_duration")
    def test_duration_cap_none_allows_any_length(
        self,
        mock_duration,
        mock_extract,
        mock_classify,
        mock_density,
        mock_regions,
        mock_merge,
        mock_splits,
        mock_refine,
        mock_manifest,
        mock_split,
        tmp_path,
    ):
        """max_duration_seconds=None means no cap (legacy CLI paths)."""
        video = tmp_path / "long.mp4"
        video.write_bytes(b"x")
        segment = tmp_path / "out" / "reaction_01.mp4"
        segment.parent.mkdir()
        segment.write_bytes(b"bytes")

        mock_duration.return_value = 10 * 60 * 60  # 10 hours
        mock_classify.return_value = []
        mock_density.return_value = ([], [])
        mock_regions.return_value = []
        mock_merge.return_value = []
        mock_splits.return_value = []
        mock_refine.return_value = []
        mock_split.return_value = [segment]
        mock_manifest.return_value = tmp_path / "out" / "manifest.csv"

        params = PipelineParams(
            input=str(video),
            output_dir=tmp_path / "out",
            max_duration_seconds=None,
        )
        result = run_pipeline(params)
        assert result.duration == 10 * 60 * 60

    @mock.patch("sanji.pipeline.get_video_duration")
    @pytest.mark.parametrize(
        "plan_cap_hours,video_hours,should_raise",
        [
            (2, 2, False),      # exactly at cap → allowed
            (2, 1, False),      # under cap → allowed
            (2, 3, True),       # over cap → rejected
        ],
    )
    def test_duration_cap_boundary(
        self, mock_duration, plan_cap_hours, video_hours, should_raise, tmp_path
    ):
        """Duration boundary: at or under cap passes; over cap raises."""
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        mock_duration.return_value = float(video_hours * 3600)

        params = PipelineParams(
            input=str(video),
            output_dir=tmp_path / "out",
            max_duration_seconds=plan_cap_hours * 3600,
            dry_run=True,
            no_transcribe=True,
        )

        if should_raise:
            with pytest.raises(DurationExceededError):
                run_pipeline(params)
        else:
            # The pipeline will try to run classify_audio etc. and fail without
            # mocks — we just care it doesn't raise DurationExceededError first.
            with mock.patch("sanji.pipeline.extract_audio"), \
                 mock.patch("sanji.pipeline.classify_audio", return_value=[]), \
                 mock.patch("sanji.pipeline.compute_music_density", return_value=([], [])), \
                 mock.patch("sanji.pipeline.find_song_regions", return_value=[]), \
                 mock.patch("sanji.pipeline.merge_short_regions", return_value=[]), \
                 mock.patch("sanji.pipeline.find_split_points", return_value=[]):
                result = run_pipeline(params)
            assert result.duration == float(video_hours * 3600)

    @mock.patch("sanji.pipeline.split_video")
    @mock.patch("sanji.pipeline.find_split_points")
    @mock.patch("sanji.pipeline.merge_short_regions")
    @mock.patch("sanji.pipeline.find_song_regions")
    @mock.patch("sanji.pipeline.compute_music_density")
    @mock.patch("sanji.pipeline.classify_audio")
    @mock.patch("sanji.pipeline.extract_audio")
    @mock.patch("sanji.pipeline.get_video_duration")
    def test_dry_run_skips_splitting(
        self,
        mock_duration,
        mock_extract,
        mock_classify,
        mock_density,
        mock_regions,
        mock_merge,
        mock_splits,
        mock_split,
        tmp_path,
    ):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")

        mock_duration.return_value = 120.0
        mock_classify.return_value = []
        mock_density.return_value = ([], [])
        mock_regions.return_value = []
        mock_merge.return_value = []
        mock_splits.return_value = []

        params = PipelineParams(
            input=str(video),
            output_dir=tmp_path / "out",
            dry_run=True,
            no_transcribe=True,
        )
        result = run_pipeline(params)

        mock_split.assert_not_called()
        assert result.segment_files == []
        assert result.manifest_path is None
