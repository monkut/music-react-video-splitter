"""split_video ffmpeg invocation tests (issue #39).

ffmpeg itself is mocked — these assert the command structure: input-side
seeking (-ss before -i) so each segment decodes only its own range, and -t
(duration) instead of -to, whose semantics change with input seeking.
"""

from pathlib import Path
from unittest import mock

from sanji.video import split_video


def _run_split(tmp_path: Path) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return mock.Mock(returncode=0)

    with mock.patch("sanji.video.subprocess.run", side_effect=fake_run):
        split_video(
            video_path=tmp_path / "source.mp4",
            split_points=[100.0, 250.0],
            total_duration=400.0,
            output_dir=tmp_path / "out",
            video_title="show",
        )
    return commands


def test_split_uses_input_side_seek(tmp_path):
    commands = _run_split(tmp_path)
    assert len(commands) == 3  # 2 split points -> 3 segments
    for cmd in commands:
        assert cmd.index("-ss") < cmd.index("-i"), "seek must precede the input"
        assert "-to" not in cmd


def test_split_segment_ranges(tmp_path):
    commands = _run_split(tmp_path)
    starts = [cmd[cmd.index("-ss") + 1] for cmd in commands]
    durations = [cmd[cmd.index("-t") + 1] for cmd in commands]
    assert starts == ["0.0", "100.0", "250.0"]
    assert durations == ["100.0", "150.0", "150.0"]


def test_split_reencodes_for_frame_accuracy(tmp_path):
    """Input-side seek is only frame-accurate with re-encoding — keep libx264."""
    commands = _run_split(tmp_path)
    for cmd in commands:
        codec = cmd[cmd.index("-c:v") + 1]
        assert codec == "libx264"
