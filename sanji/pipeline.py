"""Reusable video-splitting pipeline, callable from the CLI or a service worker.

Extracted from ``sanji.cli`` so the same pipeline can run behind an async job
worker (see issue #8). Behavior is identical to the previous ``cli.main()``
implementation; ``cli.main()`` is now a thin argparse adapter over
``run_pipeline``.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from sanji.audio import classify_audio, extract_audio
from sanji.functions import (
    compute_music_density,
    find_song_regions,
    find_split_points,
    merge_short_regions,
    parse_description_timestamps,
    select_best_splits,
    validate_against_timestamps,
    write_manifest,
)
from sanji.settings import (
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_SEGMENT,
    DEFAULT_MIN_SONG,
    DEFAULT_THRESHOLD,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WINDOW_SIZE,
)
from sanji.transcription import refine_splits_with_transcription
from sanji.utils import format_time
from sanji.video import download_video, get_video_duration, split_video

logger = structlog.get_logger().bind(logger=__name__)


class DurationExceededError(ValueError):
    """Raised when a video duration exceeds the plan's max-duration cap (enforcement B)."""


@dataclass(frozen=True)
class PipelineParams:
    """Tunable inputs for a single splitting run (CLI flags or job params)."""

    input: str
    output_dir: Path
    validate: bool = False
    # min_gap is retained for CLI parity; not currently consumed by the pipeline.
    min_gap: float = DEFAULT_MIN_GAP
    min_segment: float = DEFAULT_MIN_SEGMENT
    window: float = DEFAULT_WINDOW_SIZE
    threshold: float = DEFAULT_THRESHOLD
    min_song: float = DEFAULT_MIN_SONG
    expect_songs: int | None = None
    no_transcribe: bool = False
    whisper_model: str = DEFAULT_WHISPER_MODEL
    dry_run: bool = False
    max_duration_seconds: int | None = None  # None = unlimited (CLI / legacy paths)


@dataclass
class PipelineResult:
    """Outcome of a splitting run."""

    duration: float
    song_regions: list[tuple[float, float]]
    split_points: list[float]
    track_names: list[str] | None = None
    segment_files: list[Path] = field(default_factory=list)
    manifest_path: Path | None = None


def _load_video(
    input_str: str, work_dir: Path | None
) -> tuple[Path, str | None, Path | None]:
    """Download or locate the video file; return (video_path, description, work_dir)."""
    if input_str.startswith(("http://", "https://")):
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="splitter_"))
        video_path, description = download_video(input_str, work_dir)
    else:
        video_path = Path(input_str)
        if not video_path.exists():
            raise FileNotFoundError(f"file not found: {video_path}")
        desc_path = video_path.with_suffix(".description")
        description = desc_path.read_text() if desc_path.exists() else None
    return video_path, description, work_dir


def _detect_song_regions(
    segments: list, total_duration: float, params: PipelineParams
) -> list[tuple[float, float]]:
    """Steps 5-6b: density → raw regions → merged regions."""
    times, densities = compute_music_density(
        segments, total_duration, window_size=params.window
    )
    song_regions = find_song_regions(
        times, densities, threshold=params.threshold, min_song_duration=params.min_segment
    )
    logger.info(
        "song_regions_detected",
        count=len(song_regions),
        regions=[(format_time(s), format_time(e)) for s, e in song_regions],
    )

    pre_merge = len(song_regions)
    song_regions = merge_short_regions(song_regions, min_duration=params.min_song)
    if len(song_regions) != pre_merge:
        logger.info("song_regions_merged", before=pre_merge, after=len(song_regions))
    logger.info(
        "song_regions_final",
        count=len(song_regions),
        regions=[(format_time(s), format_time(e)) for s, e in song_regions],
    )
    return song_regions


def _compute_split_points(
    song_regions: list[tuple[float, float]],
    total_duration: float,
    audio_path: Path,
    params: PipelineParams,
    track_names: list[str] | None,
) -> list[float]:
    """Steps 7-7c: raw splits → filtered → transcription-refined."""
    split_points = find_split_points(song_regions, total_duration)

    if params.expect_songs and len(split_points) >= params.expect_songs:
        split_points = select_best_splits(split_points, total_duration, params.expect_songs)
        logger.info(
            "split_points_trimmed",
            expect_songs=params.expect_songs,
            kept=len(split_points),
        )

    logger.info(
        "split_points_initial",
        count=len(split_points),
        times=[format_time(sp) for sp in split_points],
    )

    if not params.no_transcribe:
        split_points = refine_splits_with_transcription(
            split_points,
            song_regions,
            audio_path,
            whisper_model=params.whisper_model,
            track_names=track_names,
        )
        logger.info(
            "split_points_refined",
            count=len(split_points),
            times=[format_time(sp) for sp in split_points],
        )

    return split_points


def _write_segments(
    video_path: Path,
    split_points: list[float],
    total_duration: float,
    output_dir: Path,
    track_names: list[str] | None,
) -> tuple[list[Path], Path]:
    """Step 10: split video, write manifest, log results."""
    video_title = video_path.stem
    if video_title == "source":
        video_title = "reaction"

    files = split_video(video_path, split_points, total_duration, output_dir, video_title, track_names)
    boundaries = [0.0] + split_points + [total_duration]
    manifest = write_manifest(output_dir, video_title, files, boundaries, track_names)

    logger.info(
        "segments_written",
        count=len(files),
        output_dir=str(output_dir),
        files=[{"name": f.name, "size_mb": round(f.stat().st_size / 1024 / 1024, 1)} for f in files],
        manifest=manifest.name,
    )
    return files, manifest


def run_pipeline(
    params: PipelineParams, *, work_dir: Path | None = None
) -> PipelineResult:
    """Run the full split pipeline and return the result.

    ``work_dir`` is an optional scratch directory for downloads and intermediate
    audio. When omitted and the input is a URL, a temp dir is created (matching
    the previous CLI behavior); a service worker injects its own job-scoped dir.

    Raises ``FileNotFoundError`` if a local input file does not exist.
    """
    output_dir = params.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path, description, work_dir = _load_video(params.input, work_dir)

    total_duration = get_video_duration(video_path)
    logger.info("video_duration", duration=format_time(total_duration))

    if params.max_duration_seconds is not None and total_duration > params.max_duration_seconds:
        raise DurationExceededError(
            f"Video duration {total_duration:.0f}s exceeds plan cap of {params.max_duration_seconds}s"
        )

    # The local-file path leaves work_dir untouched (only the URL path creates
    # it), so ensure it exists before ffmpeg writes audio.wav into it (#73).
    audio_dir = work_dir or output_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "audio.wav"
    extract_audio(video_path, audio_path)
    segments = classify_audio(audio_path)

    song_regions = _detect_song_regions(segments, total_duration, params)

    timestamps = (
        parse_description_timestamps(description, total_duration) if description else []
    )
    track_names = [name for _, name in timestamps] if timestamps else None

    split_points = _compute_split_points(
        song_regions, total_duration, audio_path, params, track_names
    )

    if params.validate and timestamps:
        validate_against_timestamps(split_points, timestamps)
    elif params.validate:
        logger.info("validation_skipped_no_timestamps")

    result = PipelineResult(
        duration=total_duration,
        song_regions=song_regions,
        split_points=split_points,
        track_names=track_names,
    )

    if params.dry_run:
        logger.info("dry_run_skipping_split")
    else:
        files, manifest = _write_segments(
            video_path, split_points, total_duration, output_dir, track_names
        )
        result.segment_files = files
        result.manifest_path = manifest

    if audio_path.exists():
        audio_path.unlink()

    return result
