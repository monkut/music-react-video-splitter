"""Reusable video-splitting pipeline, callable from the CLI or a service worker.

Extracted from ``sanji.cli`` so the same pipeline can run behind an async job
worker (see issue #8). Behavior is identical to the previous ``cli.main()``
implementation; ``cli.main()`` is now a thin argparse adapter over
``run_pipeline``.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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

    # Step 1: Get video
    is_url = params.input.startswith(("http://", "https://"))
    if is_url:
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="splitter_"))
        video_path, description = download_video(params.input, work_dir)
    else:
        video_path = Path(params.input)
        if not video_path.exists():
            raise FileNotFoundError(f"file not found: {video_path}")
        desc_path = video_path.with_suffix(".description")
        description = desc_path.read_text() if desc_path.exists() else None

    # Step 2: Get duration
    total_duration = get_video_duration(video_path)
    print(f"Video duration: {format_time(total_duration)}")

    # Enforcement B: reject before any expensive processing if duration exceeds the cap.
    if params.max_duration_seconds is not None and total_duration > params.max_duration_seconds:
        raise DurationExceededError(
            f"Video duration {total_duration:.0f}s exceeds plan cap of {params.max_duration_seconds}s"
        )

    # Step 3: Extract audio
    audio_path = (work_dir or output_dir) / "audio.wav"
    extract_audio(video_path, audio_path)

    # Step 4: Classify audio
    segments = classify_audio(audio_path)

    # Step 5: Compute music density
    times, densities = compute_music_density(
        segments,
        total_duration,
        window_size=params.window,
    )

    # Step 6: Find song regions
    song_regions = find_song_regions(
        times,
        densities,
        threshold=params.threshold,
        min_song_duration=params.min_segment,
    )

    print(f"\nDetected {len(song_regions)} raw song regions:")
    for i, (start, end) in enumerate(song_regions):
        print(
            f"  Region {i + 1}: {format_time(start)} - {format_time(end)} ({format_time(end - start)})"
        )

    # Step 6b: Merge short regions (handles mid-song pauses)
    pre_merge_count = len(song_regions)
    song_regions = merge_short_regions(song_regions, min_duration=params.min_song)

    if len(song_regions) != pre_merge_count:
        print(f"\nAfter merging short regions: {len(song_regions)} song regions")
    print(f"\nFinal {len(song_regions)} song regions:")
    for i, (start, end) in enumerate(song_regions):
        print(
            f"  Song {i + 1}: {format_time(start)} - {format_time(end)} ({format_time(end - start)})"
        )

    # Step 7: Find split points
    split_points = find_split_points(song_regions, total_duration)

    # Step 7b: If expect_songs is set, keep only the top N-1 splits
    if params.expect_songs and len(split_points) >= params.expect_songs:
        split_points = select_best_splits(
            split_points, total_duration, params.expect_songs
        )
        print(
            f"\n--expect-songs {params.expect_songs}: keeping {len(split_points)} splits (most balanced segments)"
        )

    print(f"\nInitial split points ({len(split_points)}):")
    for i, sp in enumerate(split_points):
        print(f"  Split {i + 1}: {format_time(sp)}")

    # Step 8: Parse timestamps from description (used for transcription, validation, naming)
    timestamps = (
        parse_description_timestamps(description, total_duration) if description else []
    )
    track_names = [name for _, name in timestamps] if timestamps else None

    # Step 7c: Refine splits using transcription
    if not params.no_transcribe:
        split_points = refine_splits_with_transcription(
            split_points,
            song_regions,
            audio_path,
            whisper_model=params.whisper_model,
            track_names=track_names,
        )
        print(f"\nRefined split points ({len(split_points)}):")
        for i, sp in enumerate(split_points):
            print(f"  Split {i + 1}: {format_time(sp)}")

    # Step 9: Validate if requested
    if params.validate and timestamps:
        validate_against_timestamps(split_points, timestamps)
    elif params.validate:
        print("\nNo timestamps found in description for validation")

    result = PipelineResult(
        duration=total_duration,
        song_regions=song_regions,
        split_points=split_points,
        track_names=track_names,
    )

    # Step 10: Split video
    if params.dry_run:
        print("\n[Dry run] Skipping video splitting")
    else:
        print(f"\nSplitting video into {len(split_points) + 1} segments...")
        video_title = video_path.stem
        if video_title == "source":
            video_title = "reaction"
        files = split_video(
            video_path,
            split_points,
            total_duration,
            output_dir,
            video_title,
            track_names,
        )

        boundaries = [0.0] + split_points + [total_duration]
        manifest = write_manifest(
            output_dir,
            video_title,
            files,
            boundaries,
            track_names,
        )
        result.segment_files = files
        result.manifest_path = manifest

        print(f"\nDone! {len(files)} segments written to {output_dir}/")
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.1f} MB)")
        print(f"  {manifest.name} (manifest)")

    # Cleanup temp audio
    if audio_path.exists():
        audio_path.unlink()

    return result
