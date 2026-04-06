"""CLI entry point for sanji."""

import argparse
import tempfile
from pathlib import Path

import numpy as np

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


def main():
    parser = argparse.ArgumentParser(
        description="Split music reaction videos into individual segments"
    )
    parser.add_argument("input", help="YouTube URL or local video file path")
    parser.add_argument("-o", "--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--validate", action="store_true", help="Validate against description timestamps")
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP, help=f"Minimum gap duration in seconds (default: {DEFAULT_MIN_GAP})")
    parser.add_argument("--min-segment", type=float, default=DEFAULT_MIN_SEGMENT, help=f"Minimum song segment duration in seconds (default: {DEFAULT_MIN_SEGMENT})")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_SIZE, help=f"Sliding window size for music density in seconds (default: {DEFAULT_WINDOW_SIZE})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Music density threshold 0-1 (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--min-song", type=float, default=DEFAULT_MIN_SONG, help=f"Minimum song duration in seconds; shorter regions merge with neighbors (default: {DEFAULT_MIN_SONG})")
    parser.add_argument("--expect-songs", type=int, default=None, help="Expected number of songs; keeps only the N-1 most significant splits")
    parser.add_argument("--no-transcribe", action="store_true", help="Skip transcription-based split refinement")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help=f"Whisper model for transcription (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't split")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get video
    is_url = args.input.startswith(("http://", "https://"))
    if is_url:
        work_dir = Path(tempfile.mkdtemp(prefix="splitter_"))
        video_path, description = download_video(args.input, work_dir)
    else:
        video_path = Path(args.input)
        if not video_path.exists():
            print(f"Error: file not found: {video_path}")
            raise SystemExit(1)
        desc_path = video_path.with_suffix(".description")
        description = desc_path.read_text() if desc_path.exists() else None
        work_dir = None

    # Step 2: Get duration
    total_duration = get_video_duration(video_path)
    print(f"Video duration: {format_time(total_duration)}")

    # Step 3: Extract audio
    audio_path = (work_dir or output_dir) / "audio.wav"
    extract_audio(video_path, audio_path)

    # Step 4: Classify audio
    segments = classify_audio(audio_path)

    # Step 5: Compute music density
    times, densities = compute_music_density(
        segments, total_duration,
        window_size=args.window,
    )

    # Step 6: Find song regions
    song_regions = find_song_regions(
        times, densities,
        threshold=args.threshold,
        min_song_duration=args.min_segment,
    )

    print(f"\nDetected {len(song_regions)} raw song regions:")
    for i, (start, end) in enumerate(song_regions):
        print(f"  Region {i+1}: {format_time(start)} - {format_time(end)} ({format_time(end - start)})")

    # Step 6b: Merge short regions (handles mid-song pauses)
    pre_merge_count = len(song_regions)
    song_regions = merge_short_regions(song_regions, min_duration=args.min_song)

    if len(song_regions) != pre_merge_count:
        print(f"\nAfter merging short regions: {len(song_regions)} song regions")
    print(f"\nFinal {len(song_regions)} song regions:")
    for i, (start, end) in enumerate(song_regions):
        print(f"  Song {i+1}: {format_time(start)} - {format_time(end)} ({format_time(end - start)})")

    # Step 7: Find split points
    split_points = find_split_points(song_regions, total_duration)

    # Step 7b: If --expect-songs is set, keep only the top N-1 splits
    if args.expect_songs and len(split_points) >= args.expect_songs:
        split_points = select_best_splits(split_points, total_duration, args.expect_songs)
        print(f"\n--expect-songs {args.expect_songs}: keeping {len(split_points)} splits (most balanced segments)")

    print(f"\nInitial split points ({len(split_points)}):")
    for i, sp in enumerate(split_points):
        print(f"  Split {i+1}: {format_time(sp)}")

    # Step 8: Parse timestamps from description (used for transcription, validation, naming)
    timestamps = parse_description_timestamps(description, total_duration) if description else []
    track_names = [name for _, name in timestamps] if timestamps else None

    # Step 7c: Refine splits using transcription
    if not args.no_transcribe:
        split_points = refine_splits_with_transcription(
            split_points, song_regions, audio_path,
            whisper_model=args.whisper_model,
            track_names=track_names,
        )
        print(f"\nRefined split points ({len(split_points)}):")
        for i, sp in enumerate(split_points):
            print(f"  Split {i+1}: {format_time(sp)}")

    # Step 9: Validate if requested
    if args.validate and timestamps:
        validate_against_timestamps(split_points, timestamps)
    elif args.validate:
        print("\nNo timestamps found in description for validation")

    # Step 10: Split video
    if args.dry_run:
        print("\n[Dry run] Skipping video splitting")
    else:
        print(f"\nSplitting video into {len(split_points) + 1} segments...")
        video_title = video_path.stem
        if video_title == "source":
            video_title = "reaction"
        files = split_video(
            video_path, split_points, total_duration,
            output_dir, video_title, track_names,
        )

        boundaries = [0.0] + split_points + [total_duration]
        manifest = write_manifest(
            output_dir, video_title, files, boundaries, track_names,
        )

        print(f"\nDone! {len(files)} segments written to {output_dir}/")
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.1f} MB)")
        print(f"  {manifest.name} (manifest)")

    # Cleanup temp audio
    if audio_path.exists():
        audio_path.unlink()
