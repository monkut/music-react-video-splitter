"""CLI entry point for sanji."""

import argparse
from pathlib import Path

from sanji.pipeline import PipelineParams, run_pipeline
from sanji.settings import (
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_SEGMENT,
    DEFAULT_MIN_SONG,
    DEFAULT_THRESHOLD,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WINDOW_SIZE,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Split music reaction videos into individual segments"
    )
    parser.add_argument("input", help="YouTube URL or local video file path")
    parser.add_argument(
        "-o",
        "--output",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate against description timestamps",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=DEFAULT_MIN_GAP,
        help=f"Minimum gap duration in seconds (default: {DEFAULT_MIN_GAP})",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=DEFAULT_MIN_SEGMENT,
        help=f"Minimum song segment duration in seconds (default: {DEFAULT_MIN_SEGMENT})",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Sliding window size for music density in seconds (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Music density threshold 0-1 (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--min-song",
        type=float,
        default=DEFAULT_MIN_SONG,
        help=f"Minimum song duration in seconds; shorter regions merge with neighbors (default: {DEFAULT_MIN_SONG})",
    )
    parser.add_argument(
        "--expect-songs",
        type=int,
        default=None,
        help="Expected number of songs; keeps only the N-1 most significant splits",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip transcription-based split refinement",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model for transcription (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Analyze only, don't split"
    )
    return parser


def main():
    args = build_parser().parse_args()

    params = PipelineParams(
        input=args.input,
        output_dir=Path(args.output),
        validate=args.validate,
        min_gap=args.min_gap,
        min_segment=args.min_segment,
        window=args.window,
        threshold=args.threshold,
        min_song=args.min_song,
        expect_songs=args.expect_songs,
        no_transcribe=args.no_transcribe,
        whisper_model=args.whisper_model,
        dry_run=args.dry_run,
    )

    try:
        run_pipeline(params)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
