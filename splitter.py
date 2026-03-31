"""Split music reaction videos into individual segments using audio analysis."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from inaSpeechSegmenter import Segmenter


SPEECH_LABELS = {"male", "female", "speech"}
MUSIC_LABELS = {"music"}


def download_video(url: str, output_dir: Path) -> tuple[Path, str | None]:
    """Download video from YouTube, return (video_path, description)."""
    output_template = str(output_dir / "source.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-o", output_template,
        "--write-description",
        "--no-playlist",
        url,
    ]
    print(f"Downloading: {url}")
    subprocess.run(cmd, check=True)

    # Find the downloaded video file
    for f in output_dir.iterdir():
        if f.stem == "source" and f.suffix not in (".description", ".json"):
            video_path = f
            break
    else:
        raise FileNotFoundError("Downloaded video not found")

    # Read description if available
    desc_path = output_dir / "source.description"
    description = desc_path.read_text() if desc_path.exists() else None

    print(f"Downloaded: {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return video_path, description


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract mono 16kHz WAV audio from video."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-vn",
        str(output_path),
    ]
    print("Extracting audio...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Audio extracted: {output_path.name}")
    return output_path


def classify_audio(audio_path: Path) -> list[tuple[str, float, float]]:
    """Run inaSpeechSegmenter and return [(label, start, end), ...]."""
    print("Classifying audio (this may take a minute)...")
    seg = Segmenter()
    result = seg(str(audio_path))
    print(f"Classification complete: {len(result)} segments")
    return result


def compute_music_density(
    segments: list[tuple[str, float, float]],
    total_duration: float,
    window_size: float = 120.0,
    step: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute music density over time using a sliding window.

    Returns (times, densities) where density is 0-1 ratio of music in window.
    """
    # Build a frame-level label array at 1-second resolution
    n_frames = int(total_duration) + 1
    is_music = np.zeros(n_frames, dtype=np.float32)

    for label, start, end in segments:
        if label in MUSIC_LABELS:
            s = max(0, int(start))
            e = min(n_frames, int(end) + 1)
            is_music[s:e] = 1.0

    # Sliding window average
    half_win = int(window_size / 2)
    times = np.arange(0, total_duration, step)
    densities = np.zeros(len(times), dtype=np.float32)

    for i, t in enumerate(times):
        s = max(0, int(t) - half_win)
        e = min(n_frames, int(t) + half_win)
        if e > s:
            densities[i] = is_music[s:e].mean()

    return times, densities


def find_song_regions(
    times: np.ndarray,
    densities: np.ndarray,
    threshold: float = 0.15,
    min_song_duration: float = 30.0,
) -> list[tuple[float, float]]:
    """Find contiguous regions where music density exceeds threshold.

    Returns [(start_time, end_time), ...] for each song region.
    """
    in_song = densities > threshold
    regions = []
    start = None

    for i, t in enumerate(times):
        if in_song[i] and start is None:
            start = t
        elif not in_song[i] and start is not None:
            duration = t - start
            if duration >= min_song_duration:
                regions.append((start, t))
            start = None

    # Handle region extending to end
    if start is not None:
        duration = times[-1] - start
        if duration >= min_song_duration:
            regions.append((start, times[-1]))

    return regions


def merge_short_regions(
    regions: list[tuple[float, float]],
    min_duration: float = 180.0,
) -> list[tuple[float, float]]:
    """Merge song regions shorter than min_duration with nearest neighbor.

    Handles reactors who pause mid-song to comment, creating two short
    regions where there should be one long one.
    """
    if len(regions) <= 1:
        return regions

    merged = list(regions)
    changed = True
    while changed:
        changed = False
        for i, (start, end) in enumerate(merged):
            duration = end - start
            if duration >= min_duration:
                continue

            # Find the nearest neighbor by gap size
            best_neighbor = None
            best_gap = float("inf")

            if i > 0:
                gap = start - merged[i - 1][1]
                if gap < best_gap:
                    best_gap = gap
                    best_neighbor = i - 1
            if i < len(merged) - 1:
                gap = merged[i + 1][0] - end
                if gap < best_gap:
                    best_gap = gap
                    best_neighbor = i + 1

            if best_neighbor is not None:
                ni = best_neighbor
                new_start = min(merged[i][0], merged[ni][0])
                new_end = max(merged[i][1], merged[ni][1])
                # Replace both with merged region
                low, high = min(i, ni), max(i, ni)
                merged[low] = (new_start, new_end)
                del merged[high]
                changed = True
                break

    return merged


def find_split_points(
    song_regions: list[tuple[float, float]],
    total_duration: float,
) -> list[float]:
    """Find optimal split points between song regions.

    Places the split at 80% through each gap, biased towards the start
    of the next song's intro talk.
    """
    if len(song_regions) <= 1:
        return []

    split_points = []
    for i in range(len(song_regions) - 1):
        gap_start = song_regions[i][1]
        gap_end = song_regions[i + 1][0]
        gap_duration = gap_end - gap_start

        # Place split at 80% through the gap, biased towards the next song's
        # intro talk. This works because description timestamps mark where
        # the next song starts, and the intro talk is near the end of the gap.
        split_points.append(gap_start + gap_duration * 0.80)

    return split_points


def parse_description_timestamps(
    description: str, total_duration: float | None = None,
) -> list[tuple[float, str]]:
    """Parse timestamps from video description. Returns [(seconds, title), ...]."""
    if not description:
        return []

    timestamps = []
    for line in description.strip().split("\n"):
        # Match patterns like "2:36 - Title" or "1:23:45 - Title"
        match = re.match(r"(\d+):(\d+)(?::(\d+))?\s*[-–—]\s*(.+)", line.strip())
        if match:
            groups = match.groups()
            if groups[2] is not None:  # Could be H:M:S or M:S:extra (typo)
                seconds = int(groups[0]) * 3600 + int(groups[1]) * 60 + int(groups[2])
                # Sanity check: if > duration, try M:S interpretation (common typo)
                if total_duration and seconds > total_duration:
                    seconds = int(groups[0]) * 60 + int(groups[1])
            else:  # M:S format
                seconds = int(groups[0]) * 60 + int(groups[1])
            title = groups[3].strip()
            timestamps.append((seconds, title))

    return timestamps


def validate_against_timestamps(
    split_points: list[float],
    expected_timestamps: list[tuple[float, str]],
) -> None:
    """Compare detected split points against expected timestamps."""
    # Skip the first timestamp (usually intro/welcome)
    song_starts = [ts for ts, _ in expected_timestamps[1:]]

    print("\n=== Validation Report ===")
    print(f"  Detected boundaries:  {len(split_points)}")
    print(f"  Expected boundaries:  {len(song_starts)}")

    if not split_points or not song_starts:
        print("  Cannot validate: missing data")
        return

    offsets = []
    matched = 0
    tolerance = 30  # seconds

    for i, expected in enumerate(song_starts):
        # Find closest detected split point
        if split_points:
            closest = min(split_points, key=lambda x: abs(x - expected))
            offset = closest - expected
            offsets.append(abs(offset))
            status = "OK" if abs(offset) <= tolerance else "MISS"
            if abs(offset) <= tolerance:
                matched += 1

            exp_str = f"{int(expected)//60}:{int(expected)%60:02d}"
            det_str = f"{int(closest)//60}:{int(closest)%60:02d}"
            name = expected_timestamps[i + 1][1] if i + 1 < len(expected_timestamps) else "?"
            print(f"  #{i+1}  Expected: {exp_str}  Detected: {det_str}  Offset: {offset:+.0f}s  [{status}]  {name}")

    if offsets:
        print(f"\n  Mean offset error:    {np.mean(offsets):.1f}s")
        print(f"  Max offset error:     {max(offsets):.1f}s")
        print(f"  Boundary accuracy:    {matched}/{len(song_starts)} within +/- {tolerance}s")


def extract_gap_audio(
    audio_path: Path,
    gap_start: float,
    gap_end: float,
    output_path: Path,
) -> Path:
    """Extract a portion of audio for transcription."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ss", str(gap_start),
        "-to", str(gap_end),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


# Patterns that indicate the reactor is introducing the next song.
# Ordered by specificity — more specific patterns are checked first.
INTRO_PATTERNS = [
    # Direct song introduction
    r"\b(next up|next one|next request|next song)\b",
    r"\b(let'?s check out|let'?s listen|let'?s see|let'?s hear|let'?s go)\b",
    r"\b(this one is|this is from|this is by|this next one)\b",
    r"\b(someone requested|we got a request|request from)\b",
    # Transitional phrases
    r"\b(alright so|okay so|all right so)\b",
    r"\b(moving on|on to the next)\b",
]


def _find_pattern_timestamp(
    words: list[tuple[float, float, str]],
    full_text: str,
    pattern: str,
) -> tuple[float, str] | None:
    """Search for a regex pattern in text and return the word timestamp."""
    match = re.search(pattern, full_text, re.IGNORECASE)
    if not match:
        return None

    char_pos = match.start()
    current_pos = 0
    for abs_start, _, word_text in words:
        word_len = len(word_text) + 1  # +1 for space
        if current_pos + word_len > char_pos:
            return abs_start, match.group()
        current_pos += word_len

    return None


def refine_splits_with_transcription(
    split_points: list[float],
    song_regions: list[tuple[float, float]],
    audio_path: Path,
    whisper_model: str = "base",
    track_names: list[str] | None = None,
) -> list[float]:
    """Refine split points using speech transcription.

    For each gap between song regions, transcribe the speech and find
    the point where the reactor starts introducing the next song.
    Uses faster-whisper for word-level timestamps, following the same
    Whisper engine approach as c1p-comlink.

    Searches for:
    1. The next song's artist/title name (most precise)
    2. Common introduction phrases ("next up", "let's check out", etc.)
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  faster-whisper not available, skipping transcription refinement")
        return split_points

    print(f"\nRefining split points with transcription (model: {whisper_model})...")
    model = WhisperModel(whisper_model, device="auto", compute_type="auto")

    refined = []
    for idx, sp in enumerate(split_points):
        gap_start = song_regions[idx][1] if idx < len(song_regions) else sp - 60
        gap_end = song_regions[idx + 1][0] if idx + 1 < len(song_regions) else sp + 60
        gap_duration = gap_end - gap_start

        if gap_duration < 5:
            refined.append(sp)
            continue

        # Extract gap audio to temp file
        gap_audio_path = audio_path.parent / f"gap_{idx}.wav"
        try:
            extract_gap_audio(audio_path, gap_start, gap_end, gap_audio_path)

            segments, _ = model.transcribe(
                str(gap_audio_path),
                word_timestamps=True,
                language="en",
            )

            words = []
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        words.append((
                            gap_start + word.start,
                            gap_start + word.end,
                            word.word.strip().lower(),
                        ))

            if not words:
                refined.append(sp)
                continue

            full_text = " ".join(w[2] for w in words)

            # Build patterns: track name patterns first (most specific),
            # then generic intro phrases
            patterns = []

            # If we know the next song's name, search for artist/title keywords
            if track_names and idx + 1 < len(track_names):
                next_track = track_names[idx + 1] if idx + 1 < len(track_names) else None
                if not next_track and idx + 2 < len(track_names):
                    next_track = track_names[idx + 2]
                if next_track:
                    # Extract significant words from track name (skip short words)
                    name_words = [
                        w for w in re.split(r'[\s\-–]+', next_track.lower())
                        if len(w) >= 3 and w not in {"the", "and", "for", "from"}
                    ]
                    for nw in name_words:
                        patterns.append(r"\b" + re.escape(nw) + r"\b")

            patterns.extend(INTRO_PATTERNS)

            # Search patterns in order of specificity
            best_match_time = None
            best_pattern = None

            for pattern in patterns:
                result = _find_pattern_timestamp(words, full_text, pattern)
                if result:
                    best_match_time, best_pattern = result
                    break

            if best_match_time is not None:
                new_sp = max(gap_start + 2, best_match_time - 3)
                gap_s = format_time(gap_start)
                gap_e = format_time(gap_end)
                print(f"  Split {idx+1}: gap {gap_s}-{gap_e} | "
                      f'found "{best_pattern}" at {format_time(best_match_time)} | '
                      f"{format_time(sp)} → {format_time(new_sp)}")
                refined.append(new_sp)
            else:
                print(f"  Split {idx+1}: no intro found in gap "
                      f"{format_time(gap_start)}-{format_time(gap_end)}, "
                      f"keeping {format_time(sp)}")
                print(f"           transcript: \"{full_text[:120]}...\"" if len(full_text) > 120
                      else f"           transcript: \"{full_text}\"")
                refined.append(sp)

        finally:
            if gap_audio_path.exists():
                gap_audio_path.unlink()

    return refined


def split_video(
    video_path: Path,
    split_points: list[float],
    total_duration: float,
    output_dir: Path,
    video_title: str = "segment",
    track_names: list[str] | None = None,
) -> list[Path]:
    """Split video at the given timestamps using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    # Build segment boundaries
    boundaries = [0.0] + split_points + [total_duration]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Build output filename
        if track_names and i < len(track_names):
            safe_name = re.sub(r'[^\w\s-]', '', track_names[i]).strip()
            safe_name = re.sub(r'\s+', '_', safe_name)
            filename = f"{video_title}_{i+1:02d}_{safe_name}.mp4"
        else:
            filename = f"{video_title}_{i+1:02d}.mp4"

        output_path = output_dir / filename

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(start),
            "-to", str(end),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]

        start_str = f"{int(start)//60}:{int(start)%60:02d}"
        end_str = f"{int(end)//60}:{int(end)%60:02d}"
        print(f"  Splitting segment {i+1}: {start_str} - {end_str} → {filename}")
        subprocess.run(cmd, check=True, capture_output=True)
        output_files.append(output_path)

    return output_files


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def format_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Split music reaction videos into individual segments"
    )
    parser.add_argument("input", help="YouTube URL or local video file path")
    parser.add_argument("-o", "--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--validate", action="store_true", help="Validate against description timestamps")
    parser.add_argument("--min-gap", type=float, default=5.0, help="Minimum gap duration in seconds (default: 5)")
    parser.add_argument("--min-segment", type=float, default=30.0, help="Minimum song segment duration in seconds (default: 30)")
    parser.add_argument("--window", type=float, default=60.0, help="Sliding window size for music density in seconds (default: 60)")
    parser.add_argument("--threshold", type=float, default=0.20, help="Music density threshold 0-1 (default: 0.20)")
    parser.add_argument("--min-song", type=float, default=60.0, help="Minimum song duration in seconds; shorter regions merge with neighbors (default: 60)")
    parser.add_argument("--expect-songs", type=int, default=None, help="Expected number of songs; keeps only the N-1 most significant splits")
    parser.add_argument("--no-transcribe", action="store_true", help="Skip transcription-based split refinement")
    parser.add_argument("--whisper-model", default="base", help="Whisper model for transcription (default: base)")
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
            sys.exit(1)
        # Look for a .description file alongside the video
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
    song_regions = merge_short_regions(song_regions, min_duration=args.min_song)

    if len(song_regions) != len(song_regions):
        print(f"\nAfter merging short regions: {len(song_regions)} song regions")
    print(f"\nFinal {len(song_regions)} song regions:")
    for i, (start, end) in enumerate(song_regions):
        print(f"  Song {i+1}: {format_time(start)} - {format_time(end)} ({format_time(end - start)})")

    # Step 7: Find split points
    split_points = find_split_points(song_regions, total_duration)

    # Step 7b: If --expect-songs is set, keep only the top N-1 splits
    if args.expect_songs and len(split_points) >= args.expect_songs:
        n_keep = args.expect_songs - 1
        # Try all combinations of N-1 splits and pick the set that
        # produces the most evenly-sized segments (lowest variance).
        from itertools import combinations
        boundaries = [0.0] + split_points + [total_duration]
        best_combo = None
        best_variance = float("inf")

        for combo in combinations(range(len(split_points)), n_keep):
            candidate = [0.0] + [split_points[i] for i in combo] + [total_duration]
            segment_durations = [candidate[j+1] - candidate[j] for j in range(len(candidate) - 1)]
            variance = np.var(segment_durations)
            if variance < best_variance:
                best_variance = variance
                best_combo = combo

        if best_combo is not None:
            split_points = [split_points[i] for i in best_combo]
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
        print(f"\nDone! {len(files)} segments written to {output_dir}/")
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.1f} MB)")

    # Cleanup temp audio
    if audio_path.exists():
        audio_path.unlink()


if __name__ == "__main__":
    main()
