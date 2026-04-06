"""Core analysis functions: density, regions, splits, timestamps, validation, manifest."""

import csv
import re
from pathlib import Path

import numpy as np

from sanji.settings import (
    DEFAULT_DENSITY_STEP,
    MUSIC_LABELS,
    SPLIT_GAP_BIAS,
    VALIDATION_TOLERANCE,
)
from sanji.utils import format_time, parse_artist_song


def compute_music_density(
    segments: list[tuple[str, float, float]],
    total_duration: float,
    window_size: float = 120.0,
    step: float = DEFAULT_DENSITY_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute music density over time using a sliding window.

    Returns (times, densities) where density is 0-1 ratio of music in window.
    """
    n_frames = int(total_duration) + 1
    is_music = np.zeros(n_frames, dtype=np.float32)

    for label, start, end in segments:
        if label in MUSIC_LABELS:
            s = max(0, int(start))
            e = min(n_frames, int(end) + 1)
            is_music[s:e] = 1.0

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

    Places the split biased towards the start of the next song's intro talk.
    """
    if len(song_regions) <= 1:
        return []

    split_points = []
    for i in range(len(song_regions) - 1):
        gap_start = song_regions[i][1]
        gap_end = song_regions[i + 1][0]
        gap_duration = gap_end - gap_start
        split_points.append(gap_start + gap_duration * SPLIT_GAP_BIAS)

    return split_points


def parse_description_timestamps(
    description: str, total_duration: float | None = None,
) -> list[tuple[float, str]]:
    """Parse timestamps from video description. Returns [(seconds, title), ...]."""
    if not description:
        return []

    timestamps = []
    for line in description.strip().split("\n"):
        match = re.match(r"(\d+):(\d+)(?::(\d+))?\s*[-\u2013\u2014]\s*(.+)", line.strip())
        if match:
            groups = match.groups()
            if groups[2] is not None:
                seconds = int(groups[0]) * 3600 + int(groups[1]) * 60 + int(groups[2])
                if total_duration and seconds > total_duration:
                    seconds = int(groups[0]) * 60 + int(groups[1])
            else:
                seconds = int(groups[0]) * 60 + int(groups[1])
            title = groups[3].strip()
            timestamps.append((seconds, title))

    return timestamps


def validate_against_timestamps(
    split_points: list[float],
    expected_timestamps: list[tuple[float, str]],
) -> None:
    """Compare detected split points against expected timestamps."""
    song_starts = [ts for ts, _ in expected_timestamps[1:]]

    print("\n=== Validation Report ===")
    print(f"  Detected boundaries:  {len(split_points)}")
    print(f"  Expected boundaries:  {len(song_starts)}")

    if not split_points or not song_starts:
        print("  Cannot validate: missing data")
        return

    offsets = []
    matched = 0
    tolerance = VALIDATION_TOLERANCE

    for i, expected in enumerate(song_starts):
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


def select_best_splits(
    split_points: list[float],
    total_duration: float,
    n_songs: int,
) -> list[float]:
    """Keep only the N-1 most balanced splits from candidates."""
    from itertools import combinations

    n_keep = n_songs - 1
    best_combo = None
    best_variance = float("inf")

    for combo in combinations(range(len(split_points)), n_keep):
        candidate = [0.0] + [split_points[i] for i in combo] + [total_duration]
        segment_durations = [candidate[j + 1] - candidate[j] for j in range(len(candidate) - 1)]
        variance = np.var(segment_durations)
        if variance < best_variance:
            best_variance = variance
            best_combo = combo

    if best_combo is not None:
        return [split_points[i] for i in best_combo]
    return split_points


def write_manifest(
    output_dir: Path,
    prefix: str,
    split_files: list[Path],
    boundaries: list[float],
    track_names: list[str] | None,
) -> Path:
    """Write a manifest CSV with segment metadata.

    Columns: ORDER_INDEX, SPLIT_FILENAME, ORIGINAL_START_TIME, ORIGINAL_END_TIME, ARTIST, SONG
    """
    manifest_path = output_dir / f"{prefix}_MANIFEST.csv"

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ORDER_INDEX", "SPLIT_FILENAME",
            "ORIGINAL_START_TIME", "ORIGINAL_END_TIME",
            "ARTIST", "SONG",
        ])

        for i, split_file in enumerate(split_files):
            start = boundaries[i]
            end = boundaries[i + 1]

            if track_names and i < len(track_names):
                artist, song = parse_artist_song(track_names[i])
            else:
                artist, song = "", ""

            writer.writerow([
                i + 1,
                split_file.name,
                format_time(start),
                format_time(end),
                artist,
                song,
            ])

    return manifest_path
