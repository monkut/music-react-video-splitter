"""Core analysis functions: density, regions, splits, timestamps, validation, manifest."""

import csv
import re
from pathlib import Path

import numpy as np

from sanji.settings import (
    DEFAULT_DENSITY_STEP,
    DEFAULT_WINDOW_SIZE,
    MUSIC_LABELS,
    SPLIT_GAP_BIAS,
    VALIDATION_TOLERANCE,
)
from sanji.utils import format_time, parse_artist_song


def compute_music_density(
    segments: list[tuple[str, float, float]],
    total_duration: float,
    window_size: float = DEFAULT_WINDOW_SIZE,
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
    description: str | None,
    total_duration: float | None = None,
) -> list[tuple[float, str]]:
    """Parse timestamps from video description. Returns [(seconds, title), ...]."""
    if not description:
        return []

    timestamps = []
    for line in description.strip().split("\n"):
        match = re.match(
            r"(\d+):(\d+)(?::(\d+))?\s*[-\u2013\u2014]\s*(.+)", line.strip()
        )
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

            exp_str = format_time(expected)
            det_str = format_time(closest)
            name = (
                expected_timestamps[i + 1][1]
                if i + 1 < len(expected_timestamps)
                else "?"
            )
            print(
                f"  #{i + 1}  Expected: {exp_str}  Detected: {det_str}  Offset: {offset:+.0f}s  [{status}]  {name}"
            )

    if offsets:
        print(f"\n  Mean offset error:    {np.mean(offsets):.1f}s")
        print(f"  Max offset error:     {max(offsets):.1f}s")
        print(
            f"  Boundary accuracy:    {matched}/{len(song_starts)} within +/- {tolerance}s"
        )


def select_best_splits(
    split_points: list[float],
    total_duration: float,
    n_songs: int,
) -> list[float]:
    """Keep only the N-1 most balanced splits from candidates.

    Minimizing the variance of the segment durations is equivalent to
    minimizing the sum of their squares: the mean duration is fixed at
    total_duration / n_songs no matter which splits are chosen. That makes
    this a standard O(n²·k) dynamic program over the sorted candidates —
    the previous brute force enumerated all C(n, k) combinations (>1.3M
    for 24 candidates and 12 songs) and ran np.var on each (#41).
    """
    n_keep = n_songs - 1
    n = len(split_points)
    if n_keep <= 0:
        return []
    if n_keep >= n:
        return split_points

    points = sorted(split_points)
    infinity = float("inf")

    # dp[m][j]: minimal sum of squared segment durations covering
    # [0, points[j]] using m chosen splits, the m-th being points[j].
    dp = [[infinity] * n for _ in range(n_keep + 1)]
    parent = [[-1] * n for _ in range(n_keep + 1)]
    for j in range(n):
        dp[1][j] = points[j] ** 2
    for m in range(2, n_keep + 1):
        for j in range(m - 1, n):
            best_cost = infinity
            best_prev = -1
            for i in range(m - 2, j):
                cost = dp[m - 1][i] + (points[j] - points[i]) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best_prev = i
            dp[m][j] = best_cost
            parent[m][j] = best_prev

    # close each candidate solution with the final segment up to total_duration
    best_last = min(
        range(n), key=lambda j: dp[n_keep][j] + (total_duration - points[j]) ** 2
    )
    chosen: list[float] = []
    j = best_last
    for m in range(n_keep, 0, -1):
        chosen.append(points[j])
        j = parent[m][j]
    chosen.reverse()
    return chosen


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
        writer.writerow(
            [
                "ORDER_INDEX",
                "SPLIT_FILENAME",
                "ORIGINAL_START_TIME",
                "ORIGINAL_END_TIME",
                "ARTIST",
                "SONG",
            ]
        )

        for i, split_file in enumerate(split_files):
            start = boundaries[i]
            end = boundaries[i + 1]

            if track_names and i < len(track_names):
                artist, song = parse_artist_song(track_names[i])
            else:
                artist, song = "", ""

            writer.writerow(
                [
                    i + 1,
                    split_file.name,
                    format_time(start),
                    format_time(end),
                    artist,
                    song,
                ]
            )

    return manifest_path
