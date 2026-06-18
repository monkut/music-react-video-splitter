"""Tests for sanji.functions — analysis, timestamps, and manifest generation."""

import csv

import numpy as np

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
from sanji.settings import SPLIT_GAP_BIAS


def _read_csv(path):
    return list(csv.reader(path.read_text().splitlines()))


class TestWriteManifest:
    def test_writes_header_and_metadata_rows(self, tmp_path):
        files = [tmp_path / "r_01_a.mp4", tmp_path / "r_02_b.mp4"]
        boundaries = [0.0, 120.0, 300.0]
        track_names = ["Elephant Gym - Dear Humans", "welcome back"]

        manifest = write_manifest(tmp_path, "r", files, boundaries, track_names)

        assert manifest == tmp_path / "r_MANIFEST.csv"
        rows = _read_csv(manifest)
        assert rows[0] == [
            "ORDER_INDEX",
            "SPLIT_FILENAME",
            "ORIGINAL_START_TIME",
            "ORIGINAL_END_TIME",
            "ARTIST",
            "SONG",
        ]
        assert rows[1] == [
            "1",
            "r_01_a.mp4",
            "0:00",
            "2:00",
            "Elephant Gym",
            "Dear Humans",
        ]
        assert rows[2] == ["2", "r_02_b.mp4", "2:00", "5:00", "", "welcome back"]

    def test_blank_artist_song_when_no_track_names(self, tmp_path):
        files = [tmp_path / "r_01.mp4"]
        manifest = write_manifest(tmp_path, "r", files, [0.0, 60.0], None)
        rows = _read_csv(manifest)
        assert rows[1] == ["1", "r_01.mp4", "0:00", "1:00", "", ""]

    def test_blank_metadata_for_files_beyond_track_names(self, tmp_path):
        files = [tmp_path / "r_01.mp4", tmp_path / "r_02.mp4"]
        boundaries = [0.0, 60.0, 120.0]
        manifest = write_manifest(tmp_path, "r", files, boundaries, ["A - B"])
        rows = _read_csv(manifest)
        assert rows[1][4:] == ["A", "B"]
        assert rows[2][4:] == ["", ""]


class TestComputeMusicDensity:
    def test_all_music_is_dense(self):
        times, densities = compute_music_density(
            [("music", 0, 100)], 100, window_size=20, step=5
        )
        assert len(times) == len(densities)
        assert densities.max() <= 1.0
        assert densities.max() > 0.9

    def test_no_music_is_zero_density(self):
        _, densities = compute_music_density(
            [("speech", 0, 100)], 100, window_size=20, step=5
        )
        assert densities.max() == 0.0


class TestFindSongRegions:
    def test_detects_single_region(self):
        times = np.arange(0, 200, 5)
        densities = np.where((times >= 50) & (times < 150), 0.8, 0.0).astype(np.float32)
        regions = find_song_regions(
            times, densities, threshold=0.15, min_song_duration=30
        )
        assert regions == [(50, 150)]

    def test_filters_regions_below_min_duration(self):
        times = np.arange(0, 100, 5)
        densities = np.where((times >= 50) & (times < 60), 0.8, 0.0).astype(np.float32)
        regions = find_song_regions(
            times, densities, threshold=0.15, min_song_duration=30
        )
        assert regions == []


class TestMergeShortRegions:
    def test_merges_short_region_into_nearest_neighbor(self):
        regions = [(0.0, 200.0), (205.0, 215.0)]
        assert merge_short_regions(regions, min_duration=180) == [(0.0, 215.0)]

    def test_single_region_unchanged(self):
        assert merge_short_regions([(0.0, 50.0)], min_duration=180) == [(0.0, 50.0)]

    def test_empty_unchanged(self):
        assert merge_short_regions([], min_duration=180) == []


class TestFindSplitPoints:
    def test_split_uses_gap_bias(self):
        regions = [(0.0, 100.0), (200.0, 300.0)]
        expected = 100 + (200 - 100) * SPLIT_GAP_BIAS
        assert find_split_points(regions, total_duration=300) == [expected]

    def test_single_region_has_no_splits(self):
        assert find_split_points([(0.0, 100.0)], 100) == []


class TestParseDescriptionTimestamps:
    def test_parses_mmss(self):
        desc = "0:00 - Intro\n3:45 - Song One\n10:30 - Song Two"
        assert parse_description_timestamps(desc) == [
            (0, "Intro"),
            (225, "Song One"),
            (630, "Song Two"),
        ]

    def test_parses_hhmmss(self):
        assert parse_description_timestamps("1:02:03 - Late Song") == [
            (3723, "Late Song")
        ]

    def test_empty_and_none_yield_empty_list(self):
        assert parse_description_timestamps("") == []
        assert parse_description_timestamps(None) == []

    def test_hhmmss_falls_back_to_mmss_when_beyond_duration(self):
        # 1:30:00 = 5400s exceeds a 600s video, so it is reinterpreted as 1:30 = 90s.
        assert parse_description_timestamps("1:30:00 - Weird", total_duration=600) == [
            (90, "Weird")
        ]


class TestSelectBestSplits:
    def test_keeps_most_balanced_splits(self):
        splits = [50.0, 100.0, 150.0, 200.0, 250.0]
        # Three equal 100s segments come from keeping 100 and 200.
        assert select_best_splits(splits, total_duration=300, n_songs=3) == [
            100.0,
            200.0,
        ]


class TestValidateAgainstTimestamps:
    def test_prints_report_with_accuracy(self, capsys):
        validate_against_timestamps([225.0], [(0, "Intro"), (230, "Song One")])
        out = capsys.readouterr().out
        assert "Validation Report" in out
        assert "Boundary accuracy" in out

    def test_reports_missing_data(self, capsys):
        validate_against_timestamps([], [(0, "Intro")])
        assert "Cannot validate" in capsys.readouterr().out
