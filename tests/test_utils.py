"""Tests for sanji.utils — pure formatting and parsing helpers."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sanji.utils import format_time, parse_artist_song

# Text with no separator that parse_artist_song could split on, stripped + non-empty.
no_separator_text = (
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", min_size=1)
    .map(str.strip)
    .filter(lambda s: len(s) > 0)
)


class TestFormatTime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0:00"),
            (5, "0:05"),
            (59, "0:59"),
            (60, "1:00"),
            (61, "1:01"),
            (125, "2:05"),
            (3599, "59:59"),
            (3600, "1:00:00"),
            (3661, "1:01:01"),
            (7325, "2:02:05"),
        ],
    )
    def test_known_values(self, seconds, expected):
        assert format_time(seconds) == expected

    def test_truncates_fractional_seconds(self):
        assert format_time(65.9) == "1:05"

    @given(
        st.floats(min_value=0, max_value=359999, allow_nan=False, allow_infinity=False)
    )
    def test_roundtrips_to_floored_seconds(self, seconds):
        parts = [int(p) for p in format_time(seconds).split(":")]
        if len(parts) == 2:
            total = parts[0] * 60 + parts[1]
        else:
            total = parts[0] * 3600 + parts[1] * 60 + parts[2]
        assert total == int(seconds)

    @given(
        st.floats(
            min_value=0, max_value=3599.999, allow_nan=False, allow_infinity=False
        )
    )
    def test_under_one_hour_has_no_hour_field(self, seconds):
        assert format_time(seconds).count(":") == 1

    @given(
        st.floats(
            min_value=3600, max_value=359999, allow_nan=False, allow_infinity=False
        )
    )
    def test_one_hour_or_more_zero_pads_minutes_and_seconds(self, seconds):
        _, minutes, secs = format_time(seconds).split(":")
        assert len(minutes) == 2
        assert len(secs) == 2


class TestParseArtistSong:
    @pytest.mark.parametrize(
        ("track", "artist", "song"),
        [
            (
                "Elephant Gym - Dear Humans bass playthrough",
                "Elephant Gym",
                "Dear Humans bass playthrough",
            ),
            ("X Japan - Orgasm live 1989", "X Japan", "Orgasm live 1989"),
            ("welcome back", "", "welcome back"),
            ("  spaced out  ", "", "spaced out"),
            ("A – B", "A", "B"),  # en dash separator
            ("A — B", "A", "B"),  # em dash separator
            (
                "Artist - Song - Extended",
                "Artist",
                "Song - Extended",
            ),  # split once only
        ],
    )
    def test_known_values(self, track, artist, song):
        assert parse_artist_song(track) == (artist, song)

    @given(no_separator_text, no_separator_text)
    def test_hyphen_separator_splits_into_artist_and_song(self, artist, song):
        assert parse_artist_song(f"{artist} - {song}") == (artist, song)

    @given(no_separator_text)
    def test_no_separator_yields_empty_artist(self, track):
        assert parse_artist_song(track) == ("", track)
