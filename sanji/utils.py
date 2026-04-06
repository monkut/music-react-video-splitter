"""Shared utility functions."""

import re


def format_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_artist_song(track_name: str) -> tuple[str, str]:
    """Parse 'Artist - Song' from a track name string.

    Handles formats like:
      'Elephant Gym - Dear Humans bass playthrough'
      'X Japan - Orgasm live 1989'
      'welcome back' (no separator -> artist='', song=track_name)
    """
    for sep in (" - ", " \u2013 ", " \u2014 "):
        if sep in track_name:
            artist, song = track_name.split(sep, 1)
            return artist.strip(), song.strip()
    return "", track_name.strip()
