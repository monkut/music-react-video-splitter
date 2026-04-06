"""Transcription-based split point refinement using faster-whisper."""

import re
from pathlib import Path

from sanji.audio import extract_gap_audio
from sanji.settings import DEFAULT_GAP_PADDING, INTRO_PATTERNS
from sanji.utils import format_time


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
    Uses faster-whisper for word-level timestamps.

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
        gap_start = song_regions[idx][1] if idx < len(song_regions) else sp - DEFAULT_GAP_PADDING
        gap_end = song_regions[idx + 1][0] if idx + 1 < len(song_regions) else sp + DEFAULT_GAP_PADDING
        gap_duration = gap_end - gap_start

        if gap_duration < 5:
            refined.append(sp)
            continue

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

            if track_names and idx + 1 < len(track_names):
                next_track = track_names[idx + 1] if idx + 1 < len(track_names) else None
                if not next_track and idx + 2 < len(track_names):
                    next_track = track_names[idx + 2]
                if next_track:
                    name_words = [
                        w for w in re.split(r'[\s\-\u2013]+', next_track.lower())
                        if len(w) >= 3 and w not in {"the", "and", "for", "from"}
                    ]
                    for nw in name_words:
                        patterns.append(r"\b" + re.escape(nw) + r"\b")

            patterns.extend(INTRO_PATTERNS)

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
                      f"{format_time(sp)} -> {format_time(new_sp)}")
                refined.append(new_sp)
            else:
                print(f"  Split {idx+1}: no intro found in gap "
                      f"{format_time(gap_start)}-{format_time(gap_end)}, "
                      f"keeping {format_time(sp)}")
                print(f'           transcript: "{full_text[:120]}..."' if len(full_text) > 120
                      else f'           transcript: "{full_text}"')
                refined.append(sp)

        finally:
            if gap_audio_path.exists():
                gap_audio_path.unlink()

    return refined
