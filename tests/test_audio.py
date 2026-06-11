"""Tests for the is-music heuristic classifier (issue #17)."""

import numpy as np

from sanji.audio import (
    FRAME_SAMPLES,
    FRAMES_PER_SECOND,
    SAMPLING_RATE,
    _median_smooth,
    _music_seconds,
)


def _seconds_of(signal_builder, n_seconds: int) -> np.ndarray:
    return np.concatenate([signal_builder(i) for i in range(n_seconds)]).astype(
        np.float32
    )


def _continuous_tone(_: int) -> np.ndarray:
    """Compressed, continuous audio — music-like (few low-energy frames)."""
    t = np.arange(SAMPLING_RATE) / SAMPLING_RATE
    return 0.3 * np.sin(2 * np.pi * 220 * t)


def _speech_like(_: int) -> np.ndarray:
    """Syllabic bursts: loud 100 ms bursts separated by near-silence."""
    second = np.zeros(SAMPLING_RATE)
    t = np.arange(2 * FRAME_SAMPLES) / SAMPLING_RATE
    burst = 0.3 * np.sin(2 * np.pi * 180 * t)
    for start_frame in range(
        0, FRAMES_PER_SECOND, 5
    ):  # 4 bursts/second ~ syllable rate
        start = start_frame * FRAME_SAMPLES
        second[start : start + len(burst)] = burst
    return second


def _silence(_: int) -> np.ndarray:
    return np.zeros(SAMPLING_RATE)


def test_continuous_tone_is_music():
    audio = _seconds_of(_continuous_tone, 10)
    assert _music_seconds(audio, 10).all()


def test_speech_like_bursts_are_not_music():
    audio = _seconds_of(_speech_like, 10)
    assert not _music_seconds(audio, 10).any()


def test_silence_is_not_music():
    audio = _seconds_of(_silence, 10)
    assert not _music_seconds(audio, 10).any()


def test_median_smooth_removes_single_second_flicker():
    values = np.array([1, 1, 1, 0, 1, 1, 1], dtype=np.float32)
    smoothed = _median_smooth(values, 5)
    assert (smoothed > 0.5).all()


def test_commentary_over_loud_music_reads_music():
    """Speech bursts ADDED ON TOP of continuous music must still read music-dominant."""
    music = _seconds_of(_continuous_tone, 10)
    speech = _seconds_of(_speech_like, 10)
    assert _music_seconds(music + speech, 10).all()
