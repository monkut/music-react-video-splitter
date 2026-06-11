"""Audio extraction and classification.

Classification uses the is-music heuristic from issue #17 (benchmarked at parity
with the former inaSpeechSegmenter classifier on the Phase 0 corpus, 32/57
boundaries within ±30 s): per-second low-energy frame ratio (Scheirer & Slaney
1997) decides music dominance — commentary over a loud song still reads as
music; quiet BGM under chat reads as speech — combined with Silero VAD (bundled
with faster-whisper) for speech delineation. Pure numpy + ONNX; no TensorFlow.
"""

import subprocess
from pathlib import Path

import numpy as np

from sanji.settings import (
    LOW_ENERGY_FRAME_THRESHOLD,
    MUSIC_RMS_FLOOR,
    MUSIC_SMOOTH_SECONDS,
)

SAMPLING_RATE = 16000
FRAME_SAMPLES = 800  # 50 ms
FRAMES_PER_SECOND = SAMPLING_RATE // FRAME_SAMPLES


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract mono 16kHz WAV audio from video."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    print("Extracting audio...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Audio extracted: {output_path.name}")
    return output_path


def classify_audio(audio_path: Path) -> list[tuple[str, float, float]]:
    """Classify audio per second; return [(label, start, end), ...].

    Labels: "music" (music-dominant), "speech" (VAD speech, not music-dominant),
    "noise" (neither). Downstream density analysis only consumes MUSIC_LABELS.
    """
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    print("Classifying audio (low-energy heuristic + Silero VAD)...")
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLING_RATE)
    assert isinstance(
        audio, np.ndarray
    )  # split_stereo=False (default) returns a single array
    n_seconds = len(audio) // SAMPLING_RATE
    if n_seconds == 0:
        return []

    is_music = _music_seconds(audio, n_seconds)
    is_speech = np.zeros(n_seconds, dtype=bool)
    for chunk in get_speech_timestamps(audio, VadOptions()):
        start = int(chunk["start"] / SAMPLING_RATE)
        end = min(int(chunk["end"] / SAMPLING_RATE) + 1, n_seconds)
        is_speech[start:end] = True

    segments = []
    for sec in range(n_seconds):
        if is_music[sec]:
            label = "music"
        elif is_speech[sec]:
            label = "speech"
        else:
            label = "noise"
        segments.append((label, float(sec), float(sec + 1)))
    music_seconds = int(is_music.sum())
    speech_seconds = int((~is_music & is_speech).sum())
    print(
        f"Classification complete: {n_seconds}s total, music={music_seconds}s speech={speech_seconds}s"
    )
    return segments


def _music_seconds(audio: np.ndarray, n_seconds: int) -> np.ndarray:
    """Per-second music-dominance from the low-energy frame ratio + RMS floor.

    Speech is syllabically gappy (many frames below half the window mean RMS);
    mastered music is compressed/continuous (few quiet frames).
    """
    usable = audio[: n_seconds * SAMPLING_RATE].reshape(
        n_seconds, FRAMES_PER_SECOND, FRAME_SAMPLES
    )
    frame_rms = np.sqrt(np.mean(np.square(usable, dtype=np.float64), axis=2))
    mean_rms = frame_rms.mean(axis=1)
    low_energy_ratio = np.mean(frame_rms < 0.5 * mean_rms[:, None], axis=1)
    is_music = (low_energy_ratio < LOW_ENERGY_FRAME_THRESHOLD) & (
        mean_rms > MUSIC_RMS_FLOOR
    )

    if MUSIC_SMOOTH_SECONDS > 1:
        is_music = (
            _median_smooth(is_music.astype(np.float32), MUSIC_SMOOTH_SECONDS) > 0.5
        )
    return is_music


def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))])


def extract_gap_audio(
    audio_path: Path,
    gap_start: float,
    gap_end: float,
    output_path: Path,
) -> Path:
    """Extract a portion of audio for transcription."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        str(gap_start),
        "-to",
        str(gap_end),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
