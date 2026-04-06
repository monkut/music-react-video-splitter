"""Audio extraction and classification."""

import subprocess
from pathlib import Path


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
    from inaSpeechSegmenter import Segmenter

    print("Classifying audio (this may take a minute)...")
    seg = Segmenter()
    result = seg(str(audio_path))
    print(f"Classification complete: {len(result)} segments")
    return result


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
