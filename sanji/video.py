"""Video download, duration probe, and splitting."""

import json
import re
import subprocess
import sys
from pathlib import Path

from sanji.settings import AUDIO_BITRATE, VIDEO_CRF, VIDEO_PRESET
from sanji.utils import format_time


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

    for f in output_dir.iterdir():
        if f.stem == "source" and f.suffix not in (".description", ".json"):
            video_path = f
            break
    else:
        raise FileNotFoundError("Downloaded video not found")

    desc_path = output_dir / "source.description"
    description = desc_path.read_text() if desc_path.exists() else None

    print(f"Downloaded: {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return video_path, description


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

    boundaries = [0.0] + split_points + [total_duration]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

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
            "-c:v", "libx264", "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET,
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            str(output_path),
        ]

        start_str = format_time(start)
        end_str = format_time(end)
        print(f"  Splitting segment {i+1}: {start_str} - {end_str} -> {filename}")
        subprocess.run(cmd, check=True, capture_output=True)
        output_files.append(output_path)

    return output_files
