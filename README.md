# music-react-video-splitter

CLI tool that splits music reaction videos into individual reaction segments using audio analysis and speech transcription.

Built for YouTube live request streams where a reactor watches multiple songs per stream. The tool detects where each song starts and ends, then splits the video into one clip per reaction (intro talk + song + discussion).

## How it works

1. **Download** video from YouTube via yt-dlp
2. **Classify** audio frames as speech/music/noise using inaSpeechSegmenter (CNN-based)
3. **Detect** song regions via sliding window music density analysis
4. **Transcribe** gap audio with faster-whisper (word-level timestamps) to find intro phrases and artist/song names
5. **Split** video at detected boundaries with ffmpeg

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- ffmpeg (system install)

## Setup

```bash
git clone https://github.com/monkut/music-react-video-splitter.git
cd music-react-video-splitter
uv sync
```

## Usage

### Split a YouTube video

```bash
uv run python splitter.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Split a local video file

```bash
uv run python splitter.py ./my-reaction-stream.mp4
```

### Common options

```bash
# Custom output directory
uv run python splitter.py VIDEO_URL -o ./clips/

# Analyze without splitting (see detected boundaries)
uv run python splitter.py VIDEO_URL --dry-run

# Validate detection against description timestamps
uv run python splitter.py VIDEO_URL --validate

# Skip transcription refinement (faster, less precise)
uv run python splitter.py VIDEO_URL --no-transcribe

# Use a larger Whisper model for better transcription accuracy
uv run python splitter.py VIDEO_URL --whisper-model small
```

### Tuning parameters

```bash
# Adjust music detection sensitivity
uv run python splitter.py VIDEO_URL --window 60 --threshold 0.20

# Set minimum song region duration (seconds)
uv run python splitter.py VIDEO_URL --min-segment 30

# Merge short regions with neighbors (handles mid-song pauses)
uv run python splitter.py VIDEO_URL --min-song 180
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `./output` | Output directory |
| `--validate` | off | Compare detected splits against description timestamps |
| `--dry-run` | off | Analyze only, don't split |
| `--no-transcribe` | off | Skip transcription-based split refinement |
| `--whisper-model` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `--window` | `60` | Sliding window size for music density (seconds) |
| `--threshold` | `0.20` | Music density threshold (0-1) |
| `--min-segment` | `30` | Minimum song region duration (seconds) |
| `--min-song` | `60` | Minimum song duration before merging with neighbor (seconds) |
| `--min-gap` | `5` | Minimum gap duration (seconds) |
| `--expect-songs` | auto | Expected number of songs (selects most balanced splits) |

## Output

Segments are saved as MP4 files named from the YouTube description timestamps when available:

```
output/
  reaction_01_welcome_back.mp4
  reaction_02_Elaphant_Gym_-_Dear_Humans.mp4
  reaction_03_YouTie_-_tsukinami.mp4
  ...
```

If no description timestamps exist, segments are numbered sequentially.

## Detection pipeline

```
YouTube URL
  -> yt-dlp download
  -> ffmpeg extract mono 16kHz WAV
  -> inaSpeechSegmenter classify (speech/music/noise per frame)
  -> sliding window music density -> song regions
  -> merge short regions (mid-song pause handling)
  -> 80% gap bias split placement
  -> faster-whisper transcribe gaps (word-level timestamps)
  -> pattern match: artist names, "next up", "let's check out", etc.
  -> ffmpeg split video at refined boundaries
```

## Accuracy

Tested on a 39-minute live request stream with 5 songs:

| Boundary | Expected | Detected | Offset |
|----------|----------|----------|--------|
| You&Tie - tsukinami | 11:38 | 11:32 | -6s |
| WHIZZ and GDJYB - BATTLE | 17:20 | 17:08 | -12s |
| X Japan - Orgasm live 1989 | 23:24 | 23:10 | -14s |
| Dash Rip Rock | 34:20 | 33:59 | -21s |

Mean offset: ~13 seconds on real boundaries.

## Known limitations

- Reactors who pause 2+ minutes mid-song create false split points (indistinguishable from between-song gaps via audio alone)
- Transcription refinement requires a local GPU for reasonable speed (CPU works but slower)
- First run downloads the Whisper model (~150MB for `base`)
- TensorFlow dependency from inaSpeechSegmenter is large (~2GB)
