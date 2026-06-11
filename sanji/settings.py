"""Constants and environment-configurable defaults."""

import os

SPEECH_LABELS = {"male", "female", "speech"}
MUSIC_LABELS = {"music"}

# Sliding window / density defaults
DEFAULT_WINDOW_SIZE = float(os.environ.get("SANJI_WINDOW_SIZE", "60.0"))
DEFAULT_DENSITY_STEP = float(os.environ.get("SANJI_DENSITY_STEP", "5.0"))
# 0.50 pairs with the per-second heuristic classifier (dense music labels);
# the old inaSpeechSegmenter default was 0.20. See issue #17 benchmark.
DEFAULT_THRESHOLD = float(os.environ.get("SANJI_THRESHOLD", "0.50"))

# Is-music heuristic (low-energy frame ratio — Scheirer & Slaney 1997; issue #17)
LOW_ENERGY_FRAME_THRESHOLD = float(
    os.environ.get("SANJI_LOW_ENERGY_FRAME_THRESHOLD", "0.35")
)
MUSIC_RMS_FLOOR = float(os.environ.get("SANJI_MUSIC_RMS_FLOOR", "0.01"))
MUSIC_SMOOTH_SECONDS = int(os.environ.get("SANJI_MUSIC_SMOOTH_SECONDS", "5"))

# Segment duration defaults
DEFAULT_MIN_SEGMENT = float(os.environ.get("SANJI_MIN_SEGMENT", "30.0"))
DEFAULT_MIN_SONG = float(os.environ.get("SANJI_MIN_SONG", "60.0"))
DEFAULT_MIN_GAP = float(os.environ.get("SANJI_MIN_GAP", "5.0"))

# Split point bias (fraction through gap towards next song intro)
SPLIT_GAP_BIAS = float(os.environ.get("SANJI_SPLIT_GAP_BIAS", "0.80"))

# Transcription
DEFAULT_WHISPER_MODEL = os.environ.get("SANJI_WHISPER_MODEL", "base")
DEFAULT_GAP_PADDING = float(os.environ.get("SANJI_GAP_PADDING", "60.0"))

# Validation
VALIDATION_TOLERANCE = int(os.environ.get("SANJI_VALIDATION_TOLERANCE", "30"))

# Patterns that indicate the reactor is introducing the next song.
# Ordered by specificity — more specific patterns are checked first.
INTRO_PATTERNS = [
    # Direct song introduction
    r"\b(next up|next one|next request|next song)\b",
    r"\b(let'?s check out|let'?s listen|let'?s see|let'?s hear|let'?s go)\b",
    r"\b(this one is|this is from|this is by|this next one)\b",
    r"\b(someone requested|we got a request|request from)\b",
    # Transitional phrases
    r"\b(alright so|okay so|all right so)\b",
    r"\b(moving on|on to the next)\b",
]

# Video encoding defaults
VIDEO_CRF = os.environ.get("SANJI_VIDEO_CRF", "18")
VIDEO_PRESET = os.environ.get("SANJI_VIDEO_PRESET", "fast")
AUDIO_BITRATE = os.environ.get("SANJI_AUDIO_BITRATE", "192k")
