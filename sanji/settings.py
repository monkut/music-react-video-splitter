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

# Batch worker — env var names and S3 layout
JOB_MESSAGE_ENV = "SANJI_JOB_MESSAGE"
RESULTS_BUCKET_ENV = "SANJI_RESULTS_BUCKET"
RESULTS_ROOT = os.environ.get("SANJI_RESULTS_ROOT", "results")
RESULT_MARKER_NAME = os.environ.get("SANJI_RESULT_MARKER_NAME", "result.json")

# Presigned result URL TTL (seconds)
PRESIGN_EXPIRY_SECONDS = int(os.environ.get("SANJI_PRESIGN_EXPIRY_SECONDS", "3600"))

# Direct upload (issue #65) — S3 layout, presign TTL, and size limits
UPLOADS_BUCKET_ENV = "SANJI_UPLOADS_BUCKET"
UPLOADS_ROOT = "uploads"
UPLOAD_PRESIGN_EXPIRY_SECONDS = int(
    os.environ.get("SANJI_UPLOAD_PRESIGN_EXPIRY_SECONDS", "3600")
)
# Multi-hour 1080p streams run 10-15 GB; cap uploads at 20 GiB by default.
UPLOAD_MAX_BYTES = int(os.environ.get("SANJI_UPLOAD_MAX_BYTES", str(20 * 1024**3)))
UPLOAD_PART_BYTES = int(os.environ.get("SANJI_UPLOAD_PART_BYTES", str(100 * 1024**2)))
UPLOAD_MULTIPART_THRESHOLD_BYTES = int(
    os.environ.get("SANJI_UPLOAD_MULTIPART_THRESHOLD_BYTES", str(100 * 1024**2))
)


def get_uploads_bucket() -> str | None:
    """Uploads bucket; falls back to the results bucket for single-bucket deploys."""
    return os.getenv(UPLOADS_BUCKET_ENV) or os.getenv(RESULTS_BUCKET_ENV)


# Stripe env var names (values loaded at runtime via validate_stripe_env_vars)
STRIPE_SECRET_KEY_ENV = "STRIPE_SECRET_KEY"
STRIPE_WEBHOOK_SECRET_ENV = "STRIPE_WEBHOOK_SECRET"
STRIPE_PUBLISHABLE_KEY_ENV = "STRIPE_PUBLISHABLE_KEY"

_REQUIRED_STRIPE_VARS = (
    STRIPE_SECRET_KEY_ENV,
    STRIPE_WEBHOOK_SECRET_ENV,
    STRIPE_PUBLISHABLE_KEY_ENV,
)

# Flask session-signing key env var name (value loaded at runtime in create_app)
SECRET_KEY_ENV = "SECRET_KEY"


def validate_secret_key_env_var() -> str:
    """Return the session-signing key, raising ValueError at app startup when unset.

    A hardcoded fallback would ship a publicly known signing key, letting anyone
    forge session cookies (issue #31) — fail fast instead.
    """
    secret_key = os.getenv(SECRET_KEY_ENV)
    if not secret_key:
        raise ValueError(f"Missing required environment variable: {SECRET_KEY_ENV}")
    return secret_key


def validate_stripe_env_vars() -> None:
    """Raise ValueError at app startup if any Stripe env var is absent."""
    missing = [var for var in _REQUIRED_STRIPE_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing required Stripe environment variable(s): {', '.join(missing)}"
        )
