from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SAMPLE_RATE = 16000
CLIP_SECONDS = 2.0

MAX_LEADING_ZERO_SECONDS = 0.6
ZERO_THRESHOLD = 8

MODEL_INPUT_SECONDS = 1.5
MODEL_INPUT_SAMPLES = int(MODEL_INPUT_SECONDS * SAMPLE_RATE)

CLASSES = [
    "focus",
    "start", "pause", "continue", "break", "stop", "status",
    "take-a-break", "end-session", "how-much-time", "add-five-minutes",
    "silence", "unknown",
]
WAKE_WORD = "focus"
COMMANDS = [c for c in CLASSES if c not in (WAKE_WORD, "silence", "unknown")]
