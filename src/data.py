import csv
import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from . import config

FILENAME_RE = re.compile(r"^(speaker\d+)_(.+)_(\d+)\.wav$")


def read_wav(path):
    sr, x = wavfile.read(str(path))
    if x.ndim > 1:
        x = x[:, 0]
    return sr, x


def leading_zero_samples(x_int16):
    limit = int(config.MAX_LEADING_ZERO_SECONDS * config.SAMPLE_RATE)
    nz = np.flatnonzero(np.abs(x_int16[:limit].astype(np.int32)) > config.ZERO_THRESHOLD)
    return int(nz[0]) if nz.size else limit


def load_clip(path):
    _, x = read_wav(path)
    x = x[leading_zero_samples(x):]
    return x.astype(np.float32) / 32768.0


def fix_length(x, n=config.MODEL_INPUT_SAMPLES):
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def parse_name(path):
    m = FILENAME_RE.match(Path(path).name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def list_clips(root=config.DATASET_DIR):
    items = []
    for path in sorted(Path(root).glob("*/*.wav")):
        parsed = parse_name(path)
        speaker = parsed[0] if parsed else "?"
        idx = parsed[2] if parsed else -1
        items.append({"path": path, "speaker": speaker, "label": path.parent.name, "idx": idx})
    return items


def read_metadata(root=config.DATASET_DIR):
    meta = Path(root) / "metadata.csv"
    if not meta.exists():
        return []
    with open(meta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
