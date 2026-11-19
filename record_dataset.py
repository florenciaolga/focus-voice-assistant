import csv
import random
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000
DURATION = 2.0
TAKES_PER_CLASS = 20
OUT_DIR = Path("dataset")

TARGETS = {
    "focus": "focus",
    "start": "start",
    "pause": "pause",
    "continue": "continue",
    "break": "break",
    "stop": "stop",
    "status": "status",
    "take-a-break": "take a break",
    "end-session": "end session",
    "how-much-time": "how much time",
    "add-five-minutes": "add five minutes",
}

UNKNOWN_WORDS = [
    "database", "hello", "tomorrow", "coding", "water", "okay", "coffee",
    "table", "computer", "yesterday", "music", "window", "keyboard", "river",
    "algorithm", "morning", "banana", "network", "pencil", "library",
]
UNKNOWN_TAKES = 40

SILENCE_PROMPTS = [
    "hening total", "nyalakan kipas / AC", "ketik di keyboard",
    "orang ngobrol jauh", "suara ruangan biasa",
]
SILENCE_TAKES = 20


def record_clip():
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    return audio[:, 0]


def next_index(folder: Path, speaker: str, label: str) -> int:
    existing = list(folder.glob(f"{speaker}_{label}_*.wav"))
    return len(existing) + 1


def save_clip(audio, folder: Path, speaker: str, label: str, idx: int) -> Path:
    path = folder / f"{speaker}_{label}_{idx:03d}.wav"
    wavfile.write(path, SAMPLE_RATE, audio)
    return path


def log_row(speaker, label, prompt, path):
    meta = OUT_DIR / "metadata.csv"
    new = not meta.exists()
    with open(meta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "speaker", "label", "prompt"])
        w.writerow([path.name, speaker, label, prompt])


def session(speaker, label, prompts, takes):
    folder = OUT_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    idx = next_index(folder, speaker, label)
    done = idx - 1
    print(f"\n=== Kelas: {label.upper()} (sudah ada {done}/{takes}) ===")

    last_path = None
    while done < takes:
        prompt = prompts[done % len(prompts)] if isinstance(prompts, list) else prompts
        cmd = input(f"[{done + 1}/{takes}] Ucapkan: \"{prompt}\"  (Enter=rekam, r=ulang, s=skip, q=quit) ").strip().lower()
        if cmd == "q":
            raise SystemExit("Selesai. Data tersimpan.")
        if cmd == "s":
            return
        if cmd == "r" and last_path is not None:
            if last_path.exists():
                last_path.unlink()
            done -= 1
            idx -= 1
            last_path = None
            print("Klip terakhir dihapus, ulangi.")
            continue

        print("  Mulai dalam 0.5 detik...", end="", flush=True)
        time.sleep(0.5)
        print(" REKAM!")
        audio = record_clip()

        peak = int(np.max(np.abs(audio)))
        if peak < 500:
            print("  Suara terlalu pelan / mic tidak menangkap. Coba lagi.")
            continue
        if peak > 32000:
            print("  Warning: kemungkinan clipping (terlalu keras). Klip tetap disimpan, tapi jaga jarak mic.")

        last_path = save_clip(audio, folder, speaker, label, idx)
        log_row(speaker, label, prompt, last_path)
        idx += 1
        done += 1


def main():
    print("Daftar mic:")
    print(sd.query_devices())
    speaker_num = input("\nNomor speaker (misal 1): ").strip() or "1"
    speaker = f"speaker{int(speaker_num):02d}"
    print(f"Speaker: {speaker}")
    print("Tips: rekam di sesi pendek, variasikan jarak mic dan kecepatan bicara.")

    OUT_DIR.mkdir(exist_ok=True)

    for label, spoken in TARGETS.items():
        session(speaker, label, spoken, TAKES_PER_CLASS)

    words = UNKNOWN_WORDS.copy()
    random.shuffle(words)
    session(speaker, "unknown", words, UNKNOWN_TAKES)

    session(speaker, "silence", SILENCE_PROMPTS, SILENCE_TAKES)

    print("\nSemua kelas selesai. Terima kasih!")


if __name__ == "__main__":
    main()
