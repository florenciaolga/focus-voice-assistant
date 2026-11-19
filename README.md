# Focus Voice Assistant

Timer belajar yang dikontrol suara + live transcription.

- **Jalur A** (buatan sendiri): VAD manual → MFCC manual → Model 1 (wake word "focus" + 10 command)
- **Jalur B** (pretrained): segmen VAD → Whisper → live transcript → intent parser (cadangan)

## Setup

Butuh Python 3.11.

```bash
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Dataset ada di Google Drive (tidak masuk git). Download lalu gabungkan ke `dataset/<kelas>/`,
nama file `speakerXX_<kelas>_NNN.wav`, dan gabungkan isi `metadata.csv` tiap orang.
