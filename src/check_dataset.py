import csv
from collections import defaultdict

import numpy as np

from . import config
from .data import leading_zero_samples, list_clips, read_metadata, read_wav

TARGET = {"unknown": 40, "silence": 20}
DEFAULT_TARGET = 20

FRAME = 160
EDGE_MS = 30
CLIP_PEAK = 0.99
QUIET_PEAK = 0.05


def energy_db(x):
    n = len(x) // FRAME * FRAME
    fr = x[:n].reshape(-1, FRAME)
    return 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)


def speech_bounds(x):
    e = energy_db(x)
    if e.size == 0:
        return None
    floor = np.percentile(e, 10)
    thr = max(floor + 15, e.max() - 25)
    active = np.flatnonzero(e > thr)
    if active.size == 0:
        return None
    return active[0] * FRAME / config.SAMPLE_RATE, (active[-1] + 1) * FRAME / config.SAMPLE_RATE


def audit_clip(item):
    issues = []
    sr, raw = read_wav(item["path"])
    if sr != config.SAMPLE_RATE:
        issues.append("sample rate %d" % sr)
    if raw.dtype != np.int16:
        issues.append("dtype %s" % raw.dtype)
    expected = int(config.CLIP_SECONDS * config.SAMPLE_RATE)
    if len(raw) != expected:
        issues.append("durasi %.3f s" % (len(raw) / float(sr)))

    lead = leading_zero_samples(raw)
    x = raw[lead:].astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    is_speech_class = item["label"] != "silence"

    if peak >= CLIP_PEAK:
        issues.append("clipping (peak %.2f)" % peak)
    if is_speech_class and peak < QUIET_PEAK:
        issues.append("terlalu pelan (peak %.3f)" % peak)

    bounds = speech_bounds(x) if is_speech_class else None
    onset = offset = None
    if is_speech_class:
        if bounds is None:
            issues.append("tidak ada ucapan terdeteksi")
        else:
            onset, offset = bounds
            dur = len(x) / float(config.SAMPLE_RATE)
            if onset * 1000 < EDGE_MS:
                issues.append("awal ucapan mungkin terpotong")
            if (dur - offset) * 1000 < EDGE_MS:
                issues.append("akhir ucapan mungkin terpotong")

    return {
        "file": item["path"].name,
        "speaker": item["speaker"],
        "label": item["label"],
        "lead_zero_s": lead / float(config.SAMPLE_RATE),
        "peak": peak,
        "onset_s": onset,
        "offset_s": offset,
        "speech_s": (offset - onset) if onset is not None else None,
        "issues": issues,
    }


def print_count_table(rows):
    speakers = sorted({r["speaker"] for r in rows})
    labels = [c for c in config.CLASSES if any(r["label"] == c for r in rows)]
    labels += sorted({r["label"] for r in rows} - set(labels))
    count = defaultdict(int)
    for r in rows:
        count[(r["label"], r["speaker"])] += 1

    print("\n== Jumlah klip per kelas per speaker ==")
    print("%-18s" % "kelas" + "".join("%11s" % s for s in speakers) + "   target")
    short = []
    for lab in labels:
        target = TARGET.get(lab, DEFAULT_TARGET)
        cells = []
        for s in speakers:
            n = count[(lab, s)]
            mark = "" if n >= target else "!"
            if n < target:
                short.append((lab, s, n, target))
            cells.append("%10d%s" % (n, mark or " "))
        extra = "" if lab in config.CLASSES else "   <- bukan kelas di config.CLASSES"
        print("%-18s" % lab + "".join(cells) + "   %5d%s" % (target, extra))
    print("total: %d klip, %d speaker" % (len(rows), len(speakers)))
    for lab in config.CLASSES:
        if lab not in labels:
            print("  ! kelas %s tidak ada sama sekali" % lab)
    return short


def print_class_stats(rows):
    print("\n== Durasi ucapan per kelas (detik) ==")
    print("%-18s %8s %8s %8s %8s" % ("kelas", "onset", "durasi", "durasi", "offset"))
    print("%-18s %8s %8s %8s %8s" % ("", "median", "median", "max", "max"))
    longest = 0.0
    for lab in config.CLASSES:
        rs = [r for r in rows if r["label"] == lab and r["speech_s"] is not None]
        if not rs:
            continue
        on = np.array([r["onset_s"] for r in rs])
        sp = np.array([r["speech_s"] for r in rs])
        off = np.array([r["offset_s"] for r in rs])
        longest = max(longest, sp.max())
        print("%-18s %8.2f %8.2f %8.2f %8.2f" % (lab, np.median(on), np.median(sp), sp.max(), off.max()))
    print("ucapan terpanjang: %.2f s | input model sekarang: %.2f s%s" % (
        longest, config.MODEL_INPUT_SECONDS,
        "" if longest <= config.MODEL_INPUT_SECONDS else "  <- NAIKKAN MODEL_INPUT_SECONDS"))

    leads = np.array([r["lead_zero_s"] for r in rows])
    print("\nnol digital di awal klip: min %.3f, median %.3f, max %.3f s (dibuang otomatis saat load)"
          % (leads.min(), np.median(leads), leads.max()))


def check_metadata(rows):
    meta = read_metadata()
    if not meta:
        print("\n! dataset/metadata.csv tidak ditemukan")
        return
    on_disk = {r["file"] for r in rows}
    in_meta = {m["file"] for m in meta}
    missing_file = sorted(in_meta - on_disk)
    missing_meta = sorted(on_disk - in_meta)
    print("\n== metadata.csv ==")
    print("baris: %d | file di metadata tapi tidak ada di disk: %d | file di disk tanpa metadata: %d"
          % (len(meta), len(missing_file), len(missing_meta)))
    for f in missing_file[:10]:
        print("  tidak ada di disk:", f)
    for f in missing_meta[:10]:
        print("  tanpa metadata   :", f)


def write_reports(rows):
    config.REPORTS_DIR.mkdir(exist_ok=True)
    summary = config.REPORTS_DIR / "dataset_summary.csv"
    issues = config.REPORTS_DIR / "dataset_issues.csv"
    cols = ["file", "speaker", "label", "lead_zero_s", "peak", "onset_s", "offset_s", "speech_s"]
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["issues"])
        for r in rows:
            w.writerow([r[c] for c in cols] + ["; ".join(r["issues"])])
    with open(issues, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "speaker", "label", "issue"])
        for r in rows:
            for i in r["issues"]:
                w.writerow([r["file"], r["speaker"], r["label"], i])
    return summary, issues


def main():
    items = list_clips()
    if not items:
        raise SystemExit("Tidak ada klip di %s" % config.DATASET_DIR)
    rows = [audit_clip(it) for it in items]

    short = print_count_table(rows)
    print_class_stats(rows)
    check_metadata(rows)

    flagged = [r for r in rows if r["issues"]]
    by_issue = defaultdict(int)
    for r in flagged:
        for i in r["issues"]:
            by_issue[i.split(" (")[0]] += 1
    print("\n== Masalah per klip ==")
    print("%d dari %d klip ditandai" % (len(flagged), len(rows)))
    for k, v in sorted(by_issue.items(), key=lambda kv: -kv[1]):
        print("  %-32s %d" % (k, v))
    for r in flagged[:25]:
        print("  %-42s %s" % (r["file"], "; ".join(r["issues"])))
    if len(flagged) > 25:
        print("  ... (lengkapnya di reports/dataset_issues.csv)")
    if short:
        print("\nKurang dari target: " + ", ".join("%s/%s %d/%d" % s for s in short))

    summary, issues = write_reports(rows)
    print("\nLaporan: %s\n         %s" % (summary.relative_to(config.ROOT), issues.relative_to(config.ROOT)))
    print("Dengarkan klip yang ditandai, rekam ulang yang memang rusak.")


if __name__ == "__main__":
    main()
