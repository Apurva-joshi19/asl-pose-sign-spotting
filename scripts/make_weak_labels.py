"""Translation-derived weak labels for pose sign spotting (PLAN_v2).

For every How2Sign sentence with keypoints:
  labels        = lexicon signs whose english_triggers occur in the English sentence (whole-word match)
  effective_fps = keypoint frames / (END - START), snapped to the nearest known rate if within 12%
Classes kept: >= 30 train positives and >= 5 positives in val and in test.

Outputs: annotations/weak_labels.csv, weak/classes.json, weak/labels_{split}.json, weak/clips_{split}.txt,
         weak/label_stats.csv, weak/build_report.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LEX = ROOT / "lexicon.csv"
META = ROOT / "annotations" / "clip_meta.csv"
OUT_CSV = ROOT / "annotations" / "weak_labels.csv"
OUT = ROOT / "weak"
SPLITS = ["train", "val", "test"]
KNOWN_FPS = [23.976, 24.0, 25.0, 29.97, 30.0, 31.968, 32.0, 33.333, 40.0, 50.0, 59.94, 60.0, 66.667, 80.0]
SNAP_TOL = 0.12
FPS_RANGE = (15.0, 90.0)
MIN_TRAIN, MIN_EVAL = 30, 5


def norm(text):
    return text.lower().replace("\u2019", "'").replace("\u2018", "'")


def load_patterns():
    lex = pd.read_csv(LEX, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    pats = []
    for _, r in lex.iterrows():
        trig = [t.strip().lower() for t in r["english_triggers"].split("|") if t.strip()]
        alts = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in trig)
        pats.append((r["gloss_id"], re.compile(r"(?<![a-z'])(?:" + alts + r")(?![a-z'])")))
    return pats


def snap(raw):
    best = min(KNOWN_FPS, key=lambda f: abs(f - raw))
    return (best, True) if abs(best - raw) / best <= SNAP_TOL else (raw, False)


def main():
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8-sig"))
    pats = load_patterns()
    rows, rep = [], {"splits": {}}
    for split in SPLITS:
        cache = RAW / f"kp_{split}" / "index_cache.json"
        if not cache.exists():
            sys.exit(f"{cache} missing - run scripts/inspect_data.py first")
        kp_index = json.loads(cache.read_text(encoding="utf-8"))["index"]
        df = pd.read_csv(RAW / f"csv_{split}" / manifest[f"csv_{split}"]["file"], sep="\t", dtype=str,
                         keep_default_na=False, encoding="utf-8")
        st = Counter()
        for r in df.itertuples(index=False):
            st["csv_rows"] += 1
            name = r.SENTENCE_NAME
            n = kp_index.get(name, 0)
            if n == 0:
                st["dropped_no_keypoints"] += 1
                continue
            try:
                dur = float(r.END) - float(r.START)
            except ValueError:
                dur = 0.0
            if dur <= 0:
                st["dropped_bad_duration"] += 1
                continue
            raw = n / dur
            eff, snapped = snap(raw)
            if not (FPS_RANGE[0] <= eff <= FPS_RANGE[1]):
                st["dropped_fps_out_of_range"] += 1
                continue
            st["snapped_fps"] += snapped
            labels = [g for g, rx in pats if rx.search(norm(r.SENTENCE))]
            st["kept"] += 1
            rows.append({"SENTENCE_NAME": name, "VIDEO_ID": r.VIDEO_ID, "split": split, "duration": round(dur, 3),
                         "n_frames": n, "raw_fps": round(raw, 3), "effective_fps": round(eff, 3),
                         "labels": "|".join(labels), "n_labels": len(labels), "SENTENCE": r.SENTENCE})
        rep["splits"][split] = dict(st)
        print(f"[{split}] {st['csv_rows']:,} rows -> kept {st['kept']:,} (no kp {st['dropped_no_keypoints']}, "
              f"bad duration {st['dropped_bad_duration']}, fps out of range {st['dropped_fps_out_of_range']}; "
              f"snapped {st['snapped_fps']:,})")
    wl = pd.DataFrame(rows)

    # effective_fps check against video-measured values (495 clips)
    if META.exists():
        meta = pd.read_csv(META, dtype=str, keep_default_na=False).set_index("SENTENCE_NAME")
        m = wl[wl["SENTENCE_NAME"].isin(meta.index)].copy()
        m["video_eff"] = m["SENTENCE_NAME"].map(lambda s: float(meta.at[s, "effective_fps"]))
        rel = (m["effective_fps"] - m["video_eff"]).abs() / m["video_eff"]
        exact = int((rel < 0.005).sum())
        rep["fps_check"] = {"clips": len(m), "exact": exact, "within_5pct": int((rel < 0.05).sum()),
                            "max_rel_diff": round(float(rel.max()), 4)}
        print(f"\neffective_fps vs video-measured (clip_meta, {len(m)} clips): exact {exact}/{len(m)}, "
              f"within 5% {int((rel < 0.05).sum())}/{len(m)}, max relative diff {rel.max():.3f}")
        bad = m[rel >= 0.05]
        if len(bad):
            print(bad.head(8)[["SENTENCE_NAME", "n_frames", "duration", "raw_fps", "effective_fps", "video_eff"]].to_string(index=False))

    # class selection
    counts = {s: Counter(g for labs in wl.loc[wl["split"] == s, "labels"] for g in labs.split("|") if g) for s in SPLITS}
    all_ids = [g for g, _ in pats]
    keep = [g for g in all_ids if counts["train"][g] >= MIN_TRAIN and counts["val"][g] >= MIN_EVAL and counts["test"][g] >= MIN_EVAL]
    keep_set = set(keep)
    rep["classes_kept"] = keep
    rep["classes_dropped"] = {g: {s: counts[s][g] for s in SPLITS} for g in all_ids if g not in keep_set}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classes.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")
    stats = []
    for s in SPLITS:
        sub = wl[wl["split"] == s]
        labels = {r.SENTENCE_NAME: [g for g in r.labels.split("|") if g in keep_set] for r in sub.itertuples(index=False)}
        (OUT / f"labels_{s}.json").write_text(json.dumps(labels), encoding="utf-8")
        (OUT / f"clips_{s}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        n_pos = [len(v) for v in labels.values()]
        rep["splits"][s].update({"clips": len(labels), "clips_with_no_label": int(sum(x == 0 for x in n_pos)),
                                 "mean_labels_per_clip": round(float(np.mean(n_pos)), 3)})
    for g in keep:
        stats.append({"class": g, **{f"{s}_pos": counts[s][g] for s in SPLITS},
                      **{f"{s}_rate": round(counts[s][g] / max(1, (wl['split'] == s).sum()), 4) for s in SPLITS}})
    pd.DataFrame(stats).to_csv(OUT / "label_stats.csv", index=False, encoding="utf-8")
    wl.to_csv(OUT_CSV, index=False, encoding="utf-8")
    (OUT / "build_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"\nclasses kept: {len(keep)}/{len(all_ids)}; dropped: {', '.join(rep['classes_dropped']) or 'none'}")
    for s in SPLITS:
        r = rep["splits"][s]
        print(f"[{s}] {r['clips']:,} clips; no label {r['clips_with_no_label']:,} "
              f"({100 * r['clips_with_no_label'] / max(1, r['clips']):.1f}%); mean labels/clip {r['mean_labels_per_clip']}")
    print("\nclass (train / val / test positives), most frequent first:")
    line = []
    for g in sorted(keep, key=lambda g: -counts["train"][g]):
        line.append(f"{g}({counts['train'][g]}/{counts['val'][g]}/{counts['test'][g]})")
        if len(line) == 5:
            print("  " + "  ".join(line))
            line = []
    if line:
        print("  " + "  ".join(line))
    print(f"\nwrote {OUT_CSV} and {OUT}")


if __name__ == "__main__":
    main()
