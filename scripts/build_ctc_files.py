"""Build CTC files from gloss annotations (plan section 8).

Rules (in order):
  1. sure marks only; test: drop clips with any unsure mark; train/val: keep clips with <= 1 unsure mark (mark dropped)
  2. drop glosses not in the (pruned) lexicon
  3. drop glosses with < min_train_count occurrences in train (from vocab and all labels)
  4. drop clips left with no label
  5. drop clips violating the CTC length rule ceil(T15/2) >= len + adjacent repeats

Writes to --out: vocab.json, labels_{split}.json, clips_{split}.txt, feature_stats.npz (train only), build_report.json

Usage:
  python scripts/build_ctc_files.py
  python scripts/build_ctc_files.py --ann annotations/pilot_annotations.csv --out ctc_pilot_test --min-train-count 1
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import ctc_min_len  # noqa: E402
from src.features import KP_DIR, TARGET_FPS, compute_stats, load_meta  # noqa: E402

SPLITS = ["train", "val", "test"]


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(ROOT / "annotations" / "how2sign_gloss_v1.csv"))
    ap.add_argument("--lexicon", default=str(ROOT / "lexicon.csv"))
    ap.add_argument("--out", default=str(ROOT / "ctc"))
    ap.add_argument("--min-train-count", type=int, default=3)
    ap.add_argument("--pass", dest="pass_no", default="1")
    ap.add_argument("--resolutions", default=str(ROOT / "annotations" / "test_resolutions.csv"))
    a = ap.parse_args()

    ann_p, out = Path(a.ann), Path(a.out)
    if not ann_p.exists():
        sys.exit(f"{ann_p} not found")
    lex_ids = set(read(a.lexicon)["gloss_id"])
    ann = read(ann_p)
    ann = ann[ann["pass"] == a.pass_no].copy()
    res_p = Path(a.resolutions)
    n_resolved = 0
    if res_p.exists():
        res = read(res_p).set_index("SENTENCE_NAME")
        for i, r in ann.iterrows():
            if r["split"] == "test" and r["SENTENCE_NAME"] in res.index:
                for c in ("gloss_seq", "peak_frames", "confidence_seq"):
                    ann.at[i, c] = res.at[r["SENTENCE_NAME"], c]
                n_resolved += 1

    fps_map = load_meta()
    rep = {"annotation_file": str(ann_p), "pass": a.pass_no, "min_train_count": a.min_train_count,
           "test_resolutions_applied": n_resolved, "splits": {s: Counter() for s in SPLITS}}
    staged = {}
    for r in ann.itertuples(index=False):
        s = r.split
        if s not in SPLITS:
            continue
        st = rep["splits"][s]
        st["input_clips"] += 1
        g, c = r.gloss_seq.split(), r.confidence_seq.split()
        if len(g) != len(c):
            st["dropped_malformed"] += 1
            continue
        n_u = sum(x == "U" for x in c)
        st["input_marks"] += len(g)
        st["input_unsure_marks"] += n_u
        if s == "test" and n_u > 0:
            st["dropped_test_with_unsure"] += 1
            continue
        if s in ("train", "val") and n_u > 1:
            st["dropped_more_than_1_unsure"] += 1
            continue
        seq = [x for x, y in zip(g, c) if y == "S"]
        st["marks_removed_pruned_lexicon"] += sum(x not in lex_ids for x in seq)
        staged[r.SENTENCE_NAME] = (s, [x for x in seq if x in lex_ids])

    train_counts = Counter(x for s, seq in staged.values() if s == "train" for x in seq)
    keep = sorted(g for g, n in train_counts.items() if n >= a.min_train_count)
    keep_set = set(keep)
    rep["dropped_rare_glosses"] = {g: train_counts.get(g, 0) for g in sorted(lex_ids) if g not in keep_set}
    vocab = {"<blank>": 0, **{g: i + 1 for i, g in enumerate(keep)}}

    labels = {s: {} for s in SPLITS}
    for name, (s, seq) in staged.items():
        st = rep["splits"][s]
        before = len(seq)
        seq = [x for x in seq if x in keep_set]
        st["marks_removed_rare_gloss"] += before - len(seq)
        if not seq:
            st["dropped_empty"] += 1
            continue
        T = np.load(KP_DIR / f"{name}.npy", mmap_mode="r").shape[0]
        t15 = max(1, int(round(T * TARGET_FPS / fps_map[name])))
        if math.ceil(t15 / 2) < ctc_min_len(seq):
            st["dropped_ctc_too_short"] += 1
            continue
        labels[s][name] = seq

    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    for s in SPLITS:
        (out / f"labels_{s}.json").write_text(json.dumps(labels[s], indent=1), encoding="utf-8")
        (out / f"clips_{s}.txt").write_text("\n".join(labels[s]) + ("\n" if labels[s] else ""), encoding="utf-8")
        st = rep["splits"][s]
        toks = [x for seq in labels[s].values() for x in seq]
        st["kept_clips"] = len(labels[s])
        st["kept_tokens"] = len(toks)
        st["distinct_glosses"] = len(set(toks))
        st["mean_glosses_per_clip"] = round(len(toks) / len(labels[s]), 3) if labels[s] else 0.0

    if labels["train"]:
        compute_stats(list(labels["train"]), fps_map, velocity=True, out_path=out / "feature_stats.npz")
        rep["feature_stats"] = f"computed on {len(labels['train'])} train clips"
    else:
        rep["feature_stats"] = "NOT computed (no train clips)"
    rep["vocab_size_incl_blank"] = len(vocab)
    rep["splits"] = {s: dict(v) for s, v in rep["splits"].items()}
    (out / "build_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"vocab: {len(vocab)} entries incl. blank -> {', '.join(keep) if keep else '(empty)'}")
    print(f"dropped rare glosses (< {a.min_train_count} train occurrences): {len(rep['dropped_rare_glosses'])}")
    for s in SPLITS:
        st = rep["splits"][s]
        print(f"[{s}] input {st.get('input_clips', 0)} -> kept {st['kept_clips']} clips, {st['kept_tokens']} tokens, "
              f"{st['distinct_glosses']} distinct | dropped: test_unsure {st.get('dropped_test_with_unsure', 0)}, "
              f">1 unsure {st.get('dropped_more_than_1_unsure', 0)}, empty {st.get('dropped_empty', 0)}, "
              f"ctc_short {st.get('dropped_ctc_too_short', 0)}, malformed {st.get('dropped_malformed', 0)}")
        if st["kept_clips"] == 0:
            print(f"  WARNING: no {s} clips")
    print(f"feature stats: {rep['feature_stats']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
