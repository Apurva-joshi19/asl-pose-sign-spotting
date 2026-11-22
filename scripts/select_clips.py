"""Select How2Sign clips for annotation (plan 6.4).

Score = number of distinct lexicon signs whose English triggers occur in the sentence (whole-word match).
Keep: score >= 2, duration <= 12 s, keypoints + RGB available. At most 3 clips per VIDEO_ID.
Sample (seed 42): test 80, val 50, train 300, each +15% reserve.

Output: annotations/selected_clips.csv (row order = annotation order; role main/reserve).
"""
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LEX = ROOT / "lexicon.csv"
OUT = ROOT / "annotations" / "selected_clips.csv"
QUOTA = {"test": 80, "val": 50, "train": 300}
RESERVE = 0.15
MAX_SEC = 12.0
MIN_SCORE = 2
MAX_PER_VIDEO = 3
SEED = 42


def norm(text):
    return text.lower().replace("\u2019", "'").replace("\u2018", "'")


def load_lexicon():
    lex = pd.read_csv(LEX, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    pats = []
    for _, r in lex.iterrows():
        trig = [t.strip().lower() for t in r["english_triggers"].split("|") if t.strip()]
        alts = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in trig)
        pats.append((r["gloss_id"], re.compile(r"(?<![a-z'])(?:" + alts + r")(?![a-z'])")))
    return pats


def main():
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8-sig"))
    pats = load_lexicon()
    print(f"lexicon: {len(pats)} signs")
    out_rows = []
    prelist_main = Counter()
    prelist_train_main = Counter()

    for split in ["test", "val", "train"]:
        cache_p = RAW / f"kp_{split}" / "index_cache.json"
        if not cache_p.exists():
            sys.exit(f"{cache_p} missing - run scripts/inspect_data.py first")
        kp_index = json.loads(cache_p.read_text(encoding="utf-8"))["index"]
        with zipfile.ZipFile(RAW / f"rgb_{split}" / manifest[f"rgb_{split}"]["file"]) as z:
            rgb = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".mp4")}

        df = pd.read_csv(RAW / f"csv_{split}" / manifest[f"csv_{split}"]["file"],
                         sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
        df["duration"] = pd.to_numeric(df["END"], errors="coerce") - pd.to_numeric(df["START"], errors="coerce")
        df["candidates"] = df["SENTENCE"].map(lambda s: [g for g, rx in pats if rx.search(norm(s))])
        df["score"] = df["candidates"].map(len)
        df["has_kp"] = df["SENTENCE_NAME"].isin(kp_index.keys())
        df["has_rgb"] = df["SENTENCE_NAME"].isin(rgb.keys())

        ok = df["has_kp"] & df["has_rgb"] & (df["duration"] > 0) & (df["duration"] <= MAX_SEC) & (df["score"] >= MIN_SCORE)
        elig = df[ok].copy()
        print(f"\n[{split}] {len(df):,} sentences -> {len(elig):,} eligible "
              f"(kp+rgb: {(df['has_kp'] & df['has_rgb']).sum():,}, <=12s: {(df['duration'] <= MAX_SEC).sum():,}, "
              f"score>=2: {(df['score'] >= MIN_SCORE).sum():,})")

        idx = list(elig.index)
        random.Random(SEED).shuffle(idx)
        target = math.ceil(QUOTA[split] * (1 + RESERVE))
        per_video = Counter()
        picked = []
        for i in idx:
            vid = elig.at[i, "VIDEO_ID"]
            if per_video[vid] >= MAX_PER_VIDEO:
                continue
            per_video[vid] += 1
            picked.append(i)
            if len(picked) == target:
                break
        if len(picked) < target:
            print(f"  WARNING: only {len(picked)} clips available for target {target}")

        for order, i in enumerate(picked, 1):
            r = elig.loc[i]
            role = "main" if order <= QUOTA[split] else "reserve"
            cands = r["candidates"]
            out_rows.append({
                "split": split, "order": order, "role": role,
                "SENTENCE_NAME": r["SENTENCE_NAME"], "VIDEO_ID": r["VIDEO_ID"],
                "START": r["START"], "END": r["END"], "duration": round(float(r["duration"]), 3),
                "n_json_frames": kp_index[r["SENTENCE_NAME"]],
                "score": int(r["score"]), "candidates": "|".join(cands),
                "rgb_member": rgb[r["SENTENCE_NAME"]], "SENTENCE": r["SENTENCE"],
            })
            if role == "main":
                prelist_main.update(cands)
                if split == "train":
                    prelist_train_main.update(cands)

        sel = [x for x in out_rows if x["split"] == split]
        main_sel = [x for x in sel if x["role"] == "main"]
        print(f"  selected {len(sel)} ({len(main_sel)} main + {len(sel) - len(main_sel)} reserve) "
              f"from {len(set(x['VIDEO_ID'] for x in sel))} videos; "
              f"mean score {sum(x['score'] for x in main_sel) / max(1, len(main_sel)):.2f}; "
              f"mean length {sum(x['duration'] for x in main_sel) / max(1, len(main_sel)):.2f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nWrote {OUT} ({len(out_rows)} rows)")

    all_ids = [g for g, _ in pats]
    print("\nPre-listed counts in MAIN clips (all splits / train only). English-based, NOT labels:")
    line = []
    for g in sorted(all_ids, key=lambda g: -prelist_main[g]):
        line.append(f"{g}({prelist_main[g]}/{prelist_train_main[g]})")
        if len(line) == 6:
            print("  " + "  ".join(line))
            line = []
    if line:
        print("  " + "  ".join(line))
    rare = [g for g in all_ids if prelist_train_main[g] < 3]
    print(f"\nSigns pre-listed in < 3 train main clips: {len(rare)} -> {', '.join(rare) if rare else 'none'}")


if __name__ == "__main__":
    main()
