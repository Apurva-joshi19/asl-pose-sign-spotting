"""Convert OpenPose JSON frames to keypoints/<SENTENCE_NAME>.npy (plan section 8).

Array: float32 (T, 137, 3), order pose(25) -> face(70) -> left hand(21) -> right hand(21), values (x, y, confidence).
Empty frame -> zeros. Several people -> keep the person with the highest mean pose confidence.
Frame order from the 12-digit frame number in each filename; missing numbers become zero frames (counted as gaps).

Usage: python scripts/convert_keypoints.py [--force]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KPJ = ROOT / "data" / "keypoints_json"
OUT = ROOT / "keypoints"
META = ROOT / "annotations" / "clip_meta.csv"
PARTS = [("pose_keypoints_2d", 25), ("face_keypoints_2d", 70),
         ("hand_left_keypoints_2d", 21), ("hand_right_keypoints_2d", 21)]
N_POINTS = sum(n for _, n in PARTS)  # 137
FRAME_RE = re.compile(r"_(\d{12})_keypoints\.json$")


def person_array(p):
    out = np.zeros((N_POINTS, 3), np.float32)
    o = 0
    for key, n in PARTS:
        a = np.asarray(p.get(key) or [], dtype=np.float32)
        if a.size == n * 3:
            out[o:o + n] = a.reshape(n, 3)
        o += n
    return out


def mean_pose_conf(p):
    a = np.asarray(p.get("pose_keypoints_2d") or [], dtype=np.float32)
    return float(a[2::3].mean()) if a.size >= 3 else 0.0


def convert(folder):
    files = {}
    for f in folder.glob("*_keypoints.json"):
        m = FRAME_RE.search(f.name)
        if m:
            files[int(m.group(1))] = f
    if not files:
        return None, {"frames": 0, "gaps": 0, "empty": 0, "multi": 0}
    T = max(files) + 1
    arr = np.zeros((T, N_POINTS, 3), np.float32)
    empty = multi = 0
    for i, f in files.items():
        data = json.loads(f.read_text(encoding="utf-8"))
        people = data.get("people") or []
        if not people:
            empty += 1
            continue
        if len(people) > 1:
            multi += 1
        arr[i] = person_array(max(people, key=mean_pose_conf))
    return arr, {"frames": T, "gaps": T - len(files), "empty": empty, "multi": multi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-convert even if the .npy already exists")
    args = ap.parse_args()
    if not META.exists():
        sys.exit("annotations/clip_meta.csv missing - run scripts/extract_clips.py first")
    meta = pd.read_csv(META, dtype=str, keep_default_na=False, encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    tot = {"gaps": 0, "empty": 0, "multi": 0, "frames": 0}
    mismatch = []
    for k, r in enumerate(meta.itertuples(index=False), 1):
        name = r.SENTENCE_NAME
        n_video = int(r.n_frames)
        dst = OUT / f"{name}.npy"
        if dst.exists() and not args.force:
            try:
                if np.load(dst, mmap_mode="r").shape[0] == int(r.kp_frames):
                    skipped += 1
                    continue
            except Exception:
                pass
        arr, st = convert(KPJ / name)
        if arr is None:
            failed += 1
            print(f"  FAILED (no JSON): {name}")
            continue
        np.save(dst, arr)
        done += 1
        for key in tot:
            tot[key] += st[key]
        if abs(arr.shape[0] - n_video) > 2:
            mismatch.append((name, arr.shape[0], n_video))
        if k % 50 == 0:
            print(f"  {k}/{len(meta)} clips", flush=True)

    print(f"\nconverted {done}, already present {skipped}, failed {failed} (total {len(meta)})")
    if done:
        print(f"frames converted: {tot['frames']:,}; gap frames filled with zeros: {tot['gaps']}; "
              f"frames with no person: {tot['empty']}; frames with >1 person: {tot['multi']}")
    print(f"clips where npy T differs from video frames by more than 2: {len(mismatch)}")
    for m in mismatch[:10]:
        print(f"  {m[0]}: npy T={m[1]}, video frames={m[2]}")


if __name__ == "__main__":
    main()
