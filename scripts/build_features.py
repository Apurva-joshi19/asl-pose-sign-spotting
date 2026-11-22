"""Stream How2Sign keypoint archives directly to compact 15-fps pose features (PLAN_v2).

For every clip in weak/clips_{split}.txt: JSON frames -> (T,137,3) (same rules as convert_keypoints.py)
-> src.features.clip_features(velocity=False) -> (T15, 104) = 102 positions + 2 hand flags, stored as float16.

Outputs:
  weak/features_{split}.npz : data (sum T15, 104) float16, offsets (n+1,), names (n,)
  weak/feature_stats.npz    : train-only mean/std over 206 dims (velocity recomputed as in src.dataset.add_velocity)
Check: features of clips also present in keypoints/*.npy are recomputed and compared.

Usage: python scripts/build_features.py [--force]
"""
import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from convert_keypoints import FRAME_RE, N_POINTS, mean_pose_conf, person_array  # noqa: E402
from src.dataset import add_velocity  # noqa: E402
from src.features import KP_DIR, N_POS, clip_features, feature_dim  # noqa: E402

RAW = ROOT / "data" / "raw"
WEAK = ROOT / "weak"
WL = ROOT / "annotations" / "weak_labels.csv"
ORDER = ["val", "test", "train"]


def frame_array(data):
    people = data.get("people") or []
    if not people:
        return np.zeros((N_POINTS, 3), np.float32)
    return person_array(max(people, key=mean_pose_conf))


def build_split(split, manifest, wl, force):
    out = WEAK / f"features_{split}.npz"
    if out.exists() and not force:
        print(f"[{split}] {out.name} exists - skipped (use --force to rebuild)")
        return
    names = [l.strip() for l in (WEAK / f"clips_{split}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    need = {n: int(wl.at[n, "n_frames"]) for n in names}
    eff = {n: float(wl.at[n, "effective_fps"]) for n in names}
    pending, feats = {}, {}

    def finalize(f):
        d = pending.pop(f)
        arr = np.zeros((max(d) + 1, N_POINTS, 3), np.float32)
        for i, a in d.items():
            arr[i] = a
        feats[f] = clip_features(arr, eff[f], velocity=False).astype(np.float16)

    tp = RAW / f"kp_{split}" / manifest[f"kp_{split}"]["file"]
    print(f"[{split}] streaming {tp.name} for {len(names):,} clips ...", flush=True)
    t0 = time.time()
    members = 0
    with tarfile.open(tp, mode="r|gz") as tf:
        for m in tf:
            members += 1
            if members % 250000 == 0:
                print(f"  [{split}] {members:,} members, {len(feats):,}/{len(names):,} clips done, {time.time() - t0:.0f}s", flush=True)
            if not m.isfile() or not m.name.endswith("_keypoints.json"):
                continue
            parts = m.name.split("/")
            if len(parts) < 2 or parts[-2] not in need:
                continue
            mm = FRAME_RE.search(parts[-1])
            if not mm:
                continue
            folder = parts[-2]
            pending.setdefault(folder, {})[int(mm.group(1))] = frame_array(json.loads(tf.extractfile(m).read()))
            if len(pending[folder]) == need[folder]:
                finalize(folder)
    incomplete = len(pending)
    for f in list(pending):
        finalize(f)
    kept = [n for n in names if n in feats]
    lens = np.array([len(feats[n]) for n in kept], dtype=np.int64)
    data = np.concatenate([feats[n] for n in kept]) if kept else np.zeros((0, N_POS + 2), np.float16)
    np.savez(out, data=data, offsets=np.r_[0, np.cumsum(lens)], names=np.array(kept))
    print(f"[{split}] done in {time.time() - t0:.0f}s: {len(kept):,}/{len(names):,} clips, {len(data):,} steps, "
          f"{out.stat().st_size / 1e6:.0f} MB; incomplete clips {incomplete}; missing {len(names) - len(kept)}")


def load_split(split):
    z = np.load(WEAK / f"features_{split}.npz")
    names = [str(s) for s in z["names"]]
    off = z["offsets"]
    return names, z["data"], off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8-sig"))
    wl = pd.read_csv(WL, dtype=str, keep_default_na=False, encoding="utf-8").set_index("SENTENCE_NAME")
    for split in ORDER:
        build_split(split, manifest, wl, a.force)

    # check against the 495 clips converted the old way
    diffs, checked = [], 0
    for split in ORDER:
        names, data, off = load_split(split)
        idx = {n: i for i, n in enumerate(names)}
        for p in KP_DIR.glob("*.npy"):
            n = p.stem
            if n not in idx:
                continue
            ref = clip_features(np.load(p), float(wl.at[n, "effective_fps"]), velocity=False)
            got = data[off[idx[n]]:off[idx[n] + 1]].astype(np.float32)
            if ref.shape != got.shape:
                diffs.append(float("inf"))
            else:
                diffs.append(float(np.abs(ref - got).max()))
            checked += 1
    big = sum(d > 0.02 for d in diffs)
    print(f"\ncheck vs keypoints/*.npy: {checked} clips compared; max abs diff {max(diffs) if diffs else float('nan'):.4f}; "
          f"clips with diff > 0.02: {big}  -> {'OK' if checked and big == 0 else 'CHECK'}")

    # train-only standardisation stats over 206 dims
    names, data, off = load_split("train")
    s1 = np.zeros(feature_dim(True), np.float64)
    s2 = np.zeros(feature_dim(True), np.float64)
    n_steps = 0
    for i in range(len(names)):
        x = data[off[i]:off[i + 1]].astype(np.float32)
        full = add_velocity(x[:, :N_POS], x[:, N_POS:], velocity=True).astype(np.float64)
        s1 += full.sum(0)
        s2 += (full ** 2).sum(0)
        n_steps += len(full)
    mean = s1 / n_steps
    std = np.sqrt(np.maximum(s2 / n_steps - mean ** 2, 0.0))
    std[std < 1e-6] = 1.0
    mean[-2:], std[-2:] = 0.0, 1.0
    np.savez(WEAK / "feature_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32),
             velocity=np.array(True), n_clips=np.array(len(names)), n_steps=np.array(n_steps))
    print(f"feature_stats: {len(names):,} train clips, {n_steps:,} steps, 206 dims -> weak/feature_stats.npz")


if __name__ == "__main__":
    main()
