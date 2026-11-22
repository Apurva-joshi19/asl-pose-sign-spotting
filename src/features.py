"""Pose features for CTC recognition (plan section 9).

Per 15-fps time step: 9 upper-body points + 21 left-hand + 21 right-hand points (x, y) = 102 positions,
102 velocities, 2 hand-present flags  ->  F = 206   (no-velocity option: F = 104).

- confidence < 0.2 -> missing; gaps <= 5 frames linearly interpolated, longer gaps -> 0 (+ per-hand flag)
- body: subtract neck, divide by clip shoulder width (median over frames)
- hands: subtract that hand's wrist, divide by the hand bounding-box size
- resampled to 15 fps from each clip's effective_fps (annotations/clip_meta.csv), nearest frame
- standardisation with train-only mean/std (ctc/feature_stats.npz); flags are not standardised

Self-test:  python -m src.features --check 20
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KP_DIR = ROOT / "keypoints"
META_P = ROOT / "annotations" / "clip_meta.csv"
STATS_P = ROOT / "ctc" / "feature_stats.npz"

CONF_MIN = 0.2
MAX_GAP = 5
TARGET_FPS = 15.0
BODY_IDX = list(range(9))            # BODY_25: nose, neck, R sh, R el, R wr, L sh, L el, L wr, mid-hip
NECK, R_SH, L_SH = 1, 2, 5
LH0, RH0 = 95, 116                   # start of left / right hand in the 137-point array
N_POS = (9 + 21 + 21) * 2            # 102


def feature_dim(velocity=True):
    return N_POS * 2 + 2 if velocity else N_POS + 2


def fill_gaps(values, valid, max_gap=MAX_GAP):
    """values (T, D), valid (T,) bool. Interpolate interior gaps <= max_gap; other invalid rows -> 0."""
    out = values.astype(np.float32).copy()
    nv = valid.copy()
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            w = (np.arange(1, gap + 1, dtype=np.float32) / (gap + 1))[:, None]
            out[a + 1:b] = values[a] * (1 - w) + values[b] * w
            nv[a + 1:b] = True
    out[~nv] = 0.0
    return out, nv


def _nearest_fill(x, valid):
    """Forward/backward fill rows of x where valid is False (used only for the neck reference)."""
    if not valid.any():
        return None
    idx = np.where(valid, np.arange(len(valid)), -1)
    idx = np.maximum.accumulate(idx)
    first = np.flatnonzero(valid)[0]
    idx[idx < 0] = first
    return x[idx]


def body_features(kp):
    T = kp.shape[0]
    out = np.zeros((T, 9, 2), np.float32)
    pts = np.zeros((T, 9, 2), np.float32)
    ok = np.zeros((T, 9), bool)
    for j in BODY_IDX:
        v = kp[:, j, 2] >= CONF_MIN
        filled, nv = fill_gaps(kp[:, j, :2], v)
        pts[:, j], ok[:, j] = filled, nv
    neck = _nearest_fill(pts[:, NECK], ok[:, NECK])
    if neck is None:
        anyok = ok.any(axis=1)
        if not anyok.any():
            return out.reshape(T, 18)
        neck = np.where(anyok[:, None], (pts * ok[..., None]).sum(1) / np.maximum(ok.sum(1, keepdims=True), 1), 0)
    both = ok[:, R_SH] & ok[:, L_SH]
    if both.any():
        sw = float(np.median(np.linalg.norm(pts[both, R_SH] - pts[both, L_SH], axis=1)))
    else:
        sw = 0.0
    if sw < 1e-6:
        span = pts[ok].reshape(-1, 2)
        sw = float(np.ptp(span[:, 0])) if len(span) > 1 else 1.0
        sw = sw if sw > 1e-6 else 1.0
    out = (pts - neck[:, None, :]) / sw
    out[~ok] = 0.0
    return out.reshape(T, 18).astype(np.float32)


def hand_features(kp, start):
    T = kp.shape[0]
    h = kp[:, start:start + 21, :]                       # (T, 21, 3)
    present = h[:, :, 2].mean(axis=1) >= CONF_MIN
    filled, nv = fill_gaps(h.reshape(T, 63), present)
    h = filled.reshape(T, 21, 3)
    out = np.zeros((T, 21, 2), np.float32)
    for t in np.flatnonzero(nv):
        good = h[t, :, 2] >= CONF_MIN
        if good.sum() < 2:
            nv[t] = False
            continue
        wrist = h[t, 0, :2] if good[0] else h[t, good, :2].mean(axis=0)
        g = h[t, good, :2]
        size = float(max(np.ptp(g[:, 0]), np.ptp(g[:, 1])))
        size = size if size > 1e-6 else 1.0
        loc = (h[t, :, :2] - wrist) / size
        loc[~good] = 0.0
        out[t] = loc
    return out.reshape(T, 42), nv.astype(np.float32)


def resample_index(T, eff_fps, target=TARGET_FPS):
    n_out = max(1, int(round(T * target / eff_fps)))
    return np.minimum(T - 1, np.round(np.arange(n_out) * eff_fps / target).astype(int))


def clip_features(kp, eff_fps, velocity=True):
    """kp: (T, 137, 3) float32 -> (T15, F) float32."""
    body = body_features(kp)
    lh, lf = hand_features(kp, LH0)
    rh, rf = hand_features(kp, RH0)
    pos = np.concatenate([body, lh, rh], axis=1)          # (T, 102)
    flags = np.stack([lf, rf], axis=1)                     # (T, 2)
    idx = resample_index(kp.shape[0], eff_fps)
    pos, flags = pos[idx], flags[idx]
    parts = [pos]
    if velocity:
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        lmask = (flags[1:, 0] * flags[:-1, 0])[:, None]
        rmask = (flags[1:, 1] * flags[:-1, 1])[:, None]
        vel[1:, 18:60] *= lmask
        vel[1:, 60:102] *= rmask
        parts.append(vel)
    parts.append(flags)
    return np.concatenate(parts, axis=1).astype(np.float32)


def load_meta():
    meta = pd.read_csv(META_P, dtype=str, keep_default_na=False, encoding="utf-8")
    return {r.SENTENCE_NAME: float(r.effective_fps) for r in meta.itertuples(index=False)}


def load_clip_features(name, eff_fps, velocity=True, stats=None):
    x = clip_features(np.load(KP_DIR / f"{name}.npy"), eff_fps, velocity)
    return standardise(x, stats) if stats is not None else x


def compute_stats(names, fps_map, velocity=True, out_path=STATS_P):
    """Train-only mean/std over all time steps; flags (last 2 dims) keep mean 0 / std 1."""
    xs = [clip_features(np.load(KP_DIR / f"{n}.npy"), fps_map[n], velocity) for n in names]
    allx = np.concatenate(xs, axis=0).astype(np.float64)
    mean = allx.mean(axis=0)
    std = allx.std(axis=0)
    std[std < 1e-6] = 1.0
    mean[-2:], std[-2:] = 0.0, 1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32),
             velocity=np.array(velocity), n_clips=np.array(len(names)), n_steps=np.array(len(allx)))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def load_stats(path=STATS_P):
    d = np.load(path)
    return {"mean": d["mean"], "std": d["std"]}


def standardise(x, stats):
    return ((x - stats["mean"]) / stats["std"]).astype(np.float32)


def _check(n):
    fps_map = load_meta()
    names = sorted(fps_map)
    random.Random(42).shuffle(names)
    names = names[:n]
    t0 = time.time()
    bad = 0
    lens, ratio, lpres, rpres = [], [], [], []
    for name in names:
        kp = np.load(KP_DIR / f"{name}.npy")
        for vel in (True, False):
            x = clip_features(kp, fps_map[name], vel)
            if x.shape[1] != feature_dim(vel) or not np.isfinite(x).all():
                bad += 1
                print(f"  BAD: {name} velocity={vel} shape={x.shape} finite={np.isfinite(x).all()}")
        x = clip_features(kp, fps_map[name], True)
        lens.append(x.shape[0])
        ratio.append(x.shape[0] / (kp.shape[0] / fps_map[name] * TARGET_FPS))
        lpres.append(x[:, -2].mean())
        rpres.append(x[:, -1].mean())
    print(f"clips checked: {len(names)} in {time.time() - t0:.1f}s; problems: {bad}")
    print(f"feature dim: {feature_dim(True)} (velocity) / {feature_dim(False)} (no velocity)")
    print(f"15-fps length T': min {min(lens)}, median {int(np.median(lens))}, max {max(lens)}")
    print(f"T' / (duration x 15): min {min(ratio):.3f}, max {max(ratio):.3f}  (should be ~1.0)")
    print(f"left hand present: {100 * np.mean(lpres):.1f}% of steps; right hand present: {100 * np.mean(rpres):.1f}%")
    x = clip_features(np.load(KP_DIR / f"{names[0]}.npy"), fps_map[names[0]], True)
    print(f"example {names[0]}: shape {x.shape}, body/hand position range [{x[:, :102].min():.2f}, {x[:, :102].max():.2f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0, help="self-test on N random clips")
    a = ap.parse_args()
    if a.check:
        _check(a.check)
    else:
        print(__doc__)
        sys.exit(0)
