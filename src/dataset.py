"""Dataset + collate for pose Transformer/BiGRU CTC (plan section 9).

Files (written by scripts/build_ctc_files.py):
  ctc/vocab.json            {"<blank>": 0, "GLOSS": id, ...}
  ctc/labels_{split}.json   {SENTENCE_NAME: ["GLOSS", ...]}
  ctc/clips_{split}.txt     one SENTENCE_NAME per line
  ctc/feature_stats.npz     train-only mean/std (206 dims; no-velocity uses the matching subset)

Train-only augmentation: scale +-10%, rotation +-10 deg, speed 0.9-1.1x (velocity recomputed afterwards).
Batch: x (B,T,F), mask (B,T) True = padding, input_lengths, flattened targets, target_lengths.

Self-test:  python -m src.dataset --check
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import KP_DIR, N_POS, clip_features, feature_dim, load_meta, load_stats

ROOT = Path(__file__).resolve().parents[1]
CTC_DIR = ROOT / "ctc"
LH = slice(18, 60)
RH = slice(60, 102)


def add_velocity(pos, flags, velocity=True):
    """Same layout as features.clip_features: [pos(102), vel(102)?, flags(2)]."""
    parts = [pos]
    if velocity:
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel[1:, LH] *= (flags[1:, 0] * flags[:-1, 0])[:, None]
        vel[1:, RH] *= (flags[1:, 1] * flags[:-1, 1])[:, None]
        parts.append(vel)
    parts.append(flags)
    return np.concatenate(parts, axis=1).astype(np.float32)


def augment(pos, flags, rng):
    s = rng.uniform(0.9, 1.1)
    th = math.radians(rng.uniform(-10.0, 10.0))
    c, sn = math.cos(th), math.sin(th)
    xy = pos.reshape(len(pos), -1, 2)
    x, y = xy[..., 0], xy[..., 1]
    pos2 = np.stack([s * (c * x - sn * y), s * (sn * x + c * y)], axis=-1).reshape(len(pos), -1)
    f = rng.uniform(0.9, 1.1)
    T = len(pos2)
    n = max(1, int(round(T / f)))
    idx = np.minimum(T - 1, np.round(np.arange(n) * f).astype(int))
    return pos2[idx].astype(np.float32), flags[idx]


def select_stats(stats, velocity):
    if stats is None:
        return None
    if velocity:
        return stats
    keep = np.r_[0:N_POS, len(stats["mean"]) - 2, len(stats["mean"]) - 1]
    return {"mean": stats["mean"][keep], "std": stats["std"][keep]}


class PoseCTCDataset(Dataset):
    def __init__(self, names, targets, fps_map, velocity=True, stats=None, augment=False, seed=0):
        self.names = list(names)
        self.targets = [torch.tensor(targets[n], dtype=torch.long) for n in self.names]
        self.velocity = velocity
        self.stats = select_stats(stats, velocity) if stats is not None and len(stats["mean"]) == feature_dim(True) else stats
        self.augment = augment
        self.rng = random.Random(seed)
        self.base = []
        for n in self.names:
            x = clip_features(np.load(KP_DIR / f"{n}.npy"), fps_map[n], velocity=False)
            self.base.append((x[:, :N_POS].copy(), x[:, N_POS:].copy()))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        pos, flags = self.base[i]
        if self.augment:
            pos, flags = augment(pos, flags, self.rng)
        x = add_velocity(pos, flags, self.velocity)
        if self.stats is not None:
            x = (x - self.stats["mean"]) / self.stats["std"]
        return torch.from_numpy(x.astype(np.float32)), self.targets[i], self.names[i]


def collate(batch):
    xs, ys, names = zip(*batch)
    lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    B, T, F = len(xs), int(lens.max()), xs[0].shape[1]
    x = torch.zeros(B, T, F, dtype=torch.float32)
    mask = torch.ones(B, T, dtype=torch.bool)
    for i, xi in enumerate(xs):
        x[i, :len(xi)] = xi
        mask[i, :len(xi)] = False
    targets = torch.cat(ys) if len(ys) else torch.zeros(0, dtype=torch.long)
    tlens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return {"x": x, "mask": mask, "input_lengths": lens, "targets": targets,
            "target_lengths": tlens, "names": list(names)}


def load_vocab(ctc_dir=CTC_DIR):
    return json.loads((Path(ctc_dir) / "vocab.json").read_text(encoding="utf-8"))


def load_split(split, ctc_dir=CTC_DIR):
    ctc_dir = Path(ctc_dir)
    vocab = load_vocab(ctc_dir)
    labels = json.loads((ctc_dir / f"labels_{split}.json").read_text(encoding="utf-8"))
    names = [l.strip() for l in (ctc_dir / f"clips_{split}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    targets = {n: [vocab[g] for g in labels[n]] for n in names}
    return names, targets, vocab


def make_dataset(split, velocity=True, augment=None, ctc_dir=CTC_DIR, seed=0):
    names, targets, vocab = load_split(split, ctc_dir)
    stats = load_stats(Path(ctc_dir) / "feature_stats.npz")
    aug = (split == "train") if augment is None else augment
    ds = PoseCTCDataset(names, targets, load_meta(), velocity, select_stats(stats, velocity), aug, seed)
    ds.stats = select_stats(stats, velocity)
    return ds, vocab


def ctc_min_len(target):
    t = list(target)
    return len(t) + sum(1 for a, b in zip(t[:-1], t[1:]) if a == b)


def _check():
    fps_map = load_meta()
    names = sorted(fps_map)
    random.Random(42).shuffle(names)
    names = names[:16]
    rng = random.Random(0)
    targets = {n: [rng.randint(1, 60) for _ in range(rng.randint(1, 4))] for n in names}
    xs = [clip_features(np.load(KP_DIR / f"{n}.npy"), fps_map[n], True) for n in names]
    allx = np.concatenate(xs).astype(np.float64)
    std = allx.std(0)
    std[std < 1e-6] = 1.0
    stats = {"mean": allx.mean(0).astype(np.float32), "std": std.astype(np.float32)}
    stats["mean"][-2:], stats["std"][-2:] = 0.0, 1.0

    ok = True
    for vel in (True, False):
        ds = PoseCTCDataset(names, targets, fps_map, vel, select_stats(stats, vel), augment=False)
        x, y, n = ds[0]
        good = x.shape[1] == feature_dim(vel) and torch.isfinite(x).all().item()
        ok &= good
        print(f"velocity={vel}: item x {tuple(x.shape)}, target {y.tolist()}, finite={torch.isfinite(x).all().item()} -> {'OK' if good else 'FAIL'}")

    ds_aug = PoseCTCDataset(names, targets, fps_map, True, stats, augment=True, seed=1)
    ds_plain = PoseCTCDataset(names, targets, fps_map, True, stats, augment=False)
    x0 = ds_plain[0][0]
    la = [ds_aug[0][0].shape[0] for _ in range(20)]
    xa = ds_aug[0][0]
    differs = xa.shape != x0.shape or not torch.allclose(xa, x0)
    within = all(abs(l - x0.shape[0]) <= math.ceil(0.12 * x0.shape[0]) + 1 for l in la)
    ok &= differs and within
    print(f"augmentation: plain length {x0.shape[0]}, augmented lengths {min(la)}-{max(la)}, differs={differs}, "
          f"speed within +-10%: {within}")

    dl = DataLoader(ds_plain, batch_size=8, shuffle=False, collate_fn=collate)
    b = next(iter(dl))
    shapes_ok = (b["x"].shape[0] == 8 and b["mask"].shape == b["x"].shape[:2]
                 and int((~b["mask"]).sum()) == int(b["input_lengths"].sum())
                 and len(b["targets"]) == int(b["target_lengths"].sum()))
    ok &= shapes_ok
    print(f"batch: x {tuple(b['x'].shape)}, mask {tuple(b['mask'].shape)}, input_lengths {b['input_lengths'].tolist()}, "
          f"targets {len(b['targets'])}, target_lengths {b['target_lengths'].tolist()} -> {'OK' if shapes_ok else 'FAIL'}")

    viol = sum(1 for n, x in zip(names, xs) if math.ceil(x.shape[0] / 2) < ctc_min_len(targets[n]))
    print(f"CTC length rule ceil(T15/2) >= len + repeats: {len(names) - viol}/{len(names)} clips OK (dummy targets)")
    print("ALL DATASET CHECKS PASSED" if ok else "SOME DATASET CHECKS FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        _check()
    else:
        print(__doc__)
