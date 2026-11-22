"""Multi-label dataset for pose sign spotting with translation-derived weak labels (PLAN_v2).

Files (weak/): features_{split}.npz (data float16 (sumT,104), offsets, names), feature_stats.npz (206 dims, train only),
               classes.json (class order), labels_{split}.json ({SENTENCE_NAME: [signs present]}).
Item: x (T,F) standardised (F=206, or 104 without velocity), y (C,) multi-hot float, name.
Train-only augmentation (scale +-10%, rotation +-10 deg, speed 0.9-1.1x) reused from src.dataset; velocity recomputed after.

Self-test:  python -m src.weak_dataset --check
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import add_velocity, augment, select_stats
from src.features import N_POS, feature_dim, load_stats

ROOT = Path(__file__).resolve().parents[1]
WEAK = ROOT / "weak"


def load_classes(weak_dir=WEAK):
    return json.loads((Path(weak_dir) / "classes.json").read_text(encoding="utf-8"))


class WeakSpotDataset(Dataset):
    def __init__(self, split, velocity=True, augment_data=None, weak_dir=WEAK, seed=0, max_clips=None):
        weak_dir = Path(weak_dir)
        z = np.load(weak_dir / f"features_{split}.npz")
        self.data, off = z["data"], z["offsets"]
        names = [str(n) for n in z["names"]]
        rows = list(range(len(names)))
        if max_clips and max_clips < len(rows):
            random.Random(seed).shuffle(rows)
            rows = sorted(rows[:max_clips])
        self.rows = rows
        self.starts, self.ends = off[:-1][rows], off[1:][rows]
        self.names = [names[i] for i in rows]
        self.classes = load_classes(weak_dir)
        cidx = {c: i for i, c in enumerate(self.classes)}
        labels = json.loads((weak_dir / f"labels_{split}.json").read_text(encoding="utf-8"))
        self.y = np.zeros((len(self.names), len(self.classes)), np.float32)
        for i, n in enumerate(self.names):
            for g in labels.get(n, []):
                if g in cidx:
                    self.y[i, cidx[g]] = 1.0
        self.velocity = velocity
        self.stats = select_stats(load_stats(weak_dir / "feature_stats.npz"), velocity)
        self.augment = (split == "train") if augment_data is None else augment_data
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.names)

    def raw(self, i):
        return self.data[self.starts[i]:self.ends[i]].astype(np.float32)

    def __getitem__(self, i):
        x = self.raw(i)
        pos, flags = x[:, :N_POS], x[:, N_POS:]
        if self.augment:
            pos, flags = augment(pos, flags, self.rng)
        f = add_velocity(pos, flags, self.velocity)
        f = (f - self.stats["mean"]) / self.stats["std"]
        return torch.from_numpy(f.astype(np.float32)), torch.from_numpy(self.y[i]), self.names[i]

    def pos_weight(self, cap=100.0):
        pos = self.y.sum(0)
        neg = len(self.y) - pos
        return torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, cap), dtype=torch.float32)


def collate_weak(batch):
    xs, ys, names = zip(*batch)
    lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    B, T, F = len(xs), int(lens.max()), xs[0].shape[1]
    x = torch.zeros(B, T, F, dtype=torch.float32)
    mask = torch.ones(B, T, dtype=torch.bool)
    for i, xi in enumerate(xs):
        x[i, :len(xi)] = xi
        mask[i, :len(xi)] = False
    return {"x": x, "mask": mask, "lengths": lens, "y": torch.stack(ys), "names": list(names)}


def _check():
    ok = True
    stats_csv = pd.read_csv(WEAK / "label_stats.csv").set_index("class")
    classes = load_classes()
    for split in ("val", "test", "train"):
        ds = WeakSpotDataset(split, velocity=True, augment_data=False)
        n_txt = len([l for l in (WEAK / f"clips_{split}.txt").read_text(encoding="utf-8").splitlines() if l.strip()])
        pos = ds.y.sum(0).astype(int)
        mism = [c for i, c in enumerate(classes) if pos[i] != int(stats_csv.at[c, f"{split}_pos"])]
        good = len(ds) == n_txt and not mism
        ok &= good
        print(f"[{split}] clips {len(ds):,} (list {n_txt:,}); classes {len(classes)}; positives match label_stats: "
              f"{not mism}; clips with no label {int((ds.y.sum(1) == 0).sum()):,} -> {'OK' if good else 'FAIL'}")

    tr = ds
    for vel in (True, False):
        d = WeakSpotDataset("val", velocity=vel, augment_data=False)
        x, y, n = d[0]
        good = x.shape[1] == feature_dim(vel) and bool(torch.isfinite(x).all()) and y.shape[0] == len(classes)
        ok &= good
        print(f"velocity={vel}: x {tuple(x.shape)}, y {tuple(y.shape)} ({int(y.sum())} positives) -> {'OK' if good else 'FAIL'}")

    aug = WeakSpotDataset("train", velocity=True, augment_data=True, seed=1)
    plain_len = aug.raw(0).shape[0]
    lens = [aug[0][0].shape[0] for _ in range(20)]
    fin = all(bool(torch.isfinite(aug[i][0]).all()) for i in range(0, len(aug), max(1, len(aug) // 200)))
    good = fin and all(abs(l - plain_len) <= int(np.ceil(0.12 * plain_len)) + 1 for l in lens)
    ok &= good
    print(f"augmentation: plain length {plain_len}, augmented {min(lens)}-{max(lens)}; ~200 augmented clips finite: {fin} "
          f"-> {'OK' if good else 'FAIL'}")

    b = next(iter(DataLoader(WeakSpotDataset("val", augment_data=False), batch_size=8, collate_fn=collate_weak)))
    good = (b["x"].shape[0] == 8 and b["mask"].shape == b["x"].shape[:2] and int((~b["mask"]).sum()) == int(b["lengths"].sum())
            and tuple(b["y"].shape) == (8, len(classes)))
    ok &= good
    print(f"batch: x {tuple(b['x'].shape)}, mask {tuple(b['mask'].shape)}, y {tuple(b['y'].shape)} -> {'OK' if good else 'FAIL'}")

    idx = np.random.RandomState(0).choice(len(tr), size=min(2000, len(tr)), replace=False)
    allx = torch.cat([tr[i][0] for i in idx]).numpy()[:, :-2]
    m, s = float(np.abs(allx.mean(0)).max()), allx.std(0)
    good = m < 0.15 and 0.6 < float(np.median(s)) < 1.4
    ok &= good
    print(f"standardisation (2000 train clips, non-flag dims): max |mean| {m:.3f}, median std {np.median(s):.3f} "
          f"-> {'OK' if good else 'FAIL'}")

    pw = tr.pos_weight()
    print(f"pos_weight: min {pw.min():.1f}, median {pw.median():.1f}, max {pw.max():.1f} "
          f"(class {classes[int(pw.argmax())]} rarest, {classes[int(pw.argmin())]} most common)")
    print("ALL WEAK DATASET CHECKS PASSED" if ok else "SOME WEAK DATASET CHECKS FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        _check()
    else:
        print(__doc__)
