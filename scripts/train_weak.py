"""Train pose sign-spotting models on translation-derived weak labels (PLAN_v2).

Examples:
  python scripts/train_weak.py --model transformer --seed 1
  python scripts/train_weak.py --model bigru --seed 2 --out_dir /content/drive/MyDrive/ASL_colab/outputs/bigru_full_seed2
  python scripts/train_weak.py --model transformer --no-velocity --seed 3 --resume
  python scripts/train_weak.py --max_train_clips 400 --max_val_clips 200 --epochs 2 --batch 32 --warmup 10 --out_dir checkpoints/_smoke_weak

Loss BCEWithLogits(pos_weight from train); AdamW lr 5e-4 wd 0.01; warmup then cosine; batch 64; grad clip 1.0;
length-bucketed batches; mixed precision on CUDA; up to 40 epochs; early stopping patience 6 on val mAP.
Saves in out_dir: config.json, log.csv, last.pt (every epoch, for --resume), best.pt (best val mAP), final.json.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import feature_dim  # noqa: E402
from src.ml_metrics import average_precisions  # noqa: E402
from src.spotter import Spotter  # noqa: E402
from src.weak_dataset import WeakSpotDataset, collate_weak  # noqa: E402

LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_map", "val_micro_ap", "lr", "seconds", "improved"]


class BucketBatchSampler(Sampler):
    """Shuffle, group into pools of batch*50, sort each pool by length, cut into batches, shuffle batches."""

    def __init__(self, lengths, batch, seed):
        self.lengths, self.batch, self.rng, self.epoch_seed = np.asarray(lengths), batch, random.Random(seed), seed

    def set_state(self, state):
        self.rng.setstate(state)

    def get_state(self):
        return self.rng.getstate()

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        self.rng.shuffle(idx)
        pool = self.batch * 50
        batches = []
        for k in range(0, len(idx), pool):
            chunk = sorted(idx[k:k + pool], key=lambda i: self.lengths[i])
            batches += [chunk[j:j + self.batch] for j in range(0, len(chunk), self.batch)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch)


def run_name(a):
    return f"{a.model}_{'novel' if a.no_velocity else 'full'}_seed{a.seed}"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(torch.initial_seed() % (2 ** 32))


@torch.no_grad()
def evaluate(model, loader, device, crit, amp):
    model.eval()
    tot, nb, ys, ss = 0.0, 0, [], []
    for b in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, _, _ = model(b["x"].to(device), b["lengths"].to(device))
        logits = logits.float()
        tot += crit(logits, b["y"].to(device)).item()
        nb += 1
        ys.append(b["y"].numpy())
        ss.append(torch.sigmoid(logits).cpu().numpy())
    y, s = np.concatenate(ys), np.concatenate(ss)
    return tot / max(1, nb), average_precisions(y, s)


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["transformer", "bigru"], default="transformer")
    ap.add_argument("--no-velocity", dest="no_velocity", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--weak_dir", default=str(ROOT / "weak"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=-1, help="-1 = 2 on CUDA, 0 on CPU")
    ap.add_argument("--max_train_clips", type=int, default=0)
    ap.add_argument("--max_val_clips", type=int, default=0)
    a = ap.parse_args()

    velocity = not a.no_velocity
    name = run_name(a)
    out = Path(a.out_dir) if a.out_dir else ROOT / "checkpoints" / name
    out.mkdir(parents=True, exist_ok=True)
    final_p, last_p, best_p, log_p = out / "final.json", out / "last.pt", out / "best.pt", out / "log.csv"
    if final_p.exists() and not a.resume:
        print(f"[{name}] already finished ({final_p}); nothing to do.")
        return
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          (a.device if a.device != "auto" else "cpu"))
    amp = device.type == "cuda"
    workers = (2 if device.type == "cuda" else 0) if a.workers < 0 else a.workers
    set_seed(a.seed)

    train_ds = WeakSpotDataset("train", velocity, True, a.weak_dir, a.seed, a.max_train_clips or None)
    val_ds = WeakSpotDataset("val", velocity, False, a.weak_dir, a.seed, a.max_val_clips or None)
    classes = train_ds.classes
    sampler = BucketBatchSampler(train_ds.ends - train_ds.starts, a.batch, a.seed)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_weak, num_workers=workers,
                          worker_init_fn=worker_init if workers else None, persistent_workers=bool(workers),
                          pin_memory=device.type == "cuda")
    val_order = np.argsort(val_ds.ends - val_ds.starts).tolist()
    val_batches = [val_order[k:k + a.batch] for k in range(0, len(val_order), a.batch)]
    val_dl = DataLoader(val_ds, batch_sampler=val_batches, collate_fn=collate_weak, num_workers=0)

    model = Spotter(a.model, feature_dim(velocity), len(classes)).to(device)
    pos_weight = train_ds.pos_weight().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total_steps = a.epochs * len(sampler)

    def lr_lambda(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        prog = min(1.0, (step - a.warmup) / max(1, total_steps - a.warmup))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_map, best_epoch, bad = 1, -1.0, 0, 0

    if a.resume and last_p.exists():
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_map, best_epoch, bad = ck["best_map"], ck["best_epoch"], ck["bad"]
        torch.set_rng_state(ck["rng_torch"])
        np.random.set_state(ck["rng_numpy"])
        random.setstate(ck["rng_python"])
        train_ds.rng.setstate(ck["rng_dataset"])
        sampler.set_state(ck["rng_sampler"])
        if ck.get("stopped"):
            print(f"[{name}] run had already stopped early at epoch {ck['epoch']}.")
            start_epoch = a.epochs + 1
        print(f"[{name}] resumed from epoch {ck['epoch']} (best val mAP {best_map:.4f} at epoch {best_epoch})")
    elif log_p.exists():
        log_p.unlink()

    cfg = {"run": name, "task": "weak-label sign spotting (PLAN_v2)", "model": a.model, "velocity": velocity,
           "in_dim": feature_dim(velocity), "n_classes": len(classes), "classes": classes, "seed": a.seed,
           "epochs": a.epochs, "patience": a.patience, "batch": a.batch, "lr": a.lr, "wd": a.wd, "warmup": a.warmup,
           "n_train": len(train_ds), "n_val": len(val_ds), "device": str(device), "amp": amp,
           "torch": torch.__version__, "pos_weight_cap": 100.0}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[{name}] device {device} (amp={amp}, workers={workers}); train {len(train_ds):,} / val {len(val_ds):,} clips; "
          f"F={feature_dim(velocity)}; classes={len(classes)}; params={sum(p.numel() for p in model.parameters()):,}; "
          f"steps/epoch={len(sampler)}")

    stopped = False
    epoch = start_epoch - 1
    for epoch in range(start_epoch, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for b in train_dl:
            with torch.autocast(device_type=device.type, enabled=amp):
                logits, _, _ = model(b["x"].to(device, non_blocking=True), b["lengths"].to(device))
            loss = crit(logits.float(), b["y"].to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        val_loss, m = evaluate(model, val_dl, device, crit, amp)
        improved = m["map"] > best_map + 1e-6
        if improved:
            best_map, best_epoch, bad = m["map"], epoch, 0
            save_atomic({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_map": m["map"],
                         "val_loss": val_loss}, best_p)
        else:
            bad += 1
        stopped = bad >= a.patience
        row = {"epoch": epoch, "train_loss": round(tot / max(1, nb), 5), "val_loss": round(val_loss, 5),
               "val_map": round(m["map"], 5), "val_micro_ap": round(m["micro_ap"], 5),
               "lr": f"{sched.get_last_lr()[0]:.3e}", "seconds": round(time.time() - t0, 1), "improved": int(improved)}
        new = not log_p.exists()
        with open(log_p, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(row)
        save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best_map": best_map, "best_epoch": best_epoch,
                     "bad": bad, "stopped": stopped, "config": cfg, "rng_torch": torch.get_rng_state(),
                     "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
                     "rng_dataset": train_ds.rng.getstate(), "rng_sampler": sampler.get_state()}, last_p)
        print(f"[{name}] epoch {epoch:3d} train_loss {row['train_loss']:.4f} val_loss {row['val_loss']:.4f} "
              f"val_mAP {row['val_map']:.4f} val_microAP {row['val_micro_ap']:.4f} {'*' if improved else ' '} {row['seconds']}s",
              flush=True)
        if stopped:
            print(f"[{name}] early stopping: no improvement for {a.patience} epochs")
            break

    final = {"run": name, "best_val_map": best_map, "best_epoch": best_epoch, "last_epoch": epoch,
             "stopped_early": stopped, "finished": time.strftime("%Y-%m-%d %H:%M")}
    final_p.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[{name}] done: best val mAP {best_map:.4f} at epoch {best_epoch}; files in {out}")


if __name__ == "__main__":
    main()
