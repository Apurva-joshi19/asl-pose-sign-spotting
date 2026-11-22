"""Train pose CTC models (plan section 9).

Examples:
  python scripts/train.py --model transformer --seed 1
  python scripts/train.py --model bigru --seed 2 --out_dir /content/drive/MyDrive/ASL_colab/outputs/bigru_full_seed2
  python scripts/train.py --model transformer --no-velocity --seed 3 --resume
  python scripts/train.py --dummy --epochs 2 --out_dir checkpoints/_smoke      # CPU smoke test with random labels

Saves in out_dir: config.json, log.csv, last.pt (every epoch, for --resume), best.pt (best val WER), final.json.
Early stopping: patience 20 on val WER (ties broken by lower val loss).
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import PoseCTCDataset, collate, make_dataset, select_stats  # noqa: E402
from src.features import KP_DIR, clip_features, feature_dim, load_meta  # noqa: E402
from src.metrics import corpus_wer, greedy_decode  # noqa: E402
from src.models import build_model  # noqa: E402

LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_wer", "val_S", "val_D", "val_I", "val_N", "lr", "seconds", "improved"]


def run_name(a):
    return f"{a.model}_{'novel' if a.no_velocity else 'full'}_seed{a.seed}"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def dummy_datasets(velocity, seed):
    fps = load_meta()
    names = sorted(fps)
    random.Random(0).shuffle(names)
    tr, va = names[:32], names[32:40]
    rng = random.Random(1)
    vocab = {"<blank>": 0, **{f"G{i}": i for i in range(1, 21)}}
    targets = {n: [rng.randint(1, 20) for _ in range(rng.randint(1, 3))] for n in tr + va}
    xs = [clip_features(np.load(KP_DIR / f"{n}.npy"), fps[n], True) for n in tr]
    allx = np.concatenate(xs).astype(np.float64)
    std = allx.std(0)
    std[std < 1e-6] = 1.0
    stats = {"mean": allx.mean(0).astype(np.float32), "std": std.astype(np.float32)}
    stats["mean"][-2:], stats["std"][-2:] = 0.0, 1.0
    st = select_stats(stats, velocity)
    return (PoseCTCDataset(tr, targets, fps, velocity, st, True, seed),
            PoseCTCDataset(va, targets, fps, velocity, st, False, seed), vocab)


@torch.no_grad()
def evaluate(model, loader, device, crit):
    model.eval()
    tot, nb, refs, hyps = 0.0, 0, [], []
    for b in loader:
        lp, _, ol = model(b["x"].to(device), b["input_lengths"].to(device))
        loss = crit(lp, b["targets"].to(device), ol, b["target_lengths"].to(device))
        tot += float(loss)
        nb += 1
        hyps += greedy_decode(lp, ol)
        tl = b["target_lengths"].tolist()
        t = b["targets"].tolist()
        k = 0
        for n in tl:
            refs.append(t[k:k + n])
            k += n
    return tot / max(1, nb), corpus_wer(refs, hyps)


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
    ap.add_argument("--ctc_dir", default=str(ROOT / "ctc"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--dummy", action="store_true", help="random labels on 32/8 clips (smoke test)")
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
    set_seed(a.seed)

    if a.dummy:
        train_ds, val_ds, vocab = dummy_datasets(velocity, a.seed)
    else:
        train_ds, vocab = make_dataset("train", velocity, True, a.ctc_dir, a.seed)
        val_ds, _ = make_dataset("val", velocity, False, a.ctc_dir, a.seed)
    in_dim, n_classes = feature_dim(velocity), len(vocab)
    g = torch.Generator()
    g.manual_seed(a.seed)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, collate_fn=collate, generator=g)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, collate_fn=collate)

    model = build_model(a.model, in_dim, n_classes).to(device)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total_steps = a.epochs * math.ceil(len(train_ds) / a.batch)

    def lr_lambda(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        prog = min(1.0, (step - a.warmup) / max(1, total_steps - a.warmup))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    start_epoch, best_wer, best_loss, best_epoch, bad = 1, float("inf"), float("inf"), 0, 0

    if a.resume and last_p.exists():
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_wer, best_loss, best_epoch, bad = ck["best_wer"], ck["best_loss"], ck["best_epoch"], ck["bad"]
        torch.set_rng_state(ck["rng_torch"])
        np.random.set_state(ck["rng_numpy"])
        random.setstate(ck["rng_python"])
        train_ds.rng.setstate(ck["rng_dataset"])
        g.set_state(ck["rng_loader"])
        if ck.get("stopped"):
            print(f"[{name}] run had already stopped early at epoch {ck['epoch']}.")
            start_epoch = a.epochs + 1
        print(f"[{name}] resumed from epoch {ck['epoch']} (best WER {best_wer:.4f} at epoch {best_epoch})")
    elif log_p.exists():
        log_p.unlink()

    cfg = {"run": name, "model": a.model, "velocity": velocity, "in_dim": in_dim, "n_classes": n_classes,
           "seed": a.seed, "epochs": a.epochs, "patience": a.patience, "batch": a.batch, "lr": a.lr, "wd": a.wd,
           "warmup": a.warmup, "dummy": a.dummy, "n_train": len(train_ds), "n_val": len(val_ds),
           "device": str(device), "torch": torch.__version__, "vocab": vocab}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[{name}] device {device}; train {len(train_ds)} / val {len(val_ds)} clips; F={in_dim}; classes={n_classes}; "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    stopped = False
    epoch = start_epoch - 1
    for epoch in range(start_epoch, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for b in train_dl:
            lp, _, ol = model(b["x"].to(device), b["input_lengths"].to(device))
            loss = crit(lp, b["targets"].to(device), ol, b["target_lengths"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        val_loss, w = evaluate(model, val_dl, device, crit)
        improved = (w["wer"] < best_wer - 1e-9) or (abs(w["wer"] - best_wer) <= 1e-9 and val_loss < best_loss)
        if improved:
            best_wer, best_loss, best_epoch, bad = w["wer"], val_loss, epoch, 0
            save_atomic({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_wer": w["wer"],
                         "val_loss": val_loss}, best_p)
        else:
            bad += 1
        stopped = bad >= a.patience
        row = {"epoch": epoch, "train_loss": round(tot / max(1, nb), 5), "val_loss": round(val_loss, 5),
               "val_wer": round(w["wer"], 5), "val_S": w["S"], "val_D": w["D"], "val_I": w["I"], "val_N": w["N"],
               "lr": f"{sched.get_last_lr()[0]:.3e}", "seconds": round(time.time() - t0, 1), "improved": int(improved)}
        new = not log_p.exists()
        with open(log_p, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(row)
        save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "epoch": epoch, "best_wer": best_wer, "best_loss": best_loss, "best_epoch": best_epoch,
                     "bad": bad, "stopped": stopped, "config": cfg,
                     "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
                     "rng_python": random.getstate(), "rng_dataset": train_ds.rng.getstate(),
                     "rng_loader": g.get_state()}, last_p)
        print(f"[{name}] epoch {epoch:3d} train_loss {row['train_loss']:.4f} val_loss {row['val_loss']:.4f} "
              f"val_WER {row['val_wer']:.4f} (S{w['S']} D{w['D']} I{w['I']} / N{w['N']}) "
              f"{'*' if improved else ' '} {row['seconds']}s")
        if stopped:
            print(f"[{name}] early stopping: no improvement for {a.patience} epochs")
            break

    final = {"run": name, "best_val_wer": best_wer, "best_epoch": best_epoch, "last_epoch": epoch,
             "stopped_early": stopped, "finished": time.strftime("%Y-%m-%d %H:%M")}
    final_p.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[{name}] done: best val WER {best_wer:.4f} at epoch {best_epoch}; files in {out}")


if __name__ == "__main__":
    main()
