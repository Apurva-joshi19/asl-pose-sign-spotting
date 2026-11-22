"""Export per-clip encoder features + logits for fusion (handover v1 contract).

For each run: outputs/features/<run_name>/<SENTENCE_NAME>.npz with
  enc    (T', d) float16  encoder features at 7.5 steps/s (T' = ceil(T15/2)); convert with .astype(np.float32)
  logits (C,)    float32  raw per-class scores (apply sigmoid for probabilities)
  fps    7.5,  split  "train" | "val" | "test"
plus manifest.json, and outputs/fusion_<run_name>.zip.
Runs are chosen by best VAL mAP per configuration (never test).

Usage: python scripts/export_features.py [--runs bigru_full_seed3 transformer_full_seed1]
"""
import argparse
import json
import math
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spotter import Spotter  # noqa: E402
from src.weak_dataset import WeakSpotDataset, collate_weak  # noqa: E402

SPLITS = ["train", "val", "test"]


@torch.no_grad()
def export_run(run_dir, out_root, weak, device, batch):
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    if "weak" not in str(cfg.get("task", "")):
        raise SystemExit(f"{run_dir.name} is not a PLAN_v2 sign-spotting run")
    model = Spotter(cfg["model"], cfg["in_dim"], cfg["n_classes"]).to(device).eval()
    model.load_state_dict(ck["model"])
    out = out_root / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    counts, checks = {}, []
    t0 = time.time()
    for split in SPLITS:
        ds = WeakSpotDataset(split, cfg["velocity"], False, weak)
        order = np.argsort(ds.ends - ds.starts)
        for k in range(0, len(order), batch):
            idx = order[k:k + batch].tolist()
            b = collate_weak([ds[i] for i in idx])
            logits, enc, ol = model(b["x"].to(device), b["lengths"].to(device))
            enc, logits, ol = enc.float().cpu().numpy(), logits.float().cpu().numpy(), ol.cpu().numpy()
            for j, i in enumerate(idx):
                np.savez(out / f"{ds.names[i]}.npz", enc=enc[j, :ol[j]].astype(np.float16),
                         logits=logits[j].astype(np.float32), fps=np.float32(7.5), split=split)
        counts[split] = len(ds)
        for i in random.Random(0).sample(range(len(ds)), 3):
            z = np.load(out / f"{ds.names[i]}.npz")
            t15 = int(ds.ends[i] - ds.starts[i])
            checks.append(z["enc"].shape == (math.ceil(t15 / 2), model.enc_dim) and z["logits"].shape == (cfg["n_classes"],))
        print(f"  [{run_dir.name}] {split}: {len(ds):,} clips exported ({time.time() - t0:.0f}s)", flush=True)
    manifest = {"run": run_dir.name, "model": cfg["model"], "velocity": cfg["velocity"], "seed": cfg["seed"],
                "best_epoch": ck["epoch"], "val_map": float(ck["val_map"]), "enc_dim": model.enc_dim,
                "n_classes": cfg["n_classes"], "classes": cfg["classes"], "clips": counts,
                "enc_dtype": "float16 (convert with .astype(np.float32))", "logits": "raw scores, apply sigmoid",
                "steps_per_second": 7.5, "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zp = out_root.parent / f"fusion_{run_dir.name}.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in sorted(out.iterdir()):
            z.write(f, f"{run_dir.name}/{f.name}")
    return counts, all(checks), zp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["bigru_full_seed3", "transformer_full_seed1"])
    ap.add_argument("--runs_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--out_root", default=str(ROOT / "outputs" / "features"))
    ap.add_argument("--weak_dir", default=str(ROOT / "weak"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          (a.device if a.device != "auto" else "cpu"))
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"device {device}; runs: {', '.join(a.runs)}")
    for r in a.runs:
        counts, ok, zp = export_run(Path(a.runs_dir) / r, out_root, Path(a.weak_dir), device, a.batch)
        print(f"[{r}] clips {counts}; shape check {'OK' if ok else 'FAIL'}; zip {zp.name} {zp.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
