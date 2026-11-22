"""Pre-training checks (plan section 10). Checks 1-6 run locally on CPU; check 7 (--overfit) is meant for Colab.

  1. every label gloss is in vocab; vocab ids 0..n-1 with <blank>=0; clip lists match label files
  2. keypoint npy T within +-2 of the RGB frame count (annotations/clip_meta.csv)
  3. CTC length rule (15 fps version): ceil(T15/2) >= len(label) + adjacent repeats
  4. no NaN/Inf in standardised features (velocity and no-velocity; augmented train features)
  5. forward-pass shapes correct for TransformerCTC and BiGRUCTC (F=206 and F=104)
  6. CTC loss finite for every clip (both models)
  7. --overfit: TransformerCTC on <=16 train clips, <=300 steps: final loss < 30% of initial and
     greedy decoding exactly matches >= 80% of label sequences

Usage:
  python tests/run_checks.py                          # checks 1-6 on ctc/
  python tests/run_checks.py --overfit --device auto  # checks 1-7
  python tests/run_checks.py --ctc_dir ctc_pilot_test --overfit
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import PoseCTCDataset, collate, ctc_min_len, select_stats  # noqa: E402
from src.features import KP_DIR, TARGET_FPS, feature_dim, load_meta, load_stats  # noqa: E402
from src.metrics import greedy_decode  # noqa: E402
from src.models import build_model  # noqa: E402

SPLITS = ["train", "val", "test"]
RESULTS = []


def report(num, name, ok, detail):
    RESULTS.append((num, name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] check {num}: {name} - {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctc_dir", default=str(ROOT / "ctc"))
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--overfit-clips", type=int, default=16)
    a = ap.parse_args()
    ctc = Path(a.ctc_dir)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          (a.device if a.device != "auto" else "cpu"))
    print(f"ctc_dir: {ctc}  device: {device}")

    vocab = json.loads((ctc / "vocab.json").read_text(encoding="utf-8"))
    labels, names = {}, {}
    for s in SPLITS:
        labels[s] = json.loads((ctc / f"labels_{s}.json").read_text(encoding="utf-8"))
        names[s] = [l.strip() for l in (ctc / f"clips_{s}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    splits = [s for s in SPLITS if names[s]]
    for s in SPLITS:
        if s not in splits:
            print(f"(note: split '{s}' has no clips - skipped)")
    all_names = [n for s in splits for n in names[s]]
    all_labels = {n: labels[s][n] for s in splits for n in names[s] if n in labels[s]}

    # 1
    unknown = [(n, g) for n, seq in all_labels.items() for g in seq if g not in vocab]
    ids_ok = vocab.get("<blank>") == 0 and sorted(vocab.values()) == list(range(len(vocab)))
    lists_ok = all(set(names[s]) == set(labels[s]) for s in SPLITS)
    report(1, "labels in vocab", not unknown and ids_ok and lists_ok,
           f"{len(all_labels)} clips, vocab {len(vocab)}; unknown glosses {len(unknown)}; ids contiguous {ids_ok}; "
           f"clip lists match labels {lists_ok}" + (f"; e.g. {unknown[:3]}" if unknown else ""))
    targets = {n: [vocab[g] for g in seq if g in vocab] for n, seq in all_labels.items()}

    # 2
    meta = pd.read_csv(ROOT / "annotations" / "clip_meta.csv", dtype=str, keep_default_na=False).set_index("SENTENCE_NAME")
    T = {n: np.load(KP_DIR / f"{n}.npy", mmap_mode="r").shape[0] for n in all_names}
    off = [(n, T[n], int(meta.at[n, "n_frames"])) for n in all_names if abs(T[n] - int(meta.at[n, "n_frames"])) > 2]
    report(2, "npy T vs RGB frames (+-2)", not off, f"{len(all_names) - len(off)}/{len(all_names)} clips OK"
           + (f"; e.g. {off[:3]}" if off else ""))

    # 3
    fps_map = load_meta()
    viol = []
    for n in all_names:
        t15 = max(1, int(round(T[n] * TARGET_FPS / fps_map[n])))
        if math.ceil(t15 / 2) < ctc_min_len(targets[n]):
            viol.append((n, math.ceil(t15 / 2), ctc_min_len(targets[n])))
    report(3, "CTC length rule ceil(T15/2) >= len + repeats", not viol,
           f"{len(all_names) - len(viol)}/{len(all_names)} clips OK" + (f"; e.g. {viol[:3]}" if viol else ""))

    # 4
    stats = load_stats(ctc / "feature_stats.npz")
    bad = []
    for vel in (True, False):
        ds = PoseCTCDataset(all_names, targets, fps_map, vel, select_stats(stats, vel), augment=False)
        bad += [(n, vel) for i, n in enumerate(all_names) if not torch.isfinite(ds[i][0]).all()]
    if names["train"]:
        ds_aug = PoseCTCDataset(names["train"], targets, fps_map, True, stats, augment=True, seed=0)
        bad += [(n, "aug") for i, n in enumerate(names["train"]) if not torch.isfinite(ds_aug[i][0]).all()]
    report(4, "no NaN/Inf in features", not bad, f"{2 * len(all_names) + len(names['train'])} feature arrays checked"
           + (f"; bad e.g. {bad[:3]}" if bad else ""))

    # 5
    batch_names = (names["train"] or all_names)[:8]
    shape_ok, details = True, []
    for vel in (True, False):
        ds = PoseCTCDataset(batch_names, targets, fps_map, vel, select_stats(stats, vel), augment=False)
        b = collate([ds[i] for i in range(len(ds))])
        for mname in ("transformer", "bigru"):
            m = build_model(mname, feature_dim(vel), len(vocab)).eval()
            with torch.no_grad():
                lp, enc, ol = m(b["x"], b["input_lengths"])
            Tp = math.ceil(int(b["input_lengths"].max()) / 2)
            ok = (tuple(lp.shape) == (Tp, len(batch_names), len(vocab)) and tuple(enc.shape) == (len(batch_names), Tp, m.enc_dim)
                  and ol.tolist() == [math.ceil(int(L) / 2) for L in b["input_lengths"]])
            shape_ok &= ok
            details.append(f"{mname}/F{feature_dim(vel)} {'ok' if ok else 'BAD'}")
    report(5, "forward-pass shapes", shape_ok, ", ".join(details))

    # 6
    crit = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
    nonfinite = []
    ds = PoseCTCDataset(all_names, targets, fps_map, True, stats, augment=False)
    for mname in ("transformer", "bigru"):
        torch.manual_seed(0)
        m = build_model(mname, feature_dim(True), len(vocab)).eval()
        for k in range(0, len(all_names), 16):
            idx = list(range(k, min(k + 16, len(all_names))))
            b = collate([ds[i] for i in idx])
            with torch.no_grad():
                lp, _, ol = m(b["x"], b["input_lengths"])
                l = crit(lp, b["targets"], ol, b["target_lengths"])
            nonfinite += [(mname, all_names[i]) for i, v in zip(idx, l.tolist()) if not math.isfinite(v)]
    report(6, "CTC loss finite for every clip", not nonfinite,
           f"{2 * len(all_names)} clip losses computed" + (f"; non-finite e.g. {nonfinite[:3]}" if nonfinite else ""))

    # 7
    if a.overfit:
        sel = names["train"][: a.overfit_clips]
        if not sel:
            report(7, "overfit", False, "no train clips")
        else:
            t0 = time.time()
            torch.manual_seed(0)
            ds = PoseCTCDataset(sel, targets, fps_map, True, stats, augment=False)
            b = collate([ds[i] for i in range(len(ds))])
            x, L = b["x"].to(device), b["input_lengths"].to(device)
            tg, tl = b["targets"].to(device), b["target_lengths"].to(device)
            m = build_model("transformer", feature_dim(True), len(vocab)).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
            ocrit = nn.CTCLoss(blank=0, zero_infinity=True)
            losses = []
            for _ in range(a.overfit_steps):
                m.train()
                lp, _, ol = m(x, L)
                loss = ocrit(lp, tg, ol, tl)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            m.eval()
            with torch.no_grad():
                lp, _, ol = m(x, L)
            hyps = greedy_decode(lp, ol)
            refs = [targets[n] for n in sel]
            exact = sum(h == r for h, r in zip(hyps, refs))
            first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
            ok = last < 0.3 * first and exact >= math.ceil(0.8 * len(sel))
            report(7, "overfit (TransformerCTC, full batch, lr 1e-3)", ok,
                   f"{len(sel)} clips, {a.overfit_steps} steps: loss {first:.3f} -> {last:.4f}; "
                   f"exact match {exact}/{len(sel)} (need >= 80%); {time.time() - t0:.0f}s on {device}")

    n_fail = sum(not ok for _, _, ok in RESULTS)
    print("\nALL CHECKS PASSED" if n_fail == 0 else f"\n{n_fail} CHECK(S) FAILED")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
