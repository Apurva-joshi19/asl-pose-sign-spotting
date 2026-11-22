"""Evaluate weak-label sign-spotting runs (PLAN_v2).

For every run folder under --runs_dir that contains best.pt from scripts/train_weak.py (other checkpoints are skipped):
  val + test scores -> thresholds chosen on VAL (global micro-F1, per-class F1) -> test metrics
  (mAP, micro-AP, micro/macro F1, micro P/R). Grouped by configuration: mean +- std over seeds.
Also: test baselines (class prior, random), per-class AP table, per-clip predictions, 10 examples,
error analysis (top false positives / negatives), figures.

Usage: python scripts/evaluate_weak.py --runs_dir checkpoints --out_dir reports
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml_metrics import average_precisions, baselines, choose_thresholds, f1_scores  # noqa: E402
from src.spotter import Spotter  # noqa: E402
from src.weak_dataset import WeakSpotDataset, collate_weak  # noqa: E402


@torch.no_grad()
def predict(model, ds, device, batch=64):
    model.eval()
    order = np.argsort(ds.ends - ds.starts)
    scores = np.zeros((len(ds), len(ds.classes)), np.float32)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch].tolist()
        b = collate_weak([ds[i] for i in idx])
        logits, _, _ = model(b["x"].to(device), b["lengths"].to(device))
        scores[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return scores


def config_key(cfg):
    return f"{cfg['model']}_{'full' if cfg['velocity'] else 'novel'}"


def fmt(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return "n/a"
    return f"{np.mean(vals):.4f} +- {np.std(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--out_dir", default=str(ROOT / "reports"))
    ap.add_argument("--weak_dir", default=str(ROOT / "weak"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    runs_dir, out, weak = Path(a.runs_dir), Path(a.out_dir), Path(a.weak_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          (a.device if a.device != "auto" else "cpu"))
    candidates = sorted(p.parent for p in runs_dir.glob("*/best.pt"))
    if not candidates:
        sys.exit(f"no */best.pt under {runs_dir}")
    print(f"{len(candidates)} checkpoint folder(s) found on {device}")

    wl = pd.read_csv(ROOT / "annotations" / "weak_labels.csv", dtype=str, keep_default_na=False).set_index("SENTENCE_NAME")
    ds_cache = {}
    results, per_class = {}, defaultdict(list)
    for run in candidates:
        ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
        cfg = ck.get("config", {})
        if "weak" not in str(cfg.get("task", "")):
            print(f"[{run.name}] skipped: not a PLAN_v2 sign-spotting checkpoint")
            continue
        vel = cfg["velocity"]
        if vel not in ds_cache:
            ds_cache[vel] = {s: WeakSpotDataset(s, vel, False, weak) for s in ("val", "test")}
        dv, dt = ds_cache[vel]["val"], ds_cache[vel]["test"]
        classes = dt.classes
        model = Spotter(cfg["model"], cfg["in_dim"], cfg["n_classes"]).to(device)
        model.load_state_dict(ck["model"])
        sv, st = predict(model, dv, device), predict(model, dt, device)
        g_thr, c_thr = choose_thresholds(dv.y, sv)
        m = average_precisions(dt.y, st)
        fg, fc = f1_scores(dt.y, st, g_thr), f1_scores(dt.y, st, c_thr)
        r = {"run": run.name, "config": config_key(cfg), "seed": cfg["seed"], "best_epoch": ck["epoch"],
             "val_map": float(ck["val_map"]), "test_map": m["map"], "test_micro_ap": m["micro_ap"],
             "global_threshold": g_thr, "test_micro_f1_global": fg["micro_f1"], "test_macro_f1_global": fg["macro_f1"],
             "test_micro_p_global": fg["micro_precision"], "test_micro_r_global": fg["micro_recall"],
             "test_micro_f1_perclass": fc["micro_f1"], "test_macro_f1_perclass": fc["macro_f1"]}
        results[run.name] = r
        per_class[r["config"]].append(m["per_class_ap"])
        print(f"[{run.name}] best epoch {r['best_epoch']}; val mAP {r['val_map']:.4f}; test mAP {r['test_map']:.4f}; "
              f"micro-AP {r['test_micro_ap']:.4f}; micro-F1 {r['test_micro_f1_global']:.4f} (thr {g_thr})")

        pred = st >= g_thr
        with open(out / f"test_predictions_{run.name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["SENTENCE_NAME", "true", "predicted", "top5_scores"])
            for i, n in enumerate(dt.names):
                top = np.argsort(-st[i])[:5]
                w.writerow([n, " ".join(c for c, t in zip(classes, dt.y[i]) if t),
                            " ".join(c for c, p in zip(classes, pred[i]) if p),
                            " ".join(f"{classes[j]}:{st[i, j]:.2f}" for j in top)])
        r["_scores"], r["_y"], r["_names"], r["_classes"] = st, dt.y, dt.names, classes

    if not results:
        sys.exit("no PLAN_v2 sign-spotting runs found")

    # summary per configuration
    groups = defaultdict(list)
    for r in results.values():
        groups[r["config"]].append(r)
    keys = ["val_map", "test_map", "test_micro_ap", "test_micro_f1_global", "test_macro_f1_global",
            "test_micro_p_global", "test_micro_r_global", "test_micro_f1_perclass", "test_macro_f1_perclass"]
    summary = []
    for c, rs in sorted(groups.items()):
        summary.append({"config": c, "n_seeds": len(rs), **{k: fmt([x[k] for x in rs]) for k in keys}})
    any_r = next(iter(results.values()))
    tr = WeakSpotDataset("train", True, False, weak)
    prior = tr.y.mean(0)
    b = baselines(any_r["_y"], prior)
    for bn in ("class_prior", "random"):
        summary.append({"config": f"baseline_{bn}", "n_seeds": 1, "test_map": f"{b[bn]['map']:.4f}",
                        "test_micro_ap": f"{b[bn]['micro_ap']:.4f}"})
    pd.DataFrame(summary).to_csv(out / "results_weak_summary.csv", index=False, encoding="utf-8")

    # per-class AP table
    classes = any_r["_classes"]
    pc = pd.DataFrame({"class": classes, "train_positives": tr.y.sum(0).astype(int),
                       "test_positives": any_r["_y"].sum(0).astype(int)})
    for c, aps in per_class.items():
        pc[f"ap_{c}"] = np.nanmean(np.stack(aps), axis=0)
    pc["ap_class_prior"] = b["class_prior"]["per_class_ap"]
    best_cfg = max(groups, key=lambda c: np.mean([x["test_map"] for x in groups[c]]))
    pc = pc.sort_values(f"ap_{best_cfg}", ascending=False)
    pc.to_csv(out / "per_class_ap.csv", index=False, encoding="utf-8")

    # examples + error analysis on the best run of the best configuration
    best_run = max(groups[best_cfg], key=lambda x: x["test_map"])
    st, y, names = best_run["_scores"], best_run["_y"], best_run["_names"]
    pred = st >= best_run["global_threshold"]
    tp, fp, fn = (pred & (y > 0)).sum(1), (pred & (y == 0)).sum(1), (~pred & (y > 0)).sum(1)
    clip_f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(1, 2 * tp + fp + fn), np.nan)
    has = y.sum(1) > 0
    order = np.argsort(-np.nan_to_num(clip_f1, nan=-1))
    good = [i for i in order if has[i]][:5]
    bad = [i for i in order[::-1] if has[i]][:5]
    lines = [f"# Test examples - {best_run['run']} (threshold {best_run['global_threshold']})", "",
             "Labels are translation-derived weak labels (from the English sentence), not verified glosses.", "",
             "| type | clip | English | true | predicted | clip F1 |", "|---|---|---|---|---|---|"]
    for kind, sel in (("good", good), ("bad", bad)):
        for i in sel:
            n = names[i]
            lines.append(f"| {kind} | {n} | {wl.at[n, 'SENTENCE'] if n in wl.index else ''} | "
                         f"{' '.join(c for c, t in zip(classes, y[i]) if t)} | "
                         f"{' '.join(c for c, p in zip(classes, pred[i]) if p) or '(none)'} | {clip_f1[i]:.2f} |")
    fp_c = (pred & (y == 0)).sum(0)
    fn_c = (~pred & (y > 0)).sum(0)
    lines += ["", "## Error analysis", "",
              "Most frequent false positives (predicted, not in weak label): " +
              ", ".join(f"{classes[j]} ({fp_c[j]})" for j in np.argsort(-fp_c)[:8]),
              "", "Most frequent false negatives (in weak label, not predicted): " +
              ", ".join(f"{classes[j]} ({fn_c[j]})" for j in np.argsort(-fn_c)[:8]),
              "", "Note: a 'false positive' may be a sign that is really present but not mentioned in the English,"
              " and a 'false negative' may be an English word that was not signed - the weak labels are noisy."]
    (out / "examples_weak.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # figures
    for metric, fname, ylab in (("val_map", "val_map_curves.png", "val mAP"), (None, "loss_curves.png", "loss")):
        plt.figure(figsize=(8, 4.5))
        for r in results.values():
            lp = runs_dir / r["run"] / "log.csv"
            if not lp.exists():
                continue
            lg = pd.read_csv(lp)
            if metric:
                plt.plot(lg["epoch"], lg[metric], label=r["run"])
            else:
                plt.plot(lg["epoch"], lg["train_loss"], label=f"{r['run']} train")
                plt.plot(lg["epoch"], lg["val_loss"], "--", label=f"{r['run']} val")
        plt.xlabel("epoch")
        plt.ylabel(ylab)
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=120)
        plt.close()
    plt.figure(figsize=(7, 4.5))
    plt.scatter(pc["train_positives"], pc[f"ap_{best_cfg}"], s=14, label=best_cfg)
    plt.scatter(pc["train_positives"], pc["ap_class_prior"], s=10, marker="x", label="class prior")
    plt.xscale("log")
    plt.xlabel("train positives (log)")
    plt.ylabel("test AP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "per_class_ap_vs_frequency.png", dpi=120)
    plt.close()

    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    (out / "results_weak.json").write_text(json.dumps({"runs": clean, "summary": summary, "best_config": best_cfg},
                                                      indent=2, default=float), encoding="utf-8")
    print("\nSUMMARY (test, thresholds from val):")
    print(pd.DataFrame(summary)[["config", "n_seeds", "test_map", "test_micro_ap", "test_micro_f1_global",
                                 "test_macro_f1_global"]].fillna("").to_string(index=False))
    print(f"\nbest configuration: {best_cfg}; files written to {out}")


if __name__ == "__main__":
    main()
