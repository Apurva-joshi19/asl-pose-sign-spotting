"""Multi-label metrics for sign spotting (PLAN_v2).

average_precisions: per-class AP (classes without positives -> NaN, excluded from mAP), macro mAP, micro AP
choose_thresholds:  global threshold maximising micro-F1 on VAL, and per-class thresholds maximising per-class F1
f1_scores:          micro / macro F1 at given thresholds
baselines:          class-prior scores (train frequency) and random scores
per_class_table:    AP, F1, positives per class

Self-test:  python -m src.ml_metrics --check
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score

GRID = np.round(np.arange(0.02, 0.981, 0.02), 2)


def average_precisions(y, s):
    y, s = np.asarray(y), np.asarray(s)
    ap = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        if y[:, c].sum() > 0:
            ap[c] = average_precision_score(y[:, c], s[:, c])
    micro = average_precision_score(y.ravel(), s.ravel()) if y.sum() > 0 else float("nan")
    return {"per_class_ap": ap, "map": float(np.nanmean(ap)), "micro_ap": float(micro),
            "n_classes_with_pos": int((~np.isnan(ap)).sum())}


def f1_scores(y, s, thr):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s) >= np.asarray(thr)
    tp = (pred & y).sum(0).astype(float)
    fp = (pred & ~y).sum(0).astype(float)
    fn = (~pred & y).sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    micro = 2 * tp.sum() / max(1.0, 2 * tp.sum() + fp.sum() + fn.sum())
    prec = tp.sum() / max(1.0, tp.sum() + fp.sum())
    rec = tp.sum() / max(1.0, tp.sum() + fn.sum())
    return {"micro_f1": float(micro), "macro_f1": float(np.nanmean(per)), "micro_precision": float(prec),
            "micro_recall": float(rec), "per_class_f1": per}


def choose_thresholds(y_val, s_val):
    best_g = max(GRID, key=lambda t: f1_scores(y_val, s_val, t)["micro_f1"])
    per = np.full(np.asarray(y_val).shape[1], best_g)
    for c in range(len(per)):
        yc, sc = np.asarray(y_val)[:, [c]], np.asarray(s_val)[:, [c]]
        if yc.sum() > 0:
            per[c] = max(GRID, key=lambda t: np.nan_to_num(f1_scores(yc, sc, t)["per_class_f1"][0]))
    return float(best_g), per


def baselines(y_eval, train_prior, seed=12345):
    y_eval = np.asarray(y_eval)
    prior = np.tile(np.asarray(train_prior)[None, :], (len(y_eval), 1))
    rnd = np.random.RandomState(seed).rand(*y_eval.shape)
    return {"class_prior": average_precisions(y_eval, prior), "random": average_precisions(y_eval, rnd)}


def per_class_table(classes, y, s, thr):
    ap = average_precisions(y, s)["per_class_ap"]
    f1 = f1_scores(y, s, thr)["per_class_f1"]
    pos = np.asarray(y).sum(0).astype(int)
    rows = [{"class": c, "positives": int(pos[i]), "ap": float(ap[i]), "f1": float(f1[i]),
             "threshold": float(np.broadcast_to(thr, (len(classes),))[i])} for i, c in enumerate(classes)]
    return sorted(rows, key=lambda r: -np.nan_to_num(r["ap"], nan=-1))


def _check():
    ok = True

    def expect(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name}: {detail} -> {'OK' if cond else 'FAIL'}")

    rs = np.random.RandomState(0)
    y = (rs.rand(2000, 6) < np.array([0.3, 0.1, 0.05, 0.2, 0.02, 0.0])).astype(int)
    perfect = average_precisions(y, y + 0.01 * rs.rand(*y.shape))
    expect("perfect scores", abs(perfect["map"] - 1.0) < 1e-9 and perfect["n_classes_with_pos"] == 5,
           f"mAP {perfect['map']:.3f}, classes with positives {perfect['n_classes_with_pos']} (class 6 has none)")
    rate = y[:, :5].mean(0)
    b = baselines(y, train_prior=np.r_[rate, 0.01], seed=1)
    expect("class-prior baseline AP = positive rate", np.allclose(b["class_prior"]["per_class_ap"][:5], rate),
           f"AP {np.round(b['class_prior']['per_class_ap'][:5], 3).tolist()} vs rate {np.round(rate, 3).tolist()}")
    expect("random baseline AP ~ positive rate", np.all(np.abs(b["random"]["per_class_ap"][:5] - rate) < 0.05),
           f"AP {np.round(b['random']['per_class_ap'][:5], 3).tolist()}")
    s = np.clip(y * 0.7 + rs.rand(*y.shape) * 0.5, 0, 1)
    g, per = choose_thresholds(y, s)
    f = f1_scores(y, s, g)
    expect("threshold choice on separable-ish scores", 0.2 <= g <= 0.8 and f["micro_f1"] > 0.9,
           f"global threshold {g}, micro-F1 {f['micro_f1']:.3f}, per-class thresholds {per.round(2).tolist()}")
    t = per_class_table([f"c{i}" for i in range(6)], y, s, per)
    expect("per-class table", len(t) == 6 and t[-1]["class"] == "c5", f"first {t[0]['class']} AP {t[0]['ap']:.3f}, last {t[-1]['class']}")
    print("ALL ML METRIC CHECKS PASSED" if ok else "SOME ML METRIC CHECKS FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        _check()
    else:
        print(__doc__)
