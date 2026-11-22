"""Inspect How2Sign raw archives WITHOUT extracting them, and write data_inspection.md.

Usage:
  python scripts/inspect_data.py                  # val + test + train (train keypoint scan is slow the first time)
  python scripts/inspect_data.py --skip-train-kp  # skip scanning the 21 GB train keypoint archive

Keypoint archive indexes are cached in data/raw/kp_<split>/index_cache.json, so later runs are fast.
"""
import argparse
import json
import random
import shutil
import statistics
import sys
import tarfile
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIG = ROOT / "reports" / "figures"
OUT_MD = ROOT / "data_inspection.md"
SPLITS = ["train", "val", "test"]
EXPECTED_COLS = ["VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME", "START", "END", "SENTENCE"]
HAND_MISSING_CONF = 0.2
SEED = 42
N_VIDEO_CHECK = 20
N_HAND_SAMPLE = 200
GIB = 1024 ** 3


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def load_manifest():
    p = RAW / "manifest.json"
    if not p.exists():
        sys.exit("data/raw/manifest.json not found - run download_data.py status first.")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def archive_path(manifest, key):
    rec = manifest.get(key, {})
    if rec.get("status") != "ok":
        return None
    p = RAW / key / rec["file"]
    return p if p.is_file() else None


def read_csv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=",", dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df


def rgb_index(path):
    """Read the zip central directory (fast). Returns {sentence_name: member_name} and info."""
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
    exts = Counter(Path(n).suffix.lower() for n in names)
    prefixes = Counter("/".join(n.split("/")[:-1]) or "(root)" for n in names)
    mp4 = {Path(n).stem: n for n in names if n.lower().endswith(".mp4")}
    return mp4, {"members": len(names), "exts": dict(exts), "prefixes": dict(prefixes.most_common(5))}


def frame_hand_stats(data):
    """Pick the person with highest mean pose confidence; return per-frame flags and keypoint counts."""
    people = data.get("people", []) or []
    out = {"no_person": len(people) == 0, "multi": len(people) > 1, "lh_missing": True, "rh_missing": True, "shape": None}
    if not people:
        return out

    def mean_conf(arr):
        c = arr[2::3]
        return float(np.mean(c)) if len(c) else 0.0

    best = max(people, key=lambda p: mean_conf(p.get("pose_keypoints_2d", []) or []))
    lh = best.get("hand_left_keypoints_2d", []) or []
    rh = best.get("hand_right_keypoints_2d", []) or []
    out["lh_missing"] = mean_conf(lh) < HAND_MISSING_CONF
    out["rh_missing"] = mean_conf(rh) < HAND_MISSING_CONF
    out["shape"] = (
        len(best.get("pose_keypoints_2d", []) or []) // 3,
        len(best.get("face_keypoints_2d", []) or []) // 3,
        len(lh) // 3,
        len(rh) // 3,
    )
    return out


def kp_index(path, split, sample_names):
    """Stream the tar.gz once: count JSON frames per sentence folder; read JSON for sampled folders."""
    cache_p = RAW / f"kp_{split}" / "index_cache.json"
    size = path.stat().st_size
    if cache_p.exists():
        cache = json.loads(cache_p.read_text(encoding="utf-8"))
        if cache.get("archive_bytes") == size and set(sample_names) <= set(cache.get("sample", [])):
            print(f"[kp {split}] using cached index ({len(cache['index']):,} folders)")
            return cache

    sample_set = set(sample_names)
    index = Counter()
    prefixes = Counter()
    hand = defaultdict(lambda: Counter())
    shapes = Counter()
    members = 0
    t0 = time.time()
    print(f"[kp {split}] scanning {path.name} ({size / GIB:.2f} GiB) ...", flush=True)
    with tarfile.open(path, mode="r|gz") as tf:
        for m in tf:
            members += 1
            if members % 100000 == 0:
                print(f"  [kp {split}] {members:,} members, {len(index):,} folders, {time.time() - t0:.0f}s", flush=True)
            if not m.isfile():
                continue
            parts = m.name.split("/")
            prefixes["/".join(parts[:-2]) or "(root)"] += 1
            if not m.name.endswith("_keypoints.json") or len(parts) < 2:
                continue
            folder = parts[-2]
            index[folder] += 1
            if folder in sample_set:
                f = tf.extractfile(m)
                try:
                    st = frame_hand_stats(json.load(f))
                except Exception:
                    hand[folder]["bad_json"] += 1
                    continue
                h = hand[folder]
                h["frames"] += 1
                h["lh_missing"] += st["lh_missing"]
                h["rh_missing"] += st["rh_missing"]
                h["both_missing"] += st["lh_missing"] and st["rh_missing"]
                h["no_person"] += st["no_person"]
                h["multi"] += st["multi"]
                if st["shape"]:
                    shapes[str(st["shape"])] += 1

    cache = {
        "archive_bytes": size,
        "members": members,
        "seconds": round(time.time() - t0),
        "index": dict(index),
        "prefixes": dict(prefixes.most_common(5)),
        "sample": sorted(sample_set),
        "hand": {k: dict(v) for k, v in hand.items()},
        "shapes": dict(shapes),
    }
    cache_p.write_text(json.dumps(cache), encoding="utf-8")
    print(f"[kp {split}] done: {members:,} members, {len(index):,} folders in {cache['seconds']}s", flush=True)
    return cache


def video_check(rgb_zip, names, kp_idx, rgb_map):
    rng = random.Random(SEED)
    cand = sorted(set(names) & set(kp_idx) & set(rgb_map))
    pick = rng.sample(cand, min(N_VIDEO_CHECK, len(cand)))
    rows = []
    with zipfile.ZipFile(rgb_zip) as z, tempfile.TemporaryDirectory() as td:
        for s in pick:
            out = Path(td) / f"{s}.mp4"
            with z.open(rgb_map[s]) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            cap = cv2.VideoCapture(str(out))
            fps = cap.get(cv2.CAP_PROP_FPS)
            meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cnt = 0
            while cap.grab():
                cnt += 1
            cap.release()
            out.unlink(missing_ok=True)
            rows.append({"name": s, "fps": round(fps, 3), "size": f"{w}x{h}", "frames_meta": meta,
                         "frames_read": cnt, "json_frames": kp_idx[s], "diff": cnt - kp_idx[s]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train-kp", action="store_true", help="do not scan the 21 GB train keypoint archive")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    md = ["# How2Sign data inspection", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/inspect_data.py` (no full extraction).", ""]

    # 1. Archives
    md += ["## 1. Archives", "", "| key | file | size (GiB) | format |", "|---|---|---|---|"]
    for key, rec in manifest.items():
        if rec.get("status") != "ok":
            md.append(f"| {key} | MISSING | - | - |")
            continue
        name = rec["file"]
        fmt = "tar.gz" if name.endswith(".tar.gz") else Path(name).suffix.lstrip(".")
        md.append(f"| {key} | {name} | {rec['bytes'] / GIB:.2f} | {fmt} |")

    # 2. CSVs
    csvs = {}
    md += ["", "## 2. English translation CSVs", ""]
    for s in SPLITS:
        p = archive_path(manifest, f"csv_{s}")
        if p is None:
            sys.exit(f"csv_{s} missing in manifest")
        df = read_csv(p)
        if "SENTENCE_NAME" not in df.columns:
            sys.exit(f"{p.name}: no SENTENCE_NAME column. Columns are: {list(df.columns)}")
        csvs[s] = df
        miss = [c for c in EXPECTED_COLS if c not in df.columns]
        dup = int(df["SENTENCE_NAME"].duplicated().sum())
        md.append(f"- **{s}**: {len(df):,} rows, {df['VIDEO_ID'].nunique() if 'VIDEO_ID' in df else '?'} videos; "
                  f"columns: `{', '.join(df.columns)}`; "
                  + ("all expected columns present" if not miss else f"**MISSING expected columns: {miss}**")
                  + f"; duplicate SENTENCE_NAME: {dup}")
        print(f"[csv {s}] {len(df):,} rows")

    # 3. RGB + keypoint indexes
    rgb, rgb_info = {}, {}
    for s in SPLITS:
        p = archive_path(manifest, f"rgb_{s}")
        rgb[s], rgb_info[s] = rgb_index(p)
        print(f"[rgb {s}] {len(rgb[s]):,} mp4 files")

    val_names = sorted(csvs["val"]["SENTENCE_NAME"].unique())
    hand_sample = random.Random(SEED).sample(val_names, min(N_HAND_SAMPLE, len(val_names)))
    kp = {}
    for s in SPLITS:
        if s == "train" and args.skip_train_kp:
            kp[s] = None
            continue
        kp[s] = kp_index(archive_path(manifest, f"kp_{s}"), s, hand_sample if s == "val" else [])

    md += ["", "## 3. Archive layout", ""]
    for s in SPLITS:
        info = rgb_info[s]
        md.append(f"- **rgb_{s}**: {info['members']:,} files, extensions {info['exts']}, top folders {info['prefixes']}")
        if kp[s]:
            md.append(f"- **kp_{s}**: {kp[s]['members']:,} tar members, {len(kp[s]['index']):,} sentence folders, "
                      f"top folders {kp[s]['prefixes']} (scan took {kp[s]['seconds']}s)")
        else:
            md.append(f"- **kp_{s}**: not scanned (--skip-train-kp)")

    # 4. Match rates
    md += ["", "## 4. SENTENCE_NAME <-> keypoint folder <-> RGB file", "",
           "| split | CSV sentences | with keypoints | with RGB | with both | extra kp folders | extra RGB files |",
           "|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        C = set(csvs[s]["SENTENCE_NAME"])
        R = set(rgb[s])
        K = set(kp[s]["index"]) if kp[s] else None
        if K is None:
            md.append(f"| {s} | {len(C):,} | not scanned | {pct(len(C & R), len(C))} | n/a | n/a | {len(R - C):,} |")
        else:
            md.append(f"| {s} | {len(C):,} | {pct(len(C & K), len(C))} | {pct(len(C & R), len(C))} | "
                      f"{pct(len(C & K & R), len(C))} | {len(K - C):,} | {len(R - C):,} |")

    # 5. fps and frame counts (val)
    md += ["", f"## 5. fps and RGB frames vs keypoint JSON frames ({N_VIDEO_CHECK} val clips, tolerance +-2)", ""]
    rows = video_check(archive_path(manifest, "rgb_val"), val_names, kp["val"]["index"], rgb["val"])
    ok = sum(abs(r["diff"]) <= 2 for r in rows)
    fps_vals = Counter(r["fps"] for r in rows)
    md.append(f"- fps values seen: {dict(fps_vals)}; frame sizes: {dict(Counter(r['size'] for r in rows))}")
    md.append(f"- clips within +-2 frames: **{ok}/{len(rows)}**")
    md += ["", "| clip | fps | RGB frames (read) | RGB frames (metadata) | JSON frames | diff |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['fps']} | {r['frames_read']} | {r['frames_meta']} | {r['json_frames']} | {r['diff']} |")
    print(f"[video] {ok}/{len(rows)} clips within +-2 frames; fps {dict(fps_vals)}")

    # 6. Missing hands (val sample)
    hs = kp["val"]["hand"]
    tot = Counter()
    clips_rh_mostly_missing = 0
    for name, h in hs.items():
        tot.update(h)
        if h.get("frames") and h.get("rh_missing", 0) / h["frames"] > 0.5:
            clips_rh_mostly_missing += 1
    F = tot["frames"]
    md += ["", f"## 6. Missing hands ({len(hs)} sampled val clips, {F:,} frames; missing = mean hand confidence < {HAND_MISSING_CONF})", "",
           f"- left hand missing: {pct(tot['lh_missing'], F)} of frames",
           f"- right hand missing: {pct(tot['rh_missing'], F)} of frames",
           f"- both hands missing: {pct(tot['both_missing'], F)} of frames",
           f"- frames with no person: {pct(tot['no_person'], F)}; frames with more than one person: {pct(tot['multi'], F)}",
           f"- clips where the right hand is missing in more than half the frames: {clips_rh_mostly_missing}/{len(hs)}",
           f"- keypoint counts (pose, face, left hand, right hand) seen: {kp['val']['shapes']}",
           f"- unreadable JSON files: {tot['bad_json']}"]
    print(f"[hands] right missing {pct(tot['rh_missing'], F)}, left missing {pct(tot['lh_missing'], F)}")

    # 7. Clip lengths
    md += ["", "## 7. Clip lengths (END - START, seconds)", "",
           "| split | n | mean | median | 90th pct | max | share <= 12 s |", "|---|---|---|---|---|---|---|"]
    plt.figure(figsize=(8, 4))
    for s in SPLITS:
        d = (pd.to_numeric(csvs[s]["END"], errors="coerce") - pd.to_numeric(csvs[s]["START"], errors="coerce")).dropna()
        d = d[d > 0]
        md.append(f"| {s} | {len(d):,} | {d.mean():.2f} | {d.median():.2f} | {d.quantile(0.9):.2f} | {d.max():.2f} | {pct((d <= 12).sum(), len(d))} |")
        plt.hist(d.clip(upper=40), bins=80, alpha=0.5, label=s, density=True)
    plt.axvline(12, color="k", linestyle="--", label="12 s limit")
    plt.xlabel("clip length (s, clipped at 40)")
    plt.ylabel("density")
    plt.legend()
    plt.title("How2Sign clip lengths")
    plt.tight_layout()
    fig_p = FIG / "clip_length_hist.png"
    plt.savefig(fig_p, dpi=120)
    plt.close()
    md += ["", f"![clip lengths](reports/figures/{fig_p.name})", ""]

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD} and {fig_p}")


if __name__ == "__main__":
    main()
