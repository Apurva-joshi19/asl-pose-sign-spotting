"""Extract ONLY the selected clips from the How2Sign archives (plan 6.4, member filtering).

RGB:       data/clips/<SENTENCE_NAME>.mp4
Keypoints: data/keypoints_json/<SENTENCE_NAME>/<SENTENCE_NAME>_<frame>_keypoints.json
Metadata:  annotations/clip_meta.csv (fps, frames, size, keypoint count, duration check, effective_fps)

effective_fps: metadata fps if frames/fps matches the CSV duration; metadata fps * 4/3 if the clip is
time-stretched by ~4/3 (observed for about a third of clips); otherwise flagged UNEXPLAINED.

Safe to rerun: clips already extracted completely are skipped.
"""
import json
import shutil
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEL = ROOT / "annotations" / "selected_clips.csv"
CLIPS = ROOT / "data" / "clips"
KPJ = ROOT / "data" / "keypoints_json"
META = ROOT / "annotations" / "clip_meta.csv"
DUR_TOL = 0.5                 # seconds allowed between frames/fps and CSV duration
STRETCH = 4.0 / 3.0           # observed time-stretch factor
STRETCH_RANGE = (1.20, 1.45)  # ratio video/csv accepted as the 4/3 stretch


def count_json(folder):
    return sum(1 for _ in folder.glob("*_keypoints.json")) if folder.is_dir() else 0


def main():
    if not SEL.exists():
        sys.exit("annotations/selected_clips.csv missing - run scripts/select_clips.py first")
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8-sig"))
    sel = pd.read_csv(SEL, dtype=str, keep_default_na=False, encoding="utf-8")
    CLIPS.mkdir(parents=True, exist_ok=True)
    KPJ.mkdir(parents=True, exist_ok=True)

    # 1. RGB from zip (random access)
    for split, g in sel.groupby("split"):
        zp = RAW / f"rgb_{split}" / manifest[f"rgb_{split}"]["file"]
        done = skipped = 0
        with zipfile.ZipFile(zp) as z:
            for _, r in g.iterrows():
                info = z.getinfo(r["rgb_member"])
                dst = CLIPS / f"{r['SENTENCE_NAME']}.mp4"
                if dst.exists() and dst.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                with z.open(info) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
                done += 1
        print(f"[rgb {split}] extracted {done}, already present {skipped}")

    # 2. Keypoints from tar.gz (one streaming pass per split, only selected folders written)
    for split, g in sel.groupby("split"):
        need = {n: int(c) for n, c in zip(g["SENTENCE_NAME"], g["n_json_frames"])}
        todo = {n for n, c in need.items() if count_json(KPJ / n) != c}
        if not todo:
            print(f"[kp {split}] all {len(need)} folders already complete")
            continue
        tp = RAW / f"kp_{split}" / manifest[f"kp_{split}"]["file"]
        print(f"[kp {split}] streaming {tp.name} for {len(todo)} folders ...", flush=True)
        t0 = time.time()
        members = written = 0
        with tarfile.open(tp, mode="r|gz") as tf:
            for m in tf:
                members += 1
                if members % 200000 == 0:
                    print(f"  [kp {split}] {members:,} members, {written:,} files written, {time.time() - t0:.0f}s", flush=True)
                if not m.isfile() or not m.name.endswith("_keypoints.json"):
                    continue
                parts = m.name.split("/")
                if len(parts) < 2 or parts[-2] not in todo:
                    continue
                out_dir = KPJ / parts[-2]
                out_dir.mkdir(exist_ok=True)
                f = tf.extractfile(m)
                (out_dir / Path(parts[-1]).name).write_bytes(f.read())
                written += 1
        print(f"[kp {split}] wrote {written:,} JSON files in {time.time() - t0:.0f}s")

    # 3. Metadata + checks + effective fps
    rows = []
    for _, r in sel.iterrows():
        name = r["SENTENCE_NAME"]
        mp4 = CLIPS / f"{name}.mp4"
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        kp_n = count_json(KPJ / name)
        dur_csv = float(r["duration"])
        dur_vid = frames / fps if fps > 0 else 0.0
        ratio = dur_vid / dur_csv if dur_csv > 0 else 0.0
        dur_ok = abs(dur_vid - dur_csv) <= DUR_TOL
        if dur_ok:
            eff, note = fps, "ok"
        elif STRETCH_RANGE[0] <= ratio <= STRETCH_RANGE[1]:
            eff, note = fps * STRETCH, "timescale_4/3"
        else:
            eff, note = fps, "UNEXPLAINED"
        rows.append({
            "SENTENCE_NAME": name, "split": r["split"], "role": r["role"], "order": r["order"],
            "fps": round(fps, 3), "n_frames": frames, "width": w, "height": h,
            "kp_frames": kp_n, "kp_expected": int(r["n_json_frames"]),
            "kp_ok": kp_n == int(r["n_json_frames"]) and abs(frames - kp_n) <= 2,
            "duration_csv": round(dur_csv, 3), "duration_video": round(dur_vid, 3),
            "duration_ok": dur_ok, "duration_ratio": round(ratio, 3),
            "effective_fps": round(eff, 3), "fps_note": note,
        })
    meta = pd.DataFrame(rows)
    meta.to_csv(META, index=False, encoding="utf-8")

    print(f"\nWrote {META} ({len(meta)} rows)")
    print(f"fps values: {dict(Counter(meta['fps']))}")
    print(f"frame sizes: {dict(Counter(meta['width'].astype(str) + 'x' + meta['height'].astype(str)))}")
    print(f"keypoints complete and within +-2 of video frames: {int(meta['kp_ok'].sum())}/{len(meta)}")
    print(f"frames/fps within {DUR_TOL}s of CSV duration: {int(meta['duration_ok'].sum())}/{len(meta)}")
    print(f"fps_note: {dict(Counter(meta['fps_note']))}")
    for split, g in meta.groupby("split"):
        print(f"  {split}: {dict(Counter(g['fps_note']))}")
    print(f"effective_fps values: {dict(Counter(meta['effective_fps']))}")
    bad = meta[(~meta["kp_ok"]) | (meta["fps_note"] == "UNEXPLAINED")]
    if len(bad):
        print("\nClips failing a check (first 10):")
        print(bad.head(10)[["SENTENCE_NAME", "split", "fps", "n_frames", "kp_frames",
                            "duration_csv", "duration_video", "duration_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
