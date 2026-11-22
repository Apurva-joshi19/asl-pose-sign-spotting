"""Download How2Sign files from Google Drive with gdown.

Usage:
  python scripts/download_data.py small               # 3 CSVs + val/test keypoints + val/test RGB
  python scripts/download_data.py train               # train keypoints + train RGB (large, ~52 GB)
  python scripts/download_data.py status              # show what is on disk
  python scripts/download_data.py register KEY PATH   # record a file downloaded by browser

Each file goes into data/raw/<key>/ with its original Drive filename.
Completed files are skipped on rerun; partial downloads are resumed when possible.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"
GIB = 1024 ** 3

# key: (drive_id, group, expected_size_GiB or None)
FILES = {
    "csv_train": ("1lq7ksWeD3FzaIwowRbe_BvCmSmOG12-f", "small", None),
    "csv_val":   ("1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3", "small", None),
    "csv_test":  ("1ScxYnEjILZMn22qKjQj8Wyr_F0nha7kG", "small", None),
    "kp_val":    ("1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w", "small", 1.2),
    "kp_test":   ("1g8tzzW5BNPzHXlamuMQOvdwlHRa-29Vp", "small", 1.6),
    "rgb_val":   ("1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO", "small", 1.7),
    "rgb_test":  ("1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG", "small", 2.2),
    "kp_train":  ("1TBX7hLraMiiLucknM1mhblNVomO9-Y0r", "train", 21.0),
    "rgb_train": ("1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW", "train", 31.0),
}


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    return {}


def save_manifest(m):
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")


def looks_like_html(path):
    try:
        with open(path, "rb") as f:
            head = f.read(1024).lower()
        return b"<html" in head or b"<!doctype html" in head
    except OSError:
        return False


def is_quota(msg):
    low = msg.lower()
    return "too many users" in low or "quota" in low or "html page" in low


def download_one(key, manifest):
    fid, group, exp = FILES[key]
    out_dir = RAW / key
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = manifest.get(key, {})
    if rec.get("status") == "ok":
        p = out_dir / rec.get("file", "")
        if p.is_file() and p.stat().st_size == rec.get("bytes"):
            print(f"[skip] {key}: already downloaded ({p.name})")
            return True

    last_err = ""
    for attempt in (1, 2):
        last_err = ""
        print(f"\n[download] {key} (attempt {attempt}) id={fid}")
        try:
            path = gdown.download(id=fid, output=str(out_dir) + os.sep, quiet=False, resume=True)
        except Exception as e:
            path = None
            last_err = " ".join(str(e).split()) or repr(e)

        if path and Path(path).is_file():
            p = Path(path)
            size = p.stat().st_size
            if looks_like_html(p) or (exp is not None and size < 1_000_000):
                last_err = "got an HTML page instead of the file (Drive quota or permission)"
                p.unlink(missing_ok=True)
            else:
                manifest[key] = {
                    "status": "ok", "file": p.name, "bytes": size, "drive_id": fid,
                    "source": "gdown", "downloaded_on": time.strftime("%Y-%m-%d %H:%M"),
                }
                save_manifest(manifest)
                print(f"[ok] {key}: {p.name} ({size / GIB:.2f} GiB)")
                return True
        elif not last_err:
            last_err = "gdown returned no file"

        if is_quota(last_err):
            break  # retrying immediately never helps with the quota
        if attempt == 1:
            print(f"[retry] {key}: {last_err[:120]} -> waiting 30 s")
            time.sleep(30)

    reason = "QUOTA (too many users)" if is_quota(last_err) else last_err[:200]
    manifest[key] = {
        "status": "failed", "error": reason, "drive_id": fid,
        "tried_on": time.strftime("%Y-%m-%d %H:%M"),
    }
    save_manifest(manifest)
    print(f"[FAILED] {key}: {reason}")
    if is_quota(last_err):
        print(f"         -> try the browser: https://drive.google.com/uc?id={fid}")
    return False


def register(key, src, manifest):
    if key not in FILES:
        print(f"Unknown key '{key}'. Valid keys: {', '.join(FILES)}")
        return False
    src = Path(src)
    if not src.is_file():
        print(f"File not found: {src}")
        return False
    fid, group, exp = FILES[key]
    size = src.stat().st_size
    if looks_like_html(src) or (exp is not None and size < 1_000_000):
        print(f"{src.name} looks like a web page, not the archive. Not registered.")
        return False
    out_dir = RAW / key
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    if src.resolve() != dst.resolve():
        print(f"[move] {src} -> {dst}")
        shutil.move(str(src), str(dst))
    manifest[key] = {
        "status": "ok", "file": dst.name, "bytes": size, "drive_id": fid,
        "source": "browser", "downloaded_on": time.strftime("%Y-%m-%d %H:%M"),
    }
    save_manifest(manifest)
    print(f"[ok] registered {key}: {dst.name} ({size / GIB:.2f} GiB)")
    return True


def status(manifest):
    print(f"\n{'key':10s} {'group':6s} {'status':8s} {'GiB':>7s} {'expect':>7s}  {'check':6s}  file / error")
    for key, (fid, group, exp) in FILES.items():
        rec = manifest.get(key, {})
        st = rec.get("status", "missing")
        size, check, name = 0.0, "-", rec.get("error", "")[:60]
        if st == "ok":
            p = RAW / key / rec["file"]
            name = rec["file"]
            if not p.is_file():
                st = "missing"
            else:
                size = p.stat().st_size / GIB
                if exp is None:
                    check = "OK"
                else:
                    check = "OK" if 0.75 * exp <= size <= 1.25 * exp else "SIZE?"
        exp_s = f"{exp:.1f}" if exp else "-"
        print(f"{key:10s} {group:6s} {st:8s} {size:7.2f} {exp_s:>7s}  {check:6s}  {name}")


def main():
    args = sys.argv[1:]
    manifest = load_manifest()
    if len(args) == 3 and args[0] == "register":
        ok = register(args[1], args[2], manifest)
        status(manifest)
        sys.exit(0 if ok else 1)
    if len(args) != 1 or args[0] not in ("small", "train", "status"):
        print(__doc__)
        sys.exit(2)
    mode = args[0]
    failed = []
    if mode != "status":
        keys = [k for k, v in FILES.items() if v[1] == mode]
        failed = [k for k in keys if not download_one(k, manifest)]
        print("\n===== SUMMARY =====")
        print(f"group '{mode}': {len(keys) - len(failed)}/{len(keys)} ok")
        if failed:
            print("FAILED:", ", ".join(failed))
    status(manifest)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
