"""Back up code, documentation, labels, results and trained models (no raw data) into ASL_backup_<date>.zip.

Excluded (large, recreatable or already on Drive): data/, keypoints/, weak/features_*.npz, last.pt files,
.venv, caches, colab_bundle.zip, other zips except the handover v1 package.
Usage: python scripts/make_backup.py
"""
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ["src", "scripts", "tests", "app", "annotations", "reports", "handover/v1"]
FILES = ["PLAN_v2.md", "lexicon.csv", "annotation_guidelines.md", "data_inspection.md", "requirements.txt",
         ".gitignore", "handover/member1_handover_v1.zip"]
CKPT_KEEP = {"best.pt", "log.csv", "config.json", "final.json"}


def wanted(p):
    s = p.as_posix()
    return "__pycache__" not in s and not s.endswith((".pyc", ".tmp"))


def main():
    items = []
    for d in DIRS:
        base = ROOT / d
        if base.exists():
            items += [p for p in base.rglob("*") if p.is_file() and wanted(p)]
    items += [ROOT / f for f in FILES if (ROOT / f).is_file()]
    weak = ROOT / "weak"
    items += [p for p in weak.glob("*") if p.is_file() and not p.name.startswith("features_")]
    ck = ROOT / "checkpoints"
    items += [p for p in ck.glob("*/*") if p.is_file() and p.name in CKPT_KEEP]
    items = sorted(set(items))

    out = ROOT / f"ASL_backup_{time.strftime('%Y%m%d_%H%M')}.zip"
    counts = {}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in items:
            arc = p.relative_to(ROOT).as_posix()
            z.write(p, arc)
            top = arc.split("/")[0]
            counts[top] = counts.get(top, 0) + 1
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        n = len(z.namelist())
    print(f"wrote {out.name}: {n} files, {out.stat().st_size / 1e6:.1f} MB; integrity check: {'OK' if bad is None else 'BAD ' + bad}")
    for k in sorted(counts):
        print(f"  {k:16s} {counts[k]} files")
    print(f"  trained models (best.pt): {sum(1 for p in items if p.name == 'best.pt')}")


if __name__ == "__main__":
    main()
