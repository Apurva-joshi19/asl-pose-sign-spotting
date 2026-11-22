"""Local gloss annotation tool (plan 6.5). Run:  python app/annotate.py   (opens in your browser)

Modes:
  pilot                - speed test on How2Sign clips -> annotations/pilot_annotations.csv (never used as labels)
  pass1                - first pass on How2Sign clips -> annotations/how2sign_gloss_v1.csv (pass=1)
  pass2 (blind)        - blind re-annotation of annotations/pass2_clips.csv -> same CSV (pass=2)
  calibration (blind)  - ASLLRP videos in data/calibration/videos -> annotations/calibration_annotations.csv

Every save: CSV upsert (one row per clip and pass) + timestamped JSON backup in annotations/backup/.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotations"
CLIPS = ROOT / "data" / "clips"
CALIB = ROOT / "data" / "calibration" / "videos"
LEX_P = ROOT / "lexicon.csv"
SEL_P = ANN / "selected_clips.csv"
META_P = ANN / "clip_meta.csv"
PASS2_P = ANN / "pass2_clips.csv"
BACKUP = ANN / "backup"
GUIDELINE_VERSION = "g1"
SIGNBANK_URL = "https://dai.cs.rutgers.edu/dai/s/signbank"
FRAME_W = 480
TOP15 = ["GO", "WANT", "LITTLE-BIT", "THING", "USE", "GOOD", "COME", "LOOK", "TIME", "MUST",
         "KNOW", "START", "MANY", "TALK", "IX-1p"]
COLS = ["SENTENCE_NAME", "VIDEO_ID", "split", "sentence_en", "n_frames", "fps", "gloss_seq", "peak_frames",
        "confidence_seq", "n_unsure", "minutes_spent", "pass", "annotated_on", "guideline_version", "lexicon_version"]
MODES = ["pilot", "pass1", "pass2 (blind)", "calibration (blind)"]
OUT = {
    "pilot": ANN / "pilot_annotations.csv",
    "pass1": ANN / "how2sign_gloss_v1.csv",
    "pass2 (blind)": ANN / "how2sign_gloss_v1.csv",
    "calibration (blind)": ANN / "calibration_annotations.csv",
}
CONF = {"sure (S)": "S", "unsure (U)": "U"}
CACHE = {"name": None, "frames": []}
LEXICON_VERSION = "v1"


def read_csv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def read_ann(p):
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    df = read_csv(p)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df


LEX = read_csv(LEX_P)
LEX_IDS = LEX["gloss_id"].tolist()
LEX_BY_ID = LEX.set_index("gloss_id")
SEL = read_csv(SEL_P)
SEL_BY = SEL.set_index("SENTENCE_NAME")
META = read_csv(META_P).set_index("SENTENCE_NAME") if META_P.exists() else pd.DataFrame()


def is_blind(mode):
    return "blind" in mode


def is_cal(mode):
    return mode.startswith("calibration")


def pass_no(mode):
    return 2 if mode.startswith("pass2") else 1


def clip_records(mode, split):
    if is_cal(mode):
        names = sorted(p.stem for p in CALIB.glob("*.mp4")) if CALIB.exists() else []
        return [{"SENTENCE_NAME": n, "role": "calib", "order": str(i + 1)} for i, n in enumerate(names)]
    if mode.startswith("pass2"):
        if not PASS2_P.exists():
            return []
        names = read_csv(PASS2_P)["SENTENCE_NAME"].tolist()
        return [{"SENTENCE_NAME": n, "role": SEL_BY.at[n, "role"] if n in SEL_BY.index else "",
                 "order": str(i + 1)} for i, n in enumerate(names)]
    return SEL[SEL["split"] == split][["SENTENCE_NAME", "role", "order"]].to_dict("records")


def done_names(mode):
    df = read_ann(OUT[mode])
    return set(df.loc[df["pass"] == str(pass_no(mode)), "SENTENCE_NAME"])


def clip_choices(mode, split):
    done = done_names(mode)
    out = []
    for r in clip_records(mode, split):
        mark = "✓" if r["SENTENCE_NAME"] in done else "·"
        out.append((f"{mark} {int(r['order']):03d} {r['role']:<7} {r['SENTENCE_NAME']}", r["SENTENCE_NAME"]))
    return out


def progress_text(mode, split):
    ch = clip_choices(mode, split)
    n_done = sum(1 for label, _ in ch if label.startswith("✓"))
    extra = ""
    if mode.startswith("pass2") and not PASS2_P.exists():
        extra = " (annotations/pass2_clips.csv not created yet)"
    if is_cal(mode) and not ch:
        extra = " (no videos in data/calibration/videos yet)"
    return f"**{n_done}/{len(ch)}** clips done in this list{extra}"


def refresh_list(mode, split):
    return gr.Dropdown(choices=clip_choices(mode, split), value=None), progress_text(mode, split)


def decode(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        small = cv2.resize(fr, (FRAME_W, max(1, int(h * FRAME_W / w))), interpolation=cv2.INTER_AREA)
        ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(buf.tobytes())
    cap.release()
    return frames, fps


def frame_image(idx):
    frames = CACHE["frames"]
    if not frames:
        return None
    i = int(max(0, min(int(idx or 0), len(frames) - 1)))
    img = cv2.imdecode(np.frombuffer(frames[i], np.uint8), cv2.IMREAD_COLOR)
    cv2.putText(img, f"frame {i}/{len(frames) - 1}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, f"frame {i}/{len(frames) - 1}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def marks_table(marks):
    return pd.DataFrame([{"#": i + 1, "gloss_id": m["gloss_id"], "peak_frame": m["frame"], "conf": m["conf"]}
                         for i, m in enumerate(marks)], columns=["#", "gloss_id", "peak_frame", "conf"])


def parse_marks(r):
    g, f, c = r["gloss_seq"].split(), r["peak_frames"].split(), r["confidence_seq"].split()
    return [{"gloss_id": a, "frame": int(b), "conf": cc} for a, b, cc in zip(g, f, c)]


def saved_row(mode, name):
    df = read_ann(OUT[mode])
    m = df[(df["SENTENCE_NAME"] == name) & (df["pass"] == str(pass_no(mode)))]
    return None if m.empty else m.iloc[0]


def sign_info(gid):
    if not gid or gid not in LEX_BY_ID.index:
        return ""
    r = LEX_BY_ID.loc[gid]
    return (f"**{gid}** - Sign Bank entry: `{r['asllrp_gloss']}` (search it in the [ASLLRP Sign Bank]({SIGNBANK_URL}))  \n"
            f"Accepted variants: {r['accepted_variants'].replace('|', ', ')}  \nNotes: {r['notes'] or '-'}")


def load_clip(mode, split, name, state):
    state = state or {}
    if state.get("dirty"):
        raise gr.Error("This clip has unsaved marks. Press 'Save clip' or 'Discard unsaved changes' first.")
    if not name:
        raise gr.Error("Choose a clip in the list first.")
    path = (CALIB if is_cal(mode) else CLIPS) / f"{name}.mp4"
    if not path.exists():
        raise gr.Error(f"Video not found: {path}")
    frames, fps_meta = decode(path)
    if not frames:
        raise gr.Error("Could not read any frames from this video.")
    CACHE["name"], CACHE["frames"] = name, frames
    n = len(frames)
    eff = float(META.at[name, "effective_fps"]) if (not META.empty and name in META.index) else round(fps_meta, 3)
    sel = SEL_BY.loc[name] if name in SEL_BY.index else None
    clip_split = "calibration" if is_cal(mode) else (sel["split"] if sel is not None else split)
    sentence = sel["SENTENCE"] if sel is not None else ""
    cands = [c for c in (sel["candidates"].split("|") if sel is not None and sel["candidates"] else []) if c in LEX_IDS]
    row = saved_row(mode, name)
    marks = parse_marks(row) if row is not None else []
    prior = float(row["minutes_spent"] or 0) if row is not None else 0.0
    blind = is_blind(mode)
    state = {"mode": mode, "name": name, "split": clip_split, "n_frames": n, "fps": eff, "marks": marks,
             "t_load": time.time(), "prior_minutes": prior, "dirty": False,
             "video_id": sel["VIDEO_ID"] if sel is not None else "", "sentence": sentence}
    lines = [f"**{name}** | split: {clip_split} | pass: {pass_no(mode)} | frames: {n} | "
             f"video fps: {fps_meta:.3f} | effective fps: {eff}"]
    if blind:
        lines.append("_Blind mode: English sentence, pre-listed candidates and other-pass labels are hidden._")
    else:
        lines.append(f"**English:** {sentence}")
        lines.append(f"**Pre-listed candidates:** {', '.join(cands) if cands else '(none)'}")
    lines.append(f"Also scan for: {', '.join(TOP15)}")
    if row is not None:
        lines.append(f"_Loaded your saved marks for this pass (minutes so far: {prior:.2f})._")
    return (str(path),
            gr.Slider(minimum=0, maximum=max(1, n - 1), value=0, step=1, label="Frame index"),
            frame_image(0),
            "  \n".join(lines),
            gr.Dropdown(choices=[(c, c) for c in cands], value=None, visible=not blind),
            marks_table(marks),
            state,
            f"Loaded {name} ({n} frames). Timer started.")


def next_clip(mode, split, state):
    done = done_names(mode)
    todo = [r["SENTENCE_NAME"] for r in clip_records(mode, split) if r["SENTENCE_NAME"] not in done]
    if not todo:
        raise gr.Error("All clips in this list are done.")
    return (gr.Dropdown(value=todo[0]),) + load_clip(mode, split, todo[0], state)


def nudge(delta):
    def f(v, state):
        n = (state or {}).get("n_frames", 1)
        return int(min(max(0, int(v or 0) + delta), max(0, n - 1)))
    return f


def add_mark(state, idx, cand, full, conf):
    if not state or not state.get("name"):
        raise gr.Error("Load a clip first.")
    gloss = full or cand
    if not gloss:
        raise gr.Error("Choose a sign (pre-listed candidate or full lexicon).")
    if gloss not in LEX_IDS:
        raise gr.Error(f"{gloss} is not in the lexicon.")
    marks = state["marks"] + [{"gloss_id": gloss, "frame": int(idx), "conf": CONF[conf]}]
    state["marks"] = sorted(marks, key=lambda m: (m["frame"], m["gloss_id"]))
    state["dirty"] = True
    return marks_table(state["marks"]), state, f"Added {gloss} at frame {int(idx)} ({CONF[conf]}). Not saved yet.", None, None


def delete_mark(state, num):
    if not state or not state.get("marks"):
        raise gr.Error("No marks to delete.")
    k = int(num or 0) - 1
    if k < 0 or k >= len(state["marks"]):
        raise gr.Error(f"Mark # must be between 1 and {len(state['marks'])}.")
    m = state["marks"].pop(k)
    state["dirty"] = True
    return marks_table(state["marks"]), state, f"Deleted {m['gloss_id']} at frame {m['frame']}. Not saved yet."


def discard(state):
    if not state or not state.get("name"):
        raise gr.Error("Nothing loaded.")
    row = saved_row(state["mode"], state["name"])
    state["marks"] = parse_marks(row) if row is not None else []
    state["dirty"] = False
    return marks_table(state["marks"]), state, "Unsaved changes discarded (back to last saved version)."


def save_clip(state, mode, split):
    if not state or not state.get("name"):
        raise gr.Error("Load a clip first.")
    if state["mode"] != mode:
        raise gr.Error("The mode was changed after loading. Switch back or reload the clip.")
    marks = sorted(state["marks"], key=lambda m: (m["frame"], m["gloss_id"]))
    minutes = round(state["prior_minutes"] + (time.time() - state["t_load"]) / 60.0, 2)
    p = pass_no(mode)
    row = {
        "SENTENCE_NAME": state["name"], "VIDEO_ID": state["video_id"], "split": state["split"],
        "sentence_en": state["sentence"], "n_frames": state["n_frames"], "fps": state["fps"],
        "gloss_seq": " ".join(m["gloss_id"] for m in marks),
        "peak_frames": " ".join(str(m["frame"]) for m in marks),
        "confidence_seq": " ".join(m["conf"] for m in marks),
        "n_unsure": sum(m["conf"] == "U" for m in marks), "minutes_spent": minutes, "pass": p,
        "annotated_on": datetime.now().isoformat(timespec="seconds"),
        "guideline_version": GUIDELINE_VERSION, "lexicon_version": LEXICON_VERSION,
    }
    out = OUT[mode]
    df = read_ann(out)
    keep = ~((df["SENTENCE_NAME"] == row["SENTENCE_NAME"]) & (df["pass"] == str(p)))
    df = pd.concat([df[keep][COLS], pd.DataFrame([row], columns=COLS).astype(str)], ignore_index=True)
    tmp = out.with_suffix(".tmp")
    try:
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(out)
    except PermissionError:
        raise gr.Error(f"Cannot write {out.name}. Is it open in Excel? Close it and press Save again.")
    bdir = BACKUP / mode.split()[0]
    bdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (bdir / f"{state['name']}_pass{p}_{stamp}.json").write_text(json.dumps({**row, "marks": marks}, indent=2), encoding="utf-8")
    state["prior_minutes"], state["t_load"], state["dirty"] = minutes, time.time(), False
    return (f"Saved {state['name']}: {len(marks)} marks [{row['gloss_seq'] or 'empty'}], {minutes:.2f} min total.",
            gr.Dropdown(choices=clip_choices(mode, split), value=state["name"]),
            progress_text(mode, split), state)


def build_ui():
    full_choices = [(f"{g}  ({a})", g) for g, a in zip(LEX["gloss_id"], LEX["asllrp_gloss"])]
    with gr.Blocks(title="ASL gloss annotator") as demo:
        state = gr.State({})
        gr.Markdown(f"## How2Sign gloss annotator  \nGuidelines {GUIDELINE_VERSION} | lexicon {LEXICON_VERSION} "
                    f"({len(LEX_IDS)} signs) | follow annotation_guidelines.md")
        with gr.Row():
            mode = gr.Radio(MODES, value="pilot", label="Mode")
            split = gr.Dropdown(["test", "val", "train"], value="train", label="Split (pilot/pass1 only)")
        with gr.Row():
            clip_dd = gr.Dropdown(choices=clip_choices("pilot", "train"), label="Clip", scale=4)
            load_btn = gr.Button("Load clip", variant="primary")
            next_btn = gr.Button("Next unannotated")
        progress = gr.Markdown(progress_text("pilot", "train"))
        info = gr.Markdown()
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Play once at 1x", interactive=False)
            with gr.Column():
                image = gr.Image(label="Frame", type="numpy", interactive=False)
                slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Frame index")
                with gr.Row():
                    b_m5, b_m1 = gr.Button("-5"), gr.Button("-1")
                    b_p1, b_p5 = gr.Button("+1"), gr.Button("+5")
        with gr.Row():
            cand_dd = gr.Dropdown(choices=[], label="Pre-listed candidates")
            full_dd = gr.Dropdown(choices=full_choices, label="Full lexicon (type to search)", filterable=True)
            conf = gr.Radio(list(CONF), value="sure (S)", label="Confidence")
            add_btn = gr.Button("Add mark at current frame", variant="primary")
        sign_md = gr.Markdown()
        marks_df = gr.Dataframe(value=marks_table([]), interactive=False, label="Marks (sorted by peak frame)")
        with gr.Row():
            del_num = gr.Number(value=1, precision=0, label="Mark # to delete")
            del_btn = gr.Button("Delete mark")
            discard_btn = gr.Button("Discard unsaved changes")
            save_btn = gr.Button("Save clip", variant="primary")
        status = gr.Markdown()

        load_outs = [video, slider, image, info, cand_dd, marks_df, state, status]
        mode.change(refresh_list, [mode, split], [clip_dd, progress])
        split.change(refresh_list, [mode, split], [clip_dd, progress])
        load_btn.click(load_clip, [mode, split, clip_dd, state], load_outs)
        next_btn.click(next_clip, [mode, split, state], [clip_dd] + load_outs)
        slider.change(frame_image, slider, image)
        for b, d in ((b_m5, -5), (b_m1, -1), (b_p1, 1), (b_p5, 5)):
            b.click(nudge(d), [slider, state], slider)
        cand_dd.change(sign_info, cand_dd, sign_md)
        full_dd.change(sign_info, full_dd, sign_md)
        add_btn.click(add_mark, [state, slider, cand_dd, full_dd, conf], [marks_df, state, status, cand_dd, full_dd])
        del_btn.click(delete_mark, [state, del_num], [marks_df, state, status])
        discard_btn.click(discard, [state], [marks_df, state, status])
        save_btn.click(save_clip, [state, mode, split], [status, clip_dd, progress, state])
    return demo


def main():
    global LEXICON_VERSION
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon-version", default="v1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    LEXICON_VERSION = args.lexicon_version
    for d in (ANN, BACKUP, CALIB):
        d.mkdir(parents=True, exist_ok=True)
    build_ui().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True,
                      allowed_paths=[str(CLIPS), str(CALIB)])


if __name__ == "__main__":
    main()
