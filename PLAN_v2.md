# PLAN v2 - Member 2: pose-based sign spotting with translation-derived weak labels (How2Sign)

Decided 2026-11-11 (Day 5). Replaces sections 4-14 of the original master prompt. Reason: the team has no ASL
knowledge and manual visual glossing was not feasible; the ASLLRP server (calibration source) was also unavailable.
To be confirmed with group and supervisor.

## Task
Input: pose keypoints of a How2Sign clip (frontal view). Output: which lexicon signs occur in the clip
(multi-label, order-free). Labels are generated automatically from the English translation via lexicon triggers.
Report wording: "translation-derived weak labels" - never "glosses", never expert- or Deaf-verified.

## Kept from the original plan
- How2Sign data, official train/val/test splits, lexicon.csv v1 (60 signs) and its english_triggers
- effective_fps fix; src/features.py (15 fps, F=206 / 104 without velocity)
- Front-end + TransformerCTC / BiGRUCTC encoders (7.5 steps/s; encoder features for fusion)
- 3 configurations x 3 seeds: Transformer full, BiGRU full, Transformer no-velocity (ablation); training on Colab T4
- Honest documentation; never invent results

## Changes
- Data: all clips with keypoints (~31k train, ~1.7k val, ~2.3k test), including clips with no lexicon word.
- Keypoints streamed from tar.gz directly to compact per-split feature files (pos + flags at 15 fps).
- effective_fps = keypoint frame count / (END - START) from the CSV (validated on the 495 clips in clip_meta.csv).
- Classes: lexicon signs with >= 30 train positives and >= 5 positives in val and in test (others dropped, recorded).
- Model head: masked mean+max pooling over time -> Linear -> one logit per class; loss BCEWithLogits with pos_weight.
- Metrics: mAP (macro), micro/macro F1 (threshold chosen on val), per-class AP, baselines (class prior, random).
- Dropped: manual annotation, annotation tool use, calibration, self-consistency, kappa.
- Optional: if the ASLLRP server returns, measure weak-label noise automatically (English triggers vs expert glosses).

## Cut order
ablation -> train clips 31k -> 10k -> seeds 3 -> 2. Never cut: run_checks, honest documentation.

## Integration (Member 1)
Same SENTENCE_NAME IDs, official splits, labels_{split}.json (list of signs present per clip), vocab/classes file,
15 fps resampling with effective_fps, stride 2 -> 7.5 steps/s, encoder features + per-class logits per clip.

## Schedule
Day 5 plan + weak labels | Day 6 keypoints -> features | Day 7 dataset/head/metrics/train + checks |
Day 8 Colab bundle, overfit test, trial | Day 9 9 runs | Day 10 evaluation | Day 11 handover (+ optional noise check) |
Day 12-13 reports | Day 14 clean re-run, buffer
