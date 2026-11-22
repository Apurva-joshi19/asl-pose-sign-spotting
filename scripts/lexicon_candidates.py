"""Rank English lemmas in How2Sign TRAIN sentences as candidates for the 60-sign lexicon.

Keeps NOUN/VERB/ADJ/ADV lemmas; drops stop-words, numbers, non-alphabetic tokens and proper nouns.
Ranks by number of train sentences containing the lemma (what matters for clip selection).

Output: annotations/lexicon_candidates.csv (top 200)
"""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "annotations" / "lexicon_candidates.csv"
KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV"}
FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
TOP_N = 200


def main():
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8-sig"))
    csv_p = RAW / "csv_train" / manifest["csv_train"]["file"]
    df = pd.read_csv(csv_p, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    sents = df["SENTENCE"].tolist()
    print(f"{len(sents):,} train sentences")

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    tok_count = Counter()
    sent_count = Counter()
    pos_count = defaultdict(Counter)
    example = {}
    first_person = 0
    t0 = time.time()

    for i, doc in enumerate(nlp.pipe(sents, batch_size=512)):
        seen = set()
        has_fp = False
        for t in doc:
            if t.text.lower() in FIRST_PERSON:
                has_fp = True
            if t.pos_ not in KEEP_POS or t.is_stop or t.like_num or not t.is_alpha:
                continue
            lem = t.lemma_.lower()
            if len(lem) < 2:
                continue
            tok_count[lem] += 1
            pos_count[lem][t.pos_] += 1
            if lem not in seen:
                sent_count[lem] += 1
                seen.add(lem)
            if lem not in example and 20 <= len(sents[i]) <= 120:
                example[lem] = sents[i]
        first_person += has_fp
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,} sentences ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for rank, (lem, n) in enumerate(sent_count.most_common(TOP_N), 1):
        rows.append({
            "rank": rank,
            "lemma": lem,
            "n_sentences": n,
            "share_sentences": round(n / len(sents), 4),
            "n_tokens": tok_count[lem],
            "main_pos": pos_count[lem].most_common(1)[0][0],
            "example": example.get(lem, ""),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    print(f"\nDone in {time.time() - t0:.0f}s. Wrote {OUT}")
    print(f"Sentences with I/me/my/mine/myself (IX-1p proxy): {first_person:,} ({100 * first_person / len(sents):.1f}%)")
    print("\nTOP 200 (rank lemma(n_sentences)):")
    line = []
    for r in rows:
        line.append(f"{r['rank']:3d} {r['lemma']}({r['n_sentences']})")
        if len(line) == 6:
            print("  ".join(line))
            line = []
    if line:
        print("  ".join(line))


if __name__ == "__main__":
    main()
