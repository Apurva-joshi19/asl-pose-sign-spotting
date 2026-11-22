"""Metrics for gloss CTC (plan section 9).

greedy_decode: argmax -> collapse repeats -> remove blanks
edit_ops:      Levenshtein alignment with backtrace -> substitutions / deletions / insertions
corpus_wer:    WER = (S + D + I) / N_ref over all clips
bag_of_gloss:  order-free multiset precision / recall (labels are incomplete: some "insertions" may be real signs)
error_analysis: top substitution pairs, most-deleted glosses, per-gloss recall

Self-test:  python -m src.metrics --check
"""
import argparse
from collections import Counter

import torch


def greedy_decode(log_probs, out_lens, blank=0):
    """log_probs (T', B, C), out_lens (B,) -> list of id lists."""
    best = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    res = []
    for seq, L in zip(best, out_lens.cpu().tolist()):
        out, prev = [], None
        for t in seq[:L]:
            if t != prev and t != blank:
                out.append(t)
            prev = t
        res.append(out)
    return res


def edit_ops(ref, hyp):
    """Return (S, D, I, alignment); alignment items: ('C'|'S'|'D'|'I', ref_tok, hyp_tok)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, align = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            align.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            align.append(("D", ref[i - 1], None))
            i -= 1
        else:
            align.append(("I", None, hyp[j - 1]))
            j -= 1
    align.reverse()
    S = sum(a[0] == "S" for a in align)
    D = sum(a[0] == "D" for a in align)
    I = sum(a[0] == "I" for a in align)
    return S, D, I, align


def corpus_wer(refs, hyps):
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        s, d, i, _ = edit_ops(r, h)
        S, D, I, N = S + s, D + d, I + i, N + len(r)
    wer = (S + D + I) / N if N else float("nan")
    return {"wer": wer, "S": S, "D": D, "I": I, "N": N,
            "sub_rate": S / N if N else float("nan"), "del_rate": D / N if N else float("nan"),
            "ins_rate": I / N if N else float("nan")}


def bag_of_gloss(refs, hyps):
    inter = n_hyp = n_ref = 0
    for r, h in zip(refs, hyps):
        cr, ch = Counter(r), Counter(h)
        inter += sum((cr & ch).values())
        n_hyp += len(h)
        n_ref += len(r)
    return {"precision": inter / n_hyp if n_hyp else float("nan"),
            "recall": inter / n_ref if n_ref else float("nan"),
            "matched": inter, "n_hyp": n_hyp, "n_ref": n_ref}


def error_analysis(refs, hyps, top=10):
    subs, dels, ins = Counter(), Counter(), Counter()
    ref_count, hit = Counter(), Counter()
    for r, h in zip(refs, hyps):
        _, _, _, align = edit_ops(r, h)
        for op, a, b in align:
            if a is not None:
                ref_count[a] += 1
            if op == "C":
                hit[a] += 1
            elif op == "S":
                subs[(a, b)] += 1
            elif op == "D":
                dels[a] += 1
            else:
                ins[b] += 1
    per_gloss = {g: {"n_ref": ref_count[g], "recall": hit[g] / ref_count[g]} for g in ref_count}
    return {"top_substitutions": subs.most_common(top), "most_deleted": dels.most_common(top),
            "most_inserted": ins.most_common(top), "per_gloss_recall": per_gloss}


def ids_to_glosses(ids, id2gloss):
    return [id2gloss[i] for i in ids]


def _check():
    ok = True

    def expect(name, got, want):
        nonlocal ok
        good = got == want if not isinstance(want, float) else abs(got - want) < 1e-9
        ok &= good
        print(f"{name}: got {got}, expected {want} -> {'OK' if good else 'FAIL'}")

    seq = [0, 1, 1, 0, 1, 2, 2, 0, 3]
    C = 4
    lp = torch.full((len(seq), 1, C), -10.0)
    for t, k in enumerate(seq):
        lp[t, 0, k] = 0.0
    expect("greedy decode (blank separates repeats, length cut)",
           greedy_decode(lp, torch.tensor([8]))[0], [1, 1, 2])

    s, d, i, _ = edit_ops(list("ABC"), list("AXCD"))
    expect("edit ops ABC vs AXCD (S,D,I)", (s, d, i), (1, 0, 1))
    s, d, i, _ = edit_ops(list("AB"), [])
    expect("edit ops AB vs empty (S,D,I)", (s, d, i), (0, 2, 0))
    s, d, i, _ = edit_ops([], list("AB"))
    expect("edit ops empty vs AB (S,D,I)", (s, d, i), (0, 0, 2))

    w = corpus_wer([list("ABC"), list("AB")], [list("AXCD"), []])
    expect("corpus WER (2 subs/ins + 2 del over 5 refs)", w["wer"], 4 / 5)
    expect("empty prediction WER", corpus_wer([list("ABC")], [[]])["wer"], 1.0)

    b = bag_of_gloss([list("AAB")], [list("ABB")])
    expect("bag precision", b["precision"], 2 / 3)
    expect("bag recall", b["recall"], 2 / 3)

    e = error_analysis([list("ABC"), list("AB")], [list("AXCD"), list("A")])
    expect("top substitution", e["top_substitutions"][0], (("B", "X"), 1))
    expect("most deleted", e["most_deleted"][0], ("B", 1))
    expect("recall of A", e["per_gloss_recall"]["A"]["recall"], 1.0)
    expect("recall of B", e["per_gloss_recall"]["B"]["recall"], 0.0)
    print("ALL METRIC CHECKS PASSED" if ok else "SOME METRIC CHECKS FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        _check()
    else:
        print(__doc__)
