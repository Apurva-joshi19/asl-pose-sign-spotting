"""Pose sign-spotting model (PLAN_v2): shared front-end + Transformer or BiGRU encoder (from src.models),
masked mean+max pooling over time, dropout, Linear -> one logit per class.

forward(x (B,T,F), lengths (B,)) -> (logits (B,C), encoder features (B,T',d), output_lengths (B,))
T' = ceil(T/2) (7.5 steps/s at 15 fps); d = 128 (transformer) or 256 (bigru).

Self-test:  python -m src.spotter --check
"""
import argparse
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models import BiGRUCTC, TransformerCTC, build_model, pad_mask


def encode(b, x, lengths):
    h, out_lens, mask = b.front(x, lengths)
    if isinstance(b, TransformerCTC):
        h = b.encoder(b.drop(b.pe(h)), src_key_padding_mask=mask)
    elif isinstance(b, BiGRUCTC):
        packed = pack_padded_sequence(h, out_lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = b.gru(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=mask.shape[1])
    else:
        raise TypeError(type(b))
    return h.masked_fill(mask[..., None], 0.0), out_lens, mask


class Spotter(nn.Module):
    def __init__(self, backbone, in_dim, n_classes, dropout=0.3):
        super().__init__()
        self.backbone = build_model(backbone, in_dim, 2)
        del self.backbone.head                      # CTC output layer not used
        d = self.backbone.enc_dim
        self.enc_dim = d
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * d, n_classes)

    def forward(self, x, lengths):
        h, out_lens, mask = encode(self.backbone, x, lengths)
        mean = h.sum(1) / out_lens.clamp(min=1)[:, None].to(h.dtype)
        mx = h.masked_fill(mask[..., None], float("-inf")).max(1).values
        logits = self.head(self.drop(torch.cat([mean, mx], dim=-1)))
        return logits, h, out_lens


def _check():
    torch.manual_seed(0)
    C = 55
    lengths = torch.tensor([37, 80, 5, 121])
    ok = True
    for F in (206, 104):
        x = torch.randn(len(lengths), int(lengths.max()), F)
        for i, L in enumerate(lengths):
            x[i, L:] = 0.0
        for name in ("transformer", "bigru"):
            m = Spotter(name, F, C).eval()
            with torch.no_grad():
                lg, enc, ol = m(x, lengths)
                lg0, _, _ = m(x[0:1, : int(lengths[0])], lengths[0:1])
            Tp = math.ceil(int(lengths.max()) / 2)
            shapes = tuple(lg.shape) == (4, C) and tuple(enc.shape) == (4, Tp, m.enc_dim) and ol.tolist() == [math.ceil(int(L) / 2) for L in lengths]
            inv = torch.allclose(lg[0], lg0[0], atol=1e-4)
            m.train()
            y = (torch.rand(4, C) < 0.05).float()
            loss = nn.BCEWithLogitsLoss(pos_weight=torch.full((C,), 20.0))(m(x, lengths)[0], y)
            loss.backward()
            grads = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            good = shapes and inv and bool(torch.isfinite(loss)) and grads
            ok &= good
            print(f"{name:11s} F={F}: logits {tuple(lg.shape)}, enc {tuple(enc.shape)}, out_lens {ol.tolist()}, "
                  f"padding_invariant={inv}, bce={loss.item():.3f}, grads_finite={grads}, "
                  f"params={sum(p.numel() for p in m.parameters()):,} -> {'OK' if good else 'FAIL'}")
    print("ALL SPOTTER CHECKS PASSED" if ok else "SOME SPOTTER CHECKS FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        _check()
    else:
        print(__doc__)
