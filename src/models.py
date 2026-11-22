"""Pose CTC models (plan section 9).

Shared front-end: Linear(F->128), Conv1d(128,128,k=5,stride=2,pad=2) + GELU  ->  T' = ceil(T/2)  (7.5 steps/s at 15 fps).
TransformerCTC: sinusoidal PE, 3 layers, 4 heads, d_model 128, FFN 256, dropout 0.3, padding mask, Linear -> |vocab|.
BiGRUCTC: 2-layer bidirectional GRU, hidden 128 (dropout 0.3 between layers), Linear -> |vocab|.

forward(x (B,T,F), lengths (B,)) -> (log_probs (T',B,C), encoder features (B,T',d), output_lengths (B,))
d = 128 for TransformerCTC, 256 for BiGRUCTC.

Self-test:  python -m src.models --check
"""
import argparse
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

D_MODEL = 128


def pad_mask(lengths, T):
    return torch.arange(T, device=lengths.device)[None, :] >= lengths[:, None]   # True = padding


class FrontEnd(nn.Module):
    def __init__(self, in_dim, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.conv = nn.Conv1d(d, d, kernel_size=5, stride=2, padding=2)
        self.act = nn.GELU()

    def forward(self, x, lengths):
        h = self.proj(x).masked_fill(pad_mask(lengths, x.shape[1])[..., None], 0.0)
        h = self.act(self.conv(h.transpose(1, 2))).transpose(1, 2)
        out_lens = torch.div(lengths + 1, 2, rounding_mode="floor")                 # ceil(L/2)
        mask = pad_mask(out_lens, h.shape[1])
        return h.masked_fill(mask[..., None], 0.0), out_lens, mask


class PositionalEncoding(nn.Module):
    def __init__(self, d=D_MODEL, max_len=4000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, h):
        return h + self.pe[: h.shape[1]].unsqueeze(0)


class TransformerCTC(nn.Module):
    def __init__(self, in_dim, n_classes, d=D_MODEL, layers=3, heads=4, ffn=256, dropout=0.3):
        super().__init__()
        self.front = FrontEnd(in_dim, d)
        self.pe = PositionalEncoding(d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)
        self.enc_dim = d

    def forward(self, x, lengths):
        h, out_lens, mask = self.front(x, lengths)
        h = self.encoder(self.drop(self.pe(h)), src_key_padding_mask=mask)
        h = h.masked_fill(mask[..., None], 0.0)
        log_probs = self.head(h).log_softmax(dim=-1).transpose(0, 1)
        return log_probs, h, out_lens


class BiGRUCTC(nn.Module):
    def __init__(self, in_dim, n_classes, d=D_MODEL, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.front = FrontEnd(in_dim, d)
        self.gru = nn.GRU(d, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, n_classes)
        self.enc_dim = 2 * hidden

    def forward(self, x, lengths):
        h, out_lens, mask = self.front(x, lengths)
        packed = pack_padded_sequence(h, out_lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.gru(packed)
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=h.shape[1])
        o = o.masked_fill(mask[..., None], 0.0)
        log_probs = self.head(o).log_softmax(dim=-1).transpose(0, 1)
        return log_probs, o, out_lens


def build_model(name, in_dim, n_classes):
    if name == "transformer":
        return TransformerCTC(in_dim, n_classes)
    if name == "bigru":
        return BiGRUCTC(in_dim, n_classes)
    raise ValueError(f"unknown model {name}")


def _check():
    torch.manual_seed(0)
    C = 31
    lengths = torch.tensor([37, 80, 5, 121])
    ok = True
    for F in (206, 104):
        x = torch.randn(len(lengths), int(lengths.max()), F)
        for i, L in enumerate(lengths):
            x[i, L:] = 0.0
        for name in ("transformer", "bigru"):
            m = build_model(name, F, C)
            m.eval()
            with torch.no_grad():
                lp, enc, ol = m(x, lengths)
                lp0, _, ol0 = m(x[0:1, : int(lengths[0])], lengths[0:1])
            exp_T = math.ceil(int(lengths.max()) / 2)
            shapes = (tuple(lp.shape) == (exp_T, len(lengths), C) and tuple(enc.shape) == (len(lengths), exp_T, m.enc_dim)
                      and ol.tolist() == [math.ceil(int(L) / 2) for L in lengths])
            probs = torch.allclose(lp.exp().sum(-1)[: int(ol.min())], torch.ones(1), atol=1e-4)
            invariant = torch.allclose(lp[: int(ol0[0]), 0], lp0[:, 0], atol=1e-4)
            m.train()
            targets = torch.randint(1, C, (int(sum(min(3, int(o)) for o in ol)),))
            tl = torch.tensor([min(3, int(o)) for o in ol])
            lp_t, _, ol_t = m(x, lengths)
            loss = nn.CTCLoss(blank=0, zero_infinity=True)(lp_t, targets, ol_t, tl)
            loss.backward()
            grads = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            good = shapes and probs and invariant and torch.isfinite(loss).item() and grads
            ok &= bool(good)
            n_par = sum(p.numel() for p in m.parameters())
            print(f"{name:11s} F={F}: log_probs {tuple(lp.shape)}, enc {tuple(enc.shape)}, out_lens {ol.tolist()}, "
                  f"probs_sum_1={probs}, padding_invariant={invariant}, ctc_loss={loss.item():.3f}, "
                  f"grads_finite={grads}, params={n_par:,} -> {'OK' if good else 'FAIL'}")
    print("ALL MODEL CHECKS PASSED" if ok else "SOME MODEL CHECKS FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        _check()
    else:
        print(__doc__)
