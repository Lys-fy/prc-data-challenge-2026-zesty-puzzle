#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Inference-time long-context knobs for Kumo Tabular, OFF unless the env is set; parity unchanged.
  LOGN_CLAMP=<n0>  LogScale (SSMax-style q * a_h * log n) sees key_len -> min(key_len, n0). Affects only attentions with > n0 keys (ICL context rows, column
                   inducing blocks); row blocks (<= 416 keys) are untouched. GatedLogScale inherits the patch (it calls LogScale.forward). Setting used: n0 = 60000.
  ROPE_PI=<T0>     RoPE position interpolation over the COLUMN-token axis of the row blocks: positions = arange(T) * min(1, T0 / T). Setting used: T0 = 104
                   (4 readout + <= 100 columns = the pretraining range). The RMSNorm fast path builds its own cos/sin tables, so with ROPE_PI set the rotation
                   is applied eagerly through RotaryEmbedding.forward (still fused RMSNorm afterwards); numerically the same op, slightly slower.
apply() installs the patches once and returns the dict recorded in the unit spec/meta ({} when nothing is set)."""

import os, torch

_state = {"applied": None}


def apply():
    if _state["applied"] is not None:
        return _state["applied"]
    A = {}
    n0 = os.environ.get("LOGN_CLAMP")
    t0 = os.environ.get("ROPE_PI")
    if n0:
        from sdm.nn import scaling as S

        n0 = float(n0)
        orig = S.LogScale.forward

        def fwd(self, query, *, key_len):
            kl = key_len.clamp(max=n0) if isinstance(key_len, torch.Tensor) else min(float(key_len), n0)
            return orig(self, query, key_len=kl)

        S.LogScale.forward = fwd
        A["LOGN_CLAMP"] = n0
    if t0:
        from sdm.nn import rope as R, rms_norm as N

        t0 = float(t0)

        def rfwd(self, x):
            if x.size(-1) != self.channels:
                raise ValueError(f"Expected {self.channels} channels (got {x.size(-1)})")
            T = x.size(-3)
            seq = torch.arange(T, device=x.device, dtype=torch.float32) * min(1.0, t0 / T)
            freq = seq.view(-1, 1) * self.inv_freq.view(1, -1)
            sin = freq.sin()[:, None, :].to(x.dtype)
            cos = freq.cos()[:, None, :].to(x.dtype)
            rot = x[..., : self.rotary_channels]
            if self.layout == "interleaved":
                x1, x2 = rot[..., 0::2], rot[..., 1::2]
            else:
                x1, x2 = rot.split(sin.size(-1), dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            rot = (
                torch.stack((o1, o2), dim=-1).flatten(-2)
                if self.layout == "interleaved"
                else torch.cat((o1, o2), dim=-1)
            )
            return (
                rot
                if self.rotary_channels == self.channels
                else torch.cat((rot, x[..., self.rotary_channels :]), dim=-1)
            )

        R.RotaryEmbedding.forward = rfwd
        orig_n = N.RMSNorm.forward

        def nfwd(self, x, rope=None):
            if rope is not None:
                x = rope(x)
                rope = None  # eager (patched) rotation, then the fused norm; bypasses the table fast path
            return orig_n(self, x, rope)

        N.RMSNorm.forward = nfwd
        A["ROPE_PI"] = t0
    _state["applied"] = A
    if A:
        print(f"km_adapters active: {A}", flush=True)
    return A
