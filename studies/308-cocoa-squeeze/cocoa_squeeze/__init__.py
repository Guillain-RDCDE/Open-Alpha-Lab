"""Study 308 — Cocoa-Squeeze: post-mortem of the 2024 cocoa (CC=F) parabola.

After cocoa futures went vertical in 2024, was there a *tradable edge* around the
blow-off — momentum-continuation as it climbed, or mean-reversion as it cracked — or
was the whole thing an unrepeatable, untradable freak? We test both legs with honest
inference, against a deterministic synthetic blow-off control that plants a known
truth (a bubble that *does* mean-revert vs a pure random walk).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
