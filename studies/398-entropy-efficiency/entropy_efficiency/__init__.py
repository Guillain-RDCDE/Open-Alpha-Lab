"""Study 398 — Entropy-Efficiency 🎲 (the "efficiency clock").

A statistical-mechanics framing of market timing: compute a rolling **entropy** of recent
returns — *permutation entropy* (how disordered the ordinal pattern of returns is) and
*sample entropy* (how unpredictable / self-similar the series is). The folklore claim is
that when entropy **drops** — the tape becomes more "ordered", more predictable, less random
— a tradable window opens: forward returns should be higher and more forecastable.

We compute the entropy clock on SPY, sort days into low- vs high-entropy regimes, and ask the
only two questions that matter: does *next-day* (and next-week) SPY return differ across
regimes by more than luck, and is any difference tradable net of costs? The decisive object is
not the point estimate but its significance under HAC-robust inference and a block-bootstrap
null, because entropy is a smooth, highly autocorrelated series — naive t-stats over
overlapping windows wildly overstate confidence.

See :mod:`entropy_efficiency.data` (real SPY loader + a deterministic synthetic control that
toggles between random and *structured* — low-entropy — regimes with a PLANTED forward edge)
and :mod:`entropy_efficiency.strategy` (permutation/sample entropy, regime split, HAC t /
block-bootstrap null, win-rate vs base rate, 1-day execution lag, one-way costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
