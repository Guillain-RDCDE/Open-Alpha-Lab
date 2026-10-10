"""Study 1021 — Stability Breeds Instability.

Hyman Minsky's line was that stability is destabilising: a long run of quiet markets
teaches everyone that risk is low, so they take more of it, and the quiet ends badly.
Danielsson, Valenzuela & Zer (2018) turned the line into a regression — long stretches of
unusually *low* volatility, measured against a slow-moving trend, came before crises years
later. The trading-desk version is shorter: "low VIX means complacency, sell".

The hard part is not finding the correlation. It is that volatility **mean-reverts**, so a
calm today is mechanically followed by a less calm tomorrow even in a world where nobody
ever changes their behaviour. This study builds that world — a GARCH and a long-memory
FIGARCH fitted to the real tape, with no risk-taking feedback at all — and asks whether the
real calm-to-crash link is stronger than what mean reversion alone delivers.

- :mod:`calm.data` — the frozen real tapes (Fama-French monthly total return 1926-2018,
  S&P 500 daily price index 1990-2022, VIX 2014-2018) and the deterministic synthetic
  generator with a planted calm-raises-crash-risk feedback.
- :mod:`calm.strategy` — the calm signal, the forward outcomes, HAC and non-overlapping
  inference, the simulated mean-reversion nulls, the de-risking rule and the verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
