"""Study 815 — Variance-Ratio Reversal.

Lo & MacKinlay (1988): the **variance ratio** ``VR(q) = Var(q-day return) / (q * Var(
1-day return))`` diagnoses departures from a random walk — ``VR < 1`` mean-reverting,
``VR > 1`` trending. We sort a liquid US cross-section on its trailing ``VR(q=5)`` and
measure the forward return of a long-low-VR (mean-reverting) / short-high-VR (trending)
book, asking whether the mean-reverters offer a tradable reversal spread.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 low-VR reversal premium, null at ``edge=0``).
* ``strategy`` — the Lo-MacKinlay overlapping variance-ratio signal, the point-in-time
                 cross-sectional sort, the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
