"""Study 816 — Drawdown Duration.

The question: does the **fraction of the trailing year a name spent underwater** — its
cumulative total return sitting **below its running high-water mark** — predict its
cross-section of forward returns? A high time-underwater is *persistent-drawdown* risk.
Does the market **pay** for bearing persistent-drawdown names (a risk premium), or do
they just **keep sinking**? We sort a liquid US cross-section on its trailing-year
time-underwater and measure the forward return of a long-high-underwater /
short-low-underwater book. Honest sign — we report whatever the tape gives.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 time-underwater -> forward-return relation, null at ``knob=0``).
* ``strategy`` — the high-water-mark time-underwater signal, the point-in-time
                 cross-sectional sort, the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
