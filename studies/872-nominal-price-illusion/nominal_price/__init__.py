"""Study 872 — Nominal-Price Illusion.

Kumar (2009) & Birru-Wang (2016): the **nominal share price** ($10 vs $500) is a pure
money-illusion characteristic — it says nothing about value — yet retail lottery demand
clusters in low-priced names. We sort a liquid US cross-section on its **nominal price
level** and ask whether cheap-looking names carry the lottery look (higher volatility /
right skew) and **lower risk-adjusted** forward returns.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (cheapness plants
                 the lottery look always, and — with ``edge>0`` — a forward under-earn;
                 null at ``edge=0``).
* ``strategy`` — the price-level signal, the point-in-time cross-sectional sort, the
                 inference primitives (Welch / one-sample / Newey-West HAC / Wilson /
                 placebo), the per-book vol/skew/Sharpe read, and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
