"""Study 359 — Farmland (the billionaires' "secret inflation hedge").

Tests the "farmland is the best inflation hedge" claim against tradable evidence. The
celebrated object is the **NCREIF Farmland Index** — a quarterly, *appraisal-based* total
return series with famously smooth, low-volatility, near-zero-correlation numbers (and the
folklore that Bill Gates is the largest private US farmland owner). The only *tradable*
farmland exposure for a retail investor is two small listed REITs — **LAND** (Gladstone Land)
and **FPI** (Farmland Partners) — plus broad real-asset funds.

We pull LAND, FPI and SPY from yfinance and contrast them with a HARDCODED, publicly-cited
NCREIF farmland annual-return series (~1992-2023). The punchline: the smooth appraisal index
is an *illusion* — appraisal smoothing autocorrelates and shrinks measured volatility and beta;
once you un-smooth it (Geltner / Fisher-Geltner) or look at the volatile listed proxies, the
"uncorrelated, low-risk, best inflation hedge" story is far weaker. Real low-beta diversifier
(WEAK signal), but a leveraged + illiquid vehicle to actually hold (FRAGILE tradability).

See :mod:`farmland.data` (yfinance loader + hardcoded NCREIF series + deterministic synthetic
appraisal-smoothing control) and :mod:`farmland.strategy` (beta/inflation-beta regressions,
un-smoothing, t-stats vs proper nulls, cost-aware tradability).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
