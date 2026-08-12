"""Study 890 — Sector Risk-Parity.

Cap-weight concentrates the S&P 500 into a few mega-cap sectors. This study equal-**risk**-
weights the eleven GICS sector ETFs (inverse-vol, or full equal-risk-contribution),
rebalances quarterly with realistic weight drift and costs, and races the result against
cap-weight **SPY** — both measured **excess of cash (BIL)** — on Sharpe, drawdown and a
calendar-year table. Diversification *within* equities, not forecasting.

* ``data``     — the real sector panel (yfinance daily total-return closes for the 11 SPDR
                 Select-Sector ETFs + SPY + BIL, cached under the study's own ``_cache/``)
                 plus a deterministic seeded synthetic world (planted vol dispersion, null
                 at ``vol_spread = 0``).
* ``strategy`` — the inverse-vol / ERC / equal-weight allocators, the vectorised quarterly
                 rebalancer with drift and turnover costs, the excess-vs-excess race, the
                 Sharpe-difference bootstrap, the HAC *t*, the levered-to-SPY-vol timer, and
                 the synthetic-control detector.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
