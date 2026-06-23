"""Study 382 — Treasury cash-futures Basis Trade (free carry, or hidden leverage?).

The hedge-fund Treasury **basis trade**: buy the *cheap* cash Treasury, sell the
*rich* Treasury future, and harvest the convergence — the net **carry** (coupon income
minus repo financing cost) — at extreme leverage (often 50–100× via repo). The pitch is
"free carry": a duration-hedged, almost-riskless spread that prints money every day. The
nightmare is a **funding shock** (a repo spike / margin spiral, e.g. **March 2020**), when
the basis dislocates, margin is called, and the levered book is force-unwound into the
move.

True cash-vs-futures bond prices and repo rates are not cleanly on yfinance, so we build a
**transparent carry / implied-repo model** (labelled a model everywhere): the net carry is
the 10-year yield (``^TNX``) minus a short funding rate (``^IRX``, the 13-week T-bill as a
repo proxy), and the residual mark-to-market is a small-duration basis wobble proxied by
the daily change in that spread. The decisive finding is **not directional**: the carry is
statistically real (HAC *t* ≫ 2) and its *unlevered* Sharpe is high — but Sharpe is
**leverage-invariant**, so the same series at the leverage the trade actually runs turns a
silky 1.5%/yr into a fat-tailed, –75%-drawdown short-volatility position. "Free carry" is a
leverage illusion.

See :mod:`treasury_basis_trade.data` (real-tape loader + deterministic synthetic control)
and :mod:`treasury_basis_trade.strategy` (net-carry construction, levered P&L, HAC / Sharpe
inference, placebo null, tail diagnostics, funding-shock stress, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
