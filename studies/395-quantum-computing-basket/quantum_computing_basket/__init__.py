"""Study 395 — Quantum-Computing-Basket (the "next big thing" hype-cycle trade).

The pitch: hold the four pure-play quantum-computing stocks — IONQ (trapped-ion), RGTI
(superconducting), QBTS (annealing), QUBT (photonic) — equal-weight and ride the next
technological revolution instead of the boring index. On the realised 2021-2026 tape that basket's
*raw* CAGR is spectacular (and so is its drawdown), which is exactly what makes it a hype-cycle
case study rather than a strategy.

We race the equal-weight pure-play basket against the market (SPY), the tech tape (QQQ), and the
*diversified* quantum ETF (QTUM) — the sane way to "play quantum." The decisive findings are
risk-adjusted, not directional: the basket's enormous raw spread carries an ~80%-class drawdown and
a triple-digit volatility, its Sharpe is *below* the market's and far below the diversified ETF's,
and the HAC t-stat of the spread fails the t >= 2 bar — so the "next big thing" reads as a
high-volatility lottery whose CAGR is a small-sample, survivorship-tilted mirage, not a tradable
edge. (Same thematic-basket family as Study 393 (AI-Datacenter-Basket) and Study 334
(ARK-Innovation); here the critique is risk & sample size rather than ex-post name selection.)

See :mod:`quantum_computing_basket.data` (real panel loader + deterministic synthetic control) and
:mod:`quantum_computing_basket.strategy` (basket engine, the Sharpe/drawdown race, HAC inference,
per-name decomposition, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
