"""Study 356 — the "Ozempic trade": a GLP-1 / weight-loss-drug basket (LLY + NVO).

Tests the viral theme — *buy Eli Lilly (LLY) and Novo Nordisk (NVO) to ride the
weight-loss-drug boom* — against SPY and the healthcare sector (XLV). The headline
basket beats both on raw return, but:

* the **excess** over SPY is not significant (HAC-t ~ 1.1) and over XLV only borderline
  (HAC-t ~ 2.0, a bootstrap CI that nearly touches zero) — Signal is **WEAK**;
* the alpha is **misattributed**: it is essentially *all LLY* (HAC-t ~ 2.7) while NVO —
  the actual Ozempic maker — adds ~zero alpha (HAC-t ~ 0.3) and sits ~70% below its peak;
* it is a **concentrated two-name bet** with a ~49% drawdown, and the excess *flips sign*
  (negative) during the very GLP-1 mania the theme is named for — Tradability is a **MIRAGE**.

See :mod:`glp1_basket.data` (real yfinance loader + a deterministic synthetic positive
control) and :mod:`glp1_basket.strategy` (excess-return / market-model inference, the
single-name decomposition, the recency split, and the cost/concentration teardown)."""

from . import data, strategy

__all__ = ["data", "strategy"]
