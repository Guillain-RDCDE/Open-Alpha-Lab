"""Study 344 — Backtest-Overfitting: why every backtest is gorgeous and every live run disappoints.

A *research-method* demo, not a tradable signal. We build a market with **no edge at all**
(pure noise, plus one variant with a tiny real edge as a positive control), let an
optimiser sweep a grid of strategy configurations, and watch it "discover" a beautiful
in-sample Sharpe — that then collapses out-of-sample. We quantify the trap with the two
tools from Bailey & López de Prado: the **Deflated Sharpe Ratio** (haircut the best Sharpe
for the number of trials) and the **Probability of Backtest Overfitting** (PBO, via
Combinatorially-Symmetric Cross-Validation, CSCV).

The verdict lives on the *third axis* — a methodology myth-check — not on Signal/Tradability
(there is, by construction, nothing to trade)."""

from . import data, strategy

__all__ = ["data", "strategy"]
