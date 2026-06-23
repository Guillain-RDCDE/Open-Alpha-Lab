"""Study 369 — Earnings-Revision-Momentum (do up-revised stocks keep beating the market?).

The believers' claim: stocks whose earnings estimates are being revised *up* keep beating the
market — the analyst-revision / post-earnings-announcement-drift (SUE) factor. Paid analyst
revision feeds (IBES) are not free, so we build a transparent **proxy** from yfinance: a
stock's realized **earnings surprise** (reported EPS beats the consensus estimate) plus the
**q/q change in the consensus estimate**. Each earnings quarter we rank the basket by this
revision score, go long the top tercile / short the bottom tercile, hold ~a quarter, and
measure each leg's return *in excess of SPY*, then confront the long-short spread with a
Welch t, a label-shuffle placebo null, one-way costs, and short-leg borrow.

See :mod:`earnings_revision_momentum.data` (real price + earnings loaders and a deterministic
synthetic control) and :mod:`earnings_revision_momentum.strategy` (revision score, tercile
book, forward-excess returns, inference, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
