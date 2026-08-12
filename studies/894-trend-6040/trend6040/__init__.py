"""Study 894 — Trend Overlay on 60/40.

A 200-day trend filter laid over the balanced book: hold the 60/40 (SPY/IEF) when
each leg is above its own 200-day moving average, step that leg to cash (BIL)
otherwise. The question is whether the trend overlay cuts drawdown while keeping most
of the return — measured **excess-of-cash vs excess-of-cash** against the static 60/40,
with a HAC *t* on the return difference, a bootstrap Sharpe CI, a drawdown / calendar
table, an era cut, and a net-of-costs-and-tax version.

Offline & deterministic once cached. See ``data`` (tapes + synthetic control) and
``strategy`` (the overlay, benchmark, inference).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
