"""Study 573 — Customer-Concentration.

A synthetic-only teardown of the *customer-concentration* fundamental-risk claim: firms that
depend on a handful of large customers carry more supply-chain / demand fragility, so they should
show higher forward risk (volatility) and — if that risk is priced — a return premium (or, in the
behavioural telling, a discount for the fragility).

The desk has no free, point-in-time customer-concentration panel (that lives in Compustat's
segment files / 10-K "major customer" disclosures behind a paywall), so this study is
**synthetic-only** by construction. It can therefore never earn a `REAL` signal — that needs a
robust ``t >= 2`` on a real tape — and is capped at `WEAK`/`NONE`. The data-availability limit is
stated openly on the SIGNAL axis.
"""

from __future__ import annotations

from . import data, strategy

__all__ = ["data", "strategy"]
