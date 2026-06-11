"""The heart — what financing a levered position *really* costs.

Two clean financing models for the SAME exposure path. Each corresponds to a real account
type, and each charges every dollar exactly once — no cost is forgotten, none is double-counted:

  * ``mode="margin"`` — a **cash/margin account**. You hold ``e`` units of the index outright;
    only the slice *above* your capital, ``max(e − 1, 0)``, is borrowed, at the short rate plus
    the broker's markup; whatever capital is *not* deployed, ``max(1 − e, 0)``, sits in bills
    earning the short rate. Dividends accrue to the long. This is the study's **central case**.

        r = e·(ret + div) − (e−1)⁺·(rf + markup) + (1−e)⁺·rf − spread·|Δe|

  * ``mode="futures"`` — a **futures/CFD account**. The position is synthetic: the *entire*
    notional implicitly carries the financing rate (that is the cost-of-carry baked into a
    future's or CFD swap's pricing — Hull), and the broker's markup is charged on the **whole
    notional** too, not just the borrowed slice. Meanwhile your actual capital never leaves
    the bill account, so it earns ``rf`` in full:

        r = rf + e·(ret + div − rf − markup) − spread·|Δe|

**The identity that keeps both honest:** at ``markup = 0`` the two models coincide *exactly*,
for every exposure level (substitute and check — both reduce to ``e·(ret+div) + (1−e)·rf``).
Funding itself is symmetric; what separates the two accounts is only **where the markup
lands**: on the borrowed slice ``(e−1)⁺`` in a margin account, on the full notional ``e`` in
a CFD. That difference — ``min(e, 1)·markup`` per year — is the broker's *house edge*, and it
is the whole study.

(The bug this module replaces: an earlier version charged ``rf + markup`` on the full notional
*and* credited cash interest only on ``(1−e)⁺`` — paying the risk-free rate twice on the
deployed fraction. No real account does both; the phantom drag was ≈ ``avg_e × rf`` ≈ 2.7
pts/yr on the S&P run, which is exactly the deficit it showed at a zero markup.)

`net_returns` returns the daily net return of holding ``exposure`` (decided at the close,
applied to the next day's index move), so strategies are compared on identical exposure with
only the account type and markup changing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def net_returns(exposure: pd.Series, price: pd.Series, short_rate: pd.Series,
                mode: str = "margin", markup: float = 0.025, div_yield: float = 0.018,
                spread_bps: float = 2.0) -> pd.Series:
    """Daily net return of holding ``exposure`` in the index, under the chosen account type.

    ``mode="margin"`` (central case) borrows only ``max(exposure − 1, 0)`` at ``short_rate +
    markup`` and parks un-deployed capital ``max(1 − exposure, 0)`` in bills at ``short_rate``.
    ``mode="futures"`` carries the FULL notional at ``short_rate + markup`` (the cost-of-carry
    of a synthetic position) while the capital itself earns ``short_rate``. Both credit
    ``div_yield`` on the long and charge ``spread_bps`` per unit of turnover; both collapse to
    the same series at ``markup = 0``.
    """
    ret = price.pct_change()
    e = exposure.reindex(price.index).shift(1).clip(lower=0).fillna(0.0)   # decided at prior close
    turn = e.diff().abs().fillna(0.0)
    rf = short_rate.reindex(price.index).ffill().fillna(0.0) / TRADING_DAYS
    mk = markup / TRADING_DAYS
    asset = e * (ret + div_yield / TRADING_DAYS)                           # the long, dividends in
    if mode == "margin":
        borrow = (e - 1).clip(lower=0) * (rf + mk)        # finance only the borrowed slice
        cash = (1 - e).clip(lower=0) * rf                  # idle capital earns the bill
        net = asset - borrow + cash
    elif mode == "futures":
        net = rf + asset - e * (rf + mk)                   # full-notional carry; capital in bills
    else:
        raise ValueError("mode must be 'margin' or 'futures'")
    spread = turn * spread_bps * 1e-4
    return (net - spread).dropna().rename(f"net_{mode}")


def buy_and_hold(price: pd.Series, div_yield: float = 0.018) -> pd.Series:
    """A plain, un-levered total-return index hold (the honest benchmark): price return + dividends."""
    return (price.pct_change() + div_yield / TRADING_DAYS).dropna().rename("buy_and_hold")


def house_edge(exposure: pd.Series, price: pd.Series, short_rate: pd.Series,
               markup: float = 0.025, **kw) -> dict:
    """The broker's take — what the financing *markup* costs, by account type.

    Same exposure path, three fundings: flat (markup zero — what index futures roll near, and
    where both models coincide), a margin account at ``markup`` (charged on the borrowed slice
    only), and a CFD at ``markup`` (charged on the full notional). ``house_edge_ann`` is the
    flat-vs-CFD CAGR gap — the annualised rent the retail account pays the house, ≈
    ``avg_exposure × markup``.
    """
    from .strategy import summary
    flat = summary(net_returns(exposure, price, short_rate, mode="futures", markup=0.0, **kw))
    marg = summary(net_returns(exposure, price, short_rate, mode="margin", markup=markup, **kw))
    cfd = summary(net_returns(exposure, price, short_rate, mode="futures", markup=markup, **kw))
    return {"cagr_funded_flat": flat["cagr"], "cagr_margin": marg["cagr"], "cagr_cfd": cfd["cagr"],
            "house_edge_ann": float(flat["cagr"] - cfd["cagr"]),
            "margin_edge_ann": float(flat["cagr"] - marg["cagr"]),
            "avg_exposure": float(exposure.reindex(price.index).shift(1).clip(lower=0).fillna(0).mean())}
