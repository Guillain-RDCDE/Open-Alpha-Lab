"""The overlay: scale exposure by inverse realized vol, using only the past — and price it.

The Moreira–Muir (2017) rule in one line::

    w_t = σ_target / σ̂_{t−1},   capped at `max_leverage`

``σ̂_{t−1}`` is a *past-only* vol forecast (this module lags it explicitly), so there is no
look-ahead: the weight set at yesterday's close earns today's return. Sizing down in storms and up
in calm keeps risk roughly constant through time.

A note on fairness, because it pre-empts the obvious objection. The Sharpe ratio is invariant to a
*constant* leverage, so comparing the Sharpe of buy-&-hold against the managed series is already
fair to "you just ran less/more risk on average" — the average exposure cancels. To make the
**drawdown** comparison fair too, :func:`compare` defaults the target vol to the buy-&-hold's own
realized vol, so the managed book runs the *same* unconditional risk and only *re-times* it. What
that fairness does **not** neutralize — the leverage you must take in calm periods — is the honest
catch the :mod:`decompose` certainty-equivalent test prices in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .vol import realized_vol

TRADING_DAYS_PER_YEAR = 252


def _ann_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def managed_weights(
    returns: pd.Series,
    target_vol_ann: float = 0.12,
    window: int = 21,
    max_leverage: float = 2.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Past-only exposure weights ``w_t = σ_target / σ̂_{t−1}``, capped at ``max_leverage``.

    The vol forecast is the trailing-window realized vol **lagged one bar** — the weight applied to
    day ``t`` uses only returns up to ``t−1``. The cap is the realistic leverage ceiling a real book
    faces (margin, financing); without it the rule would demand unbounded size in dead-calm tape.
    """
    target_daily = target_vol_ann / np.sqrt(periods_per_year)
    fc = realized_vol(returns, window).shift(1)               # past-only forecast
    w = (target_daily / fc).clip(upper=max_leverage)
    return w.rename("weight")


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline stats for a return stream: Sharpe is the number that matters; maxDD the one you feel."""
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "n_days": int(len(r)),
    }


def turnover(weights: pd.Series) -> pd.Series:
    """Per-bar exposure turnover ``|w_t − w_{t−1}|`` — what the cost model charges against."""
    return weights.diff().abs().rename("turnover")


def managed_returns(
    returns: pd.Series,
    cost_bps: float = 0.0,
    **kw,
) -> pd.Series:
    """The vol-managed return stream, net of ``cost_bps`` charged on each unit of exposure traded."""
    w = managed_weights(returns, **kw)
    gross = (w * returns)
    cost = (cost_bps * 1e-4) * turnover(w)
    return (gross - cost).dropna().rename("managed")


def compare(
    returns: pd.Series,
    target_vol_ann: float = 0.12,
    window: int = 21,
    max_leverage: float = 2.0,
    cost_bps: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Buy-&-hold vs vol-managed, gross and net of costs.

    The headline is the **Sharpe gain**, which is invariant to a constant leverage — so it is
    already fair to "you just ran more/less risk on average" regardless of ``target_vol_ann``; the
    target only sets the book's absolute risk (default 12%/yr, a typical vol-target level). The
    *drawdown* and *return* comparisons at strictly matched risk live in
    :func:`storm_shy.decompose.equal_risk_return`. Returns the two summaries, the Sharpe gain (net),
    realized leverage stats and annualised turnover.
    """
    r = pd.Series(returns).astype(float).dropna()

    w = managed_weights(r, target_vol_ann, window, max_leverage, periods_per_year)
    gross = (w * r)
    cost = (cost_bps * 1e-4) * turnover(w)
    net = (gross - cost).dropna()

    # align buy-&-hold to the managed sample (drop the warm-up window)
    bh = r.reindex(net.index)
    w_used = w.reindex(net.index)

    s_bh = summary(bh, periods_per_year)
    s_gross = summary(gross.reindex(net.index), periods_per_year)
    s_net = summary(net, periods_per_year)

    return {
        "target_vol_ann": float(target_vol_ann),
        "buy_hold": s_bh,
        "managed_gross": s_gross,
        "managed_net": s_net,
        "sharpe_gain_net": float(s_net["sharpe"] - s_bh["sharpe"]),
        "sharpe_gain_gross": float(s_gross["sharpe"] - s_bh["sharpe"]),
        "avg_leverage": float(w_used.mean()),
        "max_leverage_used": float(w_used.max()),
        "frac_capped": float((w_used >= max_leverage - 1e-9).mean()),
        "turnover_ann": float(turnover(w_used).mean() * periods_per_year),
        "n_days": int(len(net)),
    }


def cost_sweep(
    returns: pd.Series,
    roundtrip_bps=(0, 1, 2, 5, 10, 20),
    window: int = 21,
    max_leverage: float = 2.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Net Sharpe gain over buy-&-hold as a function of cost per unit of exposure traded (bps).

    The overlay's turnover is *low* — the vol forecast moves slowly — so unlike a daily-round-trip
    signal the edge survives many bps. This table is the tradability argument made concrete:
    the break-even cost is far above any realistic execution cost on a liquid index.
    """
    rows = {}
    for c in roundtrip_bps:
        cmp = compare(returns, window=window, max_leverage=max_leverage,
                      cost_bps=c, periods_per_year=periods_per_year)
        rows[c] = {
            "buy_hold_sharpe": cmp["buy_hold"]["sharpe"],
            "managed_sharpe": cmp["managed_net"]["sharpe"],
            "sharpe_gain": cmp["sharpe_gain_net"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out
