"""Honest backtest (SOCLE 2) — the buy-at-close / sell-at-open strategy.

The gross overnight edge is real. This module asks the only question that
matters for a retail trader: *does it survive real costs?* Spoiler, encoded in
the maths: a round-trip every single night (~252/year) makes the cost term, not
the alpha, the dominant force. This is exactly why the NSPY / NIWM "night
effect" ETFs were liquidated after a year.

Cost model per *day held overnight*:
    cost_per_day = 2 * (0.5*spread_bps + commission_bps + slippage_bps)
                   + financing_bps_per_night
The factor 2 is the killer: you cross the spread to BUY at the close and again
to SELL at the open. ``financing_bps_per_night`` covers CFD/MT5 swap, which on
a leveraged long can erase the edge on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import TRADING_DAYS_PER_YEAR


def _per_day_cost(spread_bps, commission_bps, slippage_bps, financing_bps_per_night) -> float:
    half_spread = 0.5 * spread_bps
    roundtrip = 2.0 * (half_spread + commission_bps + slippage_bps)
    return (roundtrip + financing_bps_per_night) * 1e-4


def backtest_overnight(
    dec: pd.DataFrame,
    spread_bps: float = 1.0,
    commission_bps: float = 0.5,
    timing_slippage_bps: float = 1.0,
    financing_bps_per_night: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Backtest 'hold overnight every night', net of costs.

    Returns a dict with net daily series and headline stats (CAGR, Sharpe,
    max drawdown). The strategy captures ``r_overnight`` each day minus the
    per-day cost above.
    """
    cost = _per_day_cost(spread_bps, commission_bps, timing_slippage_bps, financing_bps_per_night)
    gross = dec["r_overnight"]
    net = gross - cost

    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    std = net.std(ddof=1)
    sharpe = (net.mean() / std * np.sqrt(periods_per_year)) if std > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0

    return {
        "net_daily": net,
        "equity": equity,
        "cost_bps_per_day": cost * 1e4,
        "cagr_net": float(cagr),
        "sharpe_net": float(sharpe),
        "vol_ann_net": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(drawdown.min()),
        "n_days": int(n),
    }


def breakeven_cost_bps(dec: pd.DataFrame) -> float:
    """Round-trip cost (bps) that exactly annihilates the gross overnight edge.

    Defined as the per-day cost equal to the mean overnight return, expressed as
    a round-trip in bps (i.e. mean_overnight_bps, since our cost model already
    folds the round-trip into one per-day number). If realistic costs exceed
    this, the net strategy is a loser.
    """
    mean_bps = dec["r_overnight"].mean() * 1e4
    return float(mean_bps)


def cost_sweep(
    dec: pd.DataFrame,
    roundtrip_bps=(0, 1, 2, 3, 5, 8),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Net CAGR / Sharpe / maxDD as a function of total round-trip cost (bps).

    Here ``roundtrip_bps`` is the *all-in per-day* cost, so we feed it straight
    through as a commission-equivalent (spread=slippage=financing=0). This is
    the table that tells the whole story: the edge is gone by a few bps.
    """
    rows = {}
    for c in roundtrip_bps:
        res = backtest_overnight(
            dec,
            spread_bps=0.0,
            commission_bps=c / 2.0,  # *2 inside the model -> c bps/day round-trip
            timing_slippage_bps=0.0,
            financing_bps_per_night=0.0,
            periods_per_year=periods_per_year,
        )
        rows[c] = {
            "cagr_net": res["cagr_net"],
            "sharpe_net": res["sharpe_net"],
            "max_drawdown": res["max_drawdown"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "roundtrip_bps"
    return out
