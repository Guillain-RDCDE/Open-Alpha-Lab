"""Robustness — the ways "pairs trading still works" can be a mirage in 2026.

The famous GGR numbers come from 1962–2002. Four questions decide whether they survive
to a world that has read the paper:

1. **Decay.** Has the edge faded as the trade got crowded, spreads decimalized, and
   stat-arb desks multiplied? :func:`decay_by_year` runs the *same* rule year by year
   and reads the trend straight off. This is the headline of the modern teardown.

2. **The bid-ask bounce.** GGR's own robustness check: most of the raw profit can be an
   artefact of buying at the bid and selling at the ask on the signal bar. Wait one day
   and it shrinks. :func:`wait_rule_effect` quantifies exactly how much lives in that
   first untradeable day.

3. **Is it even alpha?** Pairs trading is sold as market-neutral. :func:`market_neutrality`
   regresses the portfolio on the tape: a beta near zero is the *good* news here (the
   return isn't disguised market exposure) — which throws the whole weight of the
   verdict onto whether the residual survives costs.

4. **Noise.** A finite sample of long-short trades has a wide Sharpe distribution.
   :func:`bootstrap_sharpe` (the desk's standard, from ``quantlab.stats``) and
   :func:`capacity` (square-root impact) close the loop.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from quantlab.stats import sharpe_ci_bootstrap        # noqa: E402  (desk-standard CI)

from .backtest import CostModel, run                  # noqa: E402
from .pairs import Pair, select_pairs                 # noqa: E402


def decay_by_year(
    panel: pd.DataFrame,
    top_n: int = 20,
    form_len: int = 252,
    trade_len: int = 126,
    k: float = 2.0,
    wait: int = 1,
    costs: CostModel = CostModel(),
) -> pd.DataFrame:
    """Committed-capital monthly net return, bucketed by calendar year of the trade.

    Runs the full rolling backtest once, then groups the daily P&L by year. A monotone
    slide from a healthy early figure toward zero (or red) is the crowding signature —
    the same edge, arbitraged away as the world copied it. Columns: ``monthly_net,
    sharpe, n_days``.
    """
    res = run(panel, top_n=top_n, form_len=form_len, trade_len=trade_len,
              k=k, wait=wait, costs=costs)
    daily = res.daily
    if len(daily) == 0:
        return pd.DataFrame(columns=["monthly_net", "sharpe", "n_days"])
    rows = []
    for year, grp in daily.groupby(daily.index.year):
        sd = grp.std(ddof=1)
        rows.append({
            "year": int(year),
            "monthly_net": float(grp.mean()) * 21,
            "sharpe": float(grp.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "n_days": int(len(grp)),
        })
    return pd.DataFrame(rows).set_index("year")


def wait_rule_effect(
    panel: pd.DataFrame,
    waits=(1, 2, 3, 5),
    top_n: int = 20,
    form_len: int = 252,
    trade_len: int = 126,
    k: float = 2.0,
    costs: CostModel = CostModel(),
) -> pd.DataFrame:
    """Monthly net return vs execution lag — GGR's bid-ask-bounce control.

    ``wait=1`` is the GGR headline: the spread crosses ``k·sigma`` at close ``t``, you
    open there and earn from ``t→t+1``. The worry is that opening on that close means
    buying at the ask / selling at the bid, so the next day's close-to-close partly
    reverses the bounce and *inflates* the measured profit. Waiting extra days
    (``wait=2,3,…`` — GGR's "one day later" robustness) steps past the bounce; a steep
    fade as the lag grows means much of the edge was microstructure, not convergence.
    Columns: ``committed_monthly_net, sharpe_net, mean_trade_net, n_trades``.
    """
    rows = []
    for w in waits:
        res = run(panel, top_n=top_n, form_len=form_len, trade_len=trade_len,
                  k=k, wait=int(w), costs=costs)
        s = res.stats
        rows.append({
            "wait_days": int(w),
            "committed_monthly_net": s.get("committed_monthly_net", np.nan),
            "sharpe_net": s.get("sharpe_net", np.nan),
            "mean_trade_net": s.get("mean_trade_net", np.nan),
            "n_trades": s.get("n_trades", 0),
        })
    return pd.DataFrame(rows).set_index("wait_days")


def market_neutrality(daily: pd.Series, market: pd.Series) -> dict:
    """Regress the pairs portfolio on the tape: ``r_pairs = alpha + beta·r_mkt + eps``.

    For a dollar-neutral book the *good* outcome is ``beta ≈ 0`` — the return is not
    disguised market exposure, so the verdict turns entirely on the cost-survival of the
    residual alpha. Returns daily/annualized alpha, beta and R².
    """
    df = pd.concat([daily.rename("y"), market.rename("x")], axis=1, sort=True).dropna()
    if len(df) < 10:
        return {"alpha_daily_bps": np.nan, "alpha_ann_pct": np.nan, "beta": np.nan, "r_squared": np.nan}
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (beta * x + alpha)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return {
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float(((1 + alpha) ** 252 - 1) * 100),
        "beta": float(beta),
        "r_squared": float(r2),
    }


def bootstrap_sharpe(daily: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap CI for the annualized Sharpe — the desk-standard read on 'could be 0?'."""
    return sharpe_ci_bootstrap(daily, n_boot=n_boot, seed=seed)


def selection_recall(selected: list[Pair], true_pairs) -> float:
    """Fraction of the ground-truth twins recovered in ``selected`` (synthetic check).

    Used offline to assert the selector actually finds the baked-in cointegrated pairs
    before we trust it on real data.
    """
    truth = {frozenset((p.a, p.b)) for p in true_pairs}
    if not truth:
        return float("nan")
    found = {frozenset((p.a, p.b)) for p in selected}
    return len(truth & found) / len(truth)


def capacity(
    dollar_volume: pd.DataFrame,
    trades,
    edge_bps: float,
    impact_coef: float = 0.1,
) -> dict:
    """Per-leg dollar size at which square-root impact alone equals ``edge_bps``.

    ``impact_bps(N) = impact_coef·1e4·sqrt(N / ADV$)`` with ``ADV$`` the median daily
    dollar volume across the traded legs over the sample, solved for ``N``. Unlike the
    micro-cap feed in Study 04, this universe is liquid, so capacity is large — and
    therefore *not* the binding constraint; costs and decay are. Returns ``median_adv_usd,
    capacity_usd_per_leg, edge_bps``.
    """
    names = sorted({t.a for t in trades} | {t.b for t in trades})
    advs = [float(dollar_volume[n].median()) for n in names
            if n in dollar_volume.columns and np.isfinite(dollar_volume[n].median())]
    if not advs or edge_bps <= 0:
        return {"median_adv_usd": np.nan, "capacity_usd_per_leg": np.nan, "edge_bps": edge_bps}
    adv = float(np.median(advs))
    cap = adv * (edge_bps / (impact_coef * 1e4)) ** 2
    return {"median_adv_usd": adv, "capacity_usd_per_leg": cap, "edge_bps": float(edge_bps)}
