"""Steelman the manipulator, then let the economics destroy it.

Knuteson alleges a large firm ("M") expands its book when the market is illiquid
(near the open, pushing prices up) and contracts it when liquid — harvesting the
overnight drift it helps create. We build that firm's P&L and ask the only
question that matters: **can it actually make money at the scale required to
move world markets?**

The economics that decide it (Huberman & Stanzl 2004; Gatheral 2010):

* **Permanent** impact must be linear in size for markets to admit no price
  manipulation — and a *round trip* (buy to establish, sell to unwind the same
  quantity) reverses the firm's own permanent push, netting ~0 from it.
* **Temporary** impact is paid on *both* legs and is never recovered.

So a firm running a daily round trip to harvest the overnight is left paying the
round-trip temporary impact every day. Granting it even captures the full
overnight drift on its capital, net P&L is

    net = capital * (overnight_drift - 2 * temp_coef * daily_vol * sqrt(capital/ADV$))

which is positive only for small capital and turns sharply negative at size.
This is the capacity argument, reframed as the manipulator's own ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import TRADING_DAYS_PER_YEAR


def manipulator_pnl_per_day(
    capital_usd: float,
    overnight_drift_bps: float,
    adv_usd: float,
    daily_vol_bps: float,
    temp_coef: float = 1.0,
) -> dict:
    """Expected daily P&L (USD) of the alleged round-trip overnight harvester.

    Grants the firm the *entire* overnight drift on its capital (a generous
    steelman), then charges round-trip temporary impact under the square-root
    law. Permanent impact nets to ~0 over the symmetric round trip.
    """
    participation = capital_usd / adv_usd
    drift = overnight_drift_bps * 1e-4
    temp_rt = 2.0 * temp_coef * (daily_vol_bps * 1e-4) * np.sqrt(participation)
    gross = capital_usd * drift
    impact = capital_usd * temp_rt
    net = gross - impact
    return {
        "capital_usd": capital_usd,
        "participation": participation,
        "gross_usd": gross,
        "impact_usd": impact,
        "net_usd": net,
        "net_bps_on_capital": (net / capital_usd) * 1e4 if capital_usd else np.nan,
    }


def pnl_vs_capital(
    overnight_drift_bps: float,
    adv_usd: float,
    daily_vol_bps: float,
    sizes_usd=(1e6, 1e7, 1e8, 1e9, 1e10, 1e11),
    temp_coef: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Sweep the manipulator's P&L over deployed capital.

    Returns a DataFrame indexed by capital with net bps on capital, net $/day
    and annualised net $/year. The break-even capital and the (negative) P&L at
    "market-moving" scale fall straight out.
    """
    rows = {}
    for k in sizes_usd:
        r = manipulator_pnl_per_day(k, overnight_drift_bps, adv_usd, daily_vol_bps, temp_coef)
        rows[float(k)] = {
            "participation": r["participation"],
            "net_bps_on_capital": r["net_bps_on_capital"],
            "net_usd_per_day": r["net_usd"],
            "net_usd_per_year": r["net_usd"] * periods_per_year,
        }
    out = pd.DataFrame(rows).T
    out.index.name = "capital_usd"
    return out


def breakeven_capital(
    overnight_drift_bps: float, adv_usd: float, daily_vol_bps: float, temp_coef: float = 1.0
) -> float:
    """Largest capital at which the manipulator still breaks even.

    Solve drift = 2*temp_coef*vol*sqrt(K/ADV$)  ->  K = ADV$ * (drift/(2*temp_coef*vol))^2.
    """
    drift = overnight_drift_bps * 1e-4
    vol = daily_vol_bps * 1e-4
    return float(adv_usd * (drift / (2.0 * temp_coef * vol)) ** 2)


def monte_carlo_pnl(
    capital_usd: float,
    overnight_drift_bps: float,
    overnight_vol_bps: float,
    adv_usd: float,
    daily_vol_bps: float,
    n_days: int = TRADING_DAYS_PER_YEAR,
    n_paths: int = 2000,
    temp_coef: float = 1.0,
    seed: int = 0,
) -> dict:
    """Distribution of one year of the manipulator's net P&L at a fixed capital.

    Overnight returns are drawn ~ N(drift, vol); impact is the deterministic
    round-trip temporary cost. Shows that even the *upside* is capped: at
    market-moving scale the whole distribution sits below zero.
    """
    rng = np.random.default_rng(seed)
    participation = capital_usd / adv_usd
    temp_rt = 2.0 * temp_coef * (daily_vol_bps * 1e-4) * np.sqrt(participation)
    draws = rng.normal(overnight_drift_bps * 1e-4, overnight_vol_bps * 1e-4, size=(n_paths, n_days))
    daily_net = capital_usd * (draws - temp_rt)
    yearly = daily_net.sum(axis=1)
    return {
        "capital_usd": capital_usd,
        "participation": participation,
        "mean_usd_per_year": float(yearly.mean()),
        "p05_usd_per_year": float(np.percentile(yearly, 5)),
        "p95_usd_per_year": float(np.percentile(yearly, 95)),
        "prob_profit": float((yearly > 0).mean()),
    }
