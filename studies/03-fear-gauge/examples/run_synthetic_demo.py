"""Offline demo — the whole Fear-Gauge pipeline on synthetic data, no network.

Builds a toy market (upward drift + clustered vol) and a toy VIX that spikes when
the market falls, then runs the full gauntlet: triggers -> event study ->
random-day null -> the cross-study control vs a price drop -> block bootstrap ->
window sensitivity -> the martingale ruin sim -> a cost-charged backtest.

The numbers are meaningless (random data); the point is that every piece wires
together and the *method* runs end-to-end before you ever touch Yahoo.

    python examples/run_synthetic_demo.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fear_gauge import backtest, benchmark, data, eventstudy, exits, robustness, triggers

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def synth(n=6700, seed=0):  # ~2000 -> ~2026 in business days, so the 2016+ window fills
    rng = np.random.default_rng(seed)
    # clustered vol: a slow-moving sigma so down-runs cluster like real crises
    sigma = 0.008 + 0.010 * np.abs(np.sin(np.linspace(0, 30, n))) ** 2
    r = 0.0003 + sigma * rng.standard_normal(n)
    close = 1000.0 * np.cumprod(1.0 + r)
    prev = np.concatenate([[1000.0], close[:-1]])
    open_ = prev * (1 + 0.002 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + np.abs(0.003 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1 - np.abs(0.003 * rng.standard_normal(n)))
    idx = pd.bdate_range("2000-01-01", periods=n)
    mkt = data.daily_returns(
        pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)
    )
    level = 14.0 + 900.0 * np.clip(-r, 0.0, None)
    level = pd.Series(level, index=idx).rolling(3, min_periods=1).max()
    vix = pd.DataFrame({"Close": level})
    vix["vix_prev"] = vix["Close"].shift(1)
    vix["vix_chg"] = vix["Close"] / vix["vix_prev"] - 1.0
    return mkt, vix


def main():
    mkt, vix = synth()
    print(f"synthetic sample: {len(mkt):,} bars\n")

    sig30 = triggers.first_crossings(triggers.level(vix, 30), cooldown=21)
    print(f"V1 (VIX>=30) fresh events: {int(sig30.sum())}")

    es = eventstudy.event_study(mkt, sig30, horizon=21)
    print("\n[event study] forward path summary (rel days 1/5/21):")
    print(es["summary"].loc[[1, 5, 21]].round(4))

    print("\n[random-day null]")
    print(benchmark.conditional_vs_unconditional(mkt, sig30, n_iter=500).round(4))

    print("\n[cross-study control] VIX>=30 vs a price -3% close")
    price = triggers.first_crossings(mkt["r_cc"] <= -0.03, cooldown=21)
    print(benchmark.excess_vs_alternative(mkt, sig30, price, n_iter=500).round(4))

    print("\n[block bootstrap] h=21")
    print({k: round(v, 4) for k, v in
           robustness.block_bootstrap_excess(mkt, sig30, n_iter=500).items()})

    print("\n[window sensitivity] h=21")
    print(robustness.window_sensitivity(mkt, sig30).round(4))

    print("\n[martingale ruin] buy 30, double 50, hold ~quarter")
    print(pd.Series(robustness.martingale_ruin(mkt, vix)).round(4))

    print("\n[backtest] VIX>=30, hold 21d, realistic costs")
    res = backtest.run(mkt, sig30, exits.ExitRule(max_hold=21), backtest.CostModel())
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.stats.items()
           if k in ("n_trades", "cagr", "sharpe", "max_drawdown", "win_rate", "exposure")})

    print("\nOK — full pipeline ran offline.")


if __name__ == "__main__":
    main()
