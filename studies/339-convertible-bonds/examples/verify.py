"""Headline run for Study 339 (Convertible-Bonds) on the real total-return tapes.

Fits CWB's convexity versus SPY (Treynor-Mazuy quadratic, HAC t + block-bootstrap CI on the
convexity coefficient gamma), measures up/down capture, and races CWB against a beta-matched
SPY/AGG blend on excess-of-cash Sharpe (SHY proxy). Prints the numbers the README front-card
and the notebooks quote, and stamps the as-of date + per-asset content fingerprints.

    PYTHONIOENCODING=utf-8 python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from convertible_bonds import data, strategy  # noqa: E402

AS_OF = "2026-05-31"
COST_BPS = 2.0


def _row(name, d):
    print(f"  {name:<26} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
          f"Sharpe(xs) {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%")


def main() -> None:
    prices = data.load_real(("CWB", "SPY", "AGG", "SHY")).loc[:AS_OF]
    rets = strategy.to_returns(prices)
    print(f"[data] total-return panel: {len(prices):,} rows  "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}  as-of {AS_OF}")
    for c in prices.columns:
        print(f"       {c:>4}  fingerprint={data.fingerprint(prices[[c]])}")
    print(f"       JOINT fingerprint={data.fingerprint(prices)}")

    rf = rets["SHY"]

    # --- Is the payoff convex? ---
    reg = strategy.convexity_regression(rets, "CWB", "SPY")
    bg = strategy.bootstrap_gamma(rets, "CWB", "SPY", block=21, n_boot=2000, seed=339)
    print("\n--- Convexity regression: r_CWB = a + b*r_SPY + gamma*max(r_SPY,0)^2 ---")
    print(f"  beta (linear)  = {reg['beta']:+.4f}")
    print(f"  gamma (convex) = {reg['gamma']:+.4f}   HAC t = {reg['t_gamma']:+.2f}   R^2 = {reg['r2']:.3f}")
    print(f"  gamma bootstrap CI [{bg['ci95'][0]:+.3f}, {bg['ci95'][1]:+.3f}]  "
          f"P(gamma>0) = {bg['frac_pos']*100:.0f}%")

    # subsamples
    for lo, hi in (("2009", "2015"), ("2016", "2026")):
        sub = strategy.convexity_regression(rets.loc[lo:hi], "CWB", "SPY")
        print(f"  {lo}-{hi}: gamma {sub['gamma']:+.3f}  t {sub['t_gamma']:+.2f}")

    # --- Capture asymmetry ---
    cap = strategy.capture_ratios(rets, "CWB", "SPY")
    print(f"\n  up-capture {cap['up_capture']:.3f}  down-capture {cap['down_capture']:.3f}  "
          f"asymmetry {cap['asymmetry']:+.3f}")

    # --- The race: beta-matched SPY/AGG blend ---
    beta = strategy.beta_to(rets, "CWB", "SPY")
    w = float(np.clip(beta, 0, 1))
    blend = strategy.matched_blend(rets, "SPY", "AGG", w_stock=w,
                                   rebalance="annual", cost_bps=COST_BPS)
    print(f"\n--- Matched blend ({w:.0%} SPY / {1-w:.0%} AGG), annual rebal, "
          f"{COST_BPS:.0f} bps/rebalance ---")
    _row("CWB (convertibles)", strategy.stats(rets["CWB"], rf=rf))
    _row("matched blend", strategy.stats(blend, rf=rf))
    _row("100% stocks (SPY)", strategy.stats(rets["SPY"], rf=rf))
    _row("AGG (agg bonds)", strategy.stats(rets["AGG"], rf=rf))

    bs = strategy.bootstrap_sharpe_diff(rets["CWB"], blend, rf=rf,
                                        block=21, n_boot=2000, seed=339)
    dr = (rets["CWB"] - blend).dropna()
    print(f"\n  CWB - blend Sharpe (xs) = {bs['point']:+.3f}")
    print(f"  Bootstrap Sharpe diff CI [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]  "
          f"CWB wins {bs['frac_a_wins']*100:.0f}% of resamples")
    print(f"  HAC t on daily (CWB - blend) return diff = {strategy.hac_tstat(dr.to_numpy()):+.2f}")

    # --- The bond floor in the worst month ---
    m = (1 + rets).resample("ME").prod() - 1
    mb = (1 + blend).resample("ME").prod() - 1
    worst = m["SPY"].idxmin()
    print(f"\n  Worst SPY month {worst.date()}:  SPY {m.loc[worst,'SPY']*100:+.1f}%   "
          f"CWB {m.loc[worst,'CWB']*100:+.1f}%   blend {mb.loc[worst]*100:+.1f}%   "
          f"AGG {m.loc[worst,'AGG']*100:+.1f}%")


if __name__ == "__main__":
    main()
