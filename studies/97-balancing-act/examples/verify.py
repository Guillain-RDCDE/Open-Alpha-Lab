"""Headline run for Study 97 (Balancing-Act) on the real total-return tapes.

Builds the fixed 60/40 (SPY/IEF) blend rebalanced annually, pins it against 100% stocks
(SPY) and a TLT variant, computes excess-of-cash Sharpes (SHY proxy), the HAC *t* and
block-bootstrap CI on the Sharpe difference, the rolling stock/bond correlation, and the
crash-cushion test (incl. the 2022 zoom). Prints the numbers the README front-card and the
notebooks quote, and stamps the as-of date + per-asset content fingerprints.

    PYTHONIOENCODING=utf-8 python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from balancing_act import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
COST_BPS = 2.0


def _row(name, d):
    print(f"  {name:<24} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
          f"Sharpe(xs) {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%")


def main() -> None:
    prices = data.load_real(("SPY", "IEF", "TLT", "SHY")).loc[:AS_OF]
    rets = strategy.to_returns(prices)
    print(f"[data] total-return panel: {len(prices):,} rows  "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}  as-of {AS_OF}")
    for c in prices.columns:
        print(f"       {c:>4}  fingerprint={data.fingerprint(prices[[c]])}")
    print(f"       JOINT fingerprint={data.fingerprint(prices)}")

    rf = rets["SHY"]  # cash / excess-of-cash proxy

    bd = strategy.rebalanced_blend(rets, {"SPY": 0.60, "IEF": 0.40},
                                   rebalance="annual", cost_bps=COST_BPS)
    blend = strategy.stats(bd, rf=rf)
    bd_tlt = strategy.rebalanced_blend(rets, {"SPY": 0.60, "TLT": 0.40},
                                       rebalance="annual", cost_bps=COST_BPS)
    blend_tlt = strategy.stats(bd_tlt, rf=rf)
    spy = strategy.stats(strategy.single_asset(rets, "SPY"), rf=rf)

    print("\n--- Real tape (SPY/IEF/TLT total return), annual rebalance, "
          f"{COST_BPS:.0f} bps/rebalance ---")
    _row("60/40 (SPY/IEF)", blend)
    _row("60/40 (SPY/TLT)", blend_tlt)
    _row("100% stocks (SPY)", spy)

    # --- Is the Sharpe improvement real and robust? (excess-of-cash) ---
    t = strategy.hac_tstat((bd - rf).reindex(bd.index).fillna(0.0).to_numpy()
                           - (rets["SPY"] - rf).to_numpy())
    boot = strategy.bootstrap_sharpe_diff(bd, rets["SPY"], rf=rf,
                                          block=21, n_boot=2000, seed=97)
    print(f"\n  60/40 - SPY Sharpe (xs-of-cash) = {blend['sharpe']-spy['sharpe']:+.3f}")
    print(f"  HAC t on daily (60/40 - SPY) xs-ret = {t:+.2f}")
    print(f"  Bootstrap Sharpe diff point={boot['point']:+.3f}  "
          f"95% CI [{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}]  "
          f"60/40 wins {boot['frac_a_wins']*100:.0f}% of resamples")

    # --- The cushion claim: rolling stock/bond correlation across regimes ---
    rc = strategy.rolling_correlation(rets, "SPY", "IEF", window=63).dropna()
    print(f"\n  Rolling 63d SPY/IEF corr: mean {rc.mean():+.2f}  "
          f"min {rc.min():+.2f}  max {rc.max():+.2f}  "
          f"frac days positive {(rc > 0).mean()*100:.0f}%")

    # --- Crash-cushion: did bonds cushion every >10% equity drawdown? ---
    print("\n  Equity drawdowns deeper than -10% (SPY) and the bond leg over the same window:")
    eps = strategy.equity_drawdowns(rets[["SPY", "IEF", "TLT"]], "SPY", thresh=-0.10)
    cushioned = 0
    for e in eps:
        ief = e["others"]["IEF"]
        if ief > 0:
            cushioned += 1
        print(f"    {e['peak'].date()} -> {e['trough'].date()}  "
              f"SPY {e['stock_loss']*100:6.1f}%   IEF {ief*100:+6.1f}%   "
              f"TLT {e['others']['TLT']*100:+6.1f}%")
    print(f"  -> bonds (IEF) rose in {cushioned}/{len(eps)} of the >10% equity crashes")

    # --- The 2022 stress test: did 60/40 protect? ---
    w22 = strategy.window_return(rets[["SPY", "IEF", "TLT"]], "2022-01-01", "2022-12-31")
    b22 = strategy.window_return(bd.to_frame("blend"), "2022-01-01", "2022-12-31")
    print(f"\n  2022 calendar year:  SPY {w22['SPY']*100:+.1f}%   "
          f"IEF {w22['IEF']*100:+.1f}%   TLT {w22['TLT']*100:+.1f}%   "
          f"60/40 {b22['blend']*100:+.1f}%")


if __name__ == "__main__":
    main()
