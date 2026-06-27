"""Real-panel verification -- Study 501 (Idiosyncratic-Volatility). Regenerates docs/results.md.

Reads the study's own yfinance cache (studies/501-idiosyncratic-volatility/_cache/), computes
rolling CAPM-residual idiosyncratic vol, sorts into quintiles, builds the low-minus-high
long-short (gross & net of costs + borrow), runs a label-shuffle placebo, and prints every
number that lands in docs/results.md. Also runs the deterministic synthetic positive control.

    python studies/501-idiosyncratic-volatility/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from idiosyncratic_volatility import data, strategy as st  # noqa: E402

COST_BPS = 5.0          # one-way trading cost (bps of NAV x turnover)
BORROW_ANN_BPS = 50.0   # annual borrow on the short (high-IVOL) leg


def synthetic_control() -> None:
    print("=== Synthetic positive control (engine faithfulness) ===")
    print("Plant a known IVOL puzzle; the low-minus-high spread should be monotone in it.")
    for prem in (-0.06, 0.0, 0.06, 0.12):
        prices, spy, truth = data.synthetic_panel(ivol_premium=prem, seed=501)
        res = st.ivol_long_short(prices, spy, min_stocks=8)
        s = st.summary(res["ls_gross"]) if not res.empty else {"mean": float("nan"), "tstat": float("nan")}
        print(f"  planted premium {prem:+.2f} -> low-minus-high {s['mean'] * 100:+6.2f}%/yr "
              f"| HAC t {s['tstat']:+.2f}")
    print()


def main() -> None:
    synthetic_control()

    prices, spy = data.fetch_panel()
    if prices.empty:
        print("Cache not found / network failed. Run from repo root with network to populate _cache/.")
        return

    print("Study 501 -- Idiosyncratic-Volatility -- real yfinance panel\n")
    print(f"Prices panel: {prices.shape}, fingerprint={data.fingerprint(prices)}")
    print(f"SPY series:   {spy.shape}, fingerprint={data.fingerprint(spy)}")
    print(f"Date range:   {prices.index[0].date()} -- {prices.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close prices\n")

    res = st.ivol_long_short(prices, spy, window=252, n_quantiles=5, min_stocks=15,
                             cost_bps=0.0, borrow_ann_bps=0.0)
    res_net = st.ivol_long_short(prices, spy, window=252, n_quantiles=5, min_stocks=15,
                                 cost_bps=COST_BPS, borrow_ann_bps=BORROW_ANN_BPS)
    if res.empty:
        print("Could not construct long-short (too few months).")
        return

    s_gross = st.summary(res["ls_gross"])
    s_net = st.summary(res_net["ls_net"])
    s_low = st.summary(res["low_ivol_ret"])
    s_high = st.summary(res["high_ivol_ret"])
    s_mkt = st.summary(res["mkt_ret"])
    p_placebo = st.placebo_pvalue(res["ls_gross"])

    print("=== IVOL low-minus-high long-short (long Q1 low, short Q5 high) ===")
    print(f"  N months:              {s_gross['n']} "
          f"({res.index[0].date()} -- {res.index[-1].date()})")
    print(f"  GROSS mean (ann):      {s_gross['mean'] * 100:+.2f}%/yr")
    print(f"  GROSS vol (ann):       {s_gross['vol'] * 100:.2f}%/yr")
    print(f"  GROSS Sharpe (ann):    {s_gross['sharpe']:+.3f}")
    print(f"  GROSS HAC t-stat:      {s_gross['tstat']:+.3f}")
    print(f"  GROSS hit rate:        {s_gross['hit_rate'] * 100:.1f}%")
    print(f"  GROSS max drawdown:    {s_gross['max_drawdown'] * 100:.1f}%")
    print(f"  Placebo (sign-flip) p: {p_placebo:.3f}")
    print(f"  NET mean (ann):        {s_net['mean'] * 100:+.2f}%/yr")
    print(f"  NET HAC t-stat:        {s_net['tstat']:+.3f}")
    print(f"  Avg one-sided turnover/mo: {res_net['turnover'].mean():.3f}")
    print(f"  Cost assumption:       {COST_BPS:.0f} bps one-way + {BORROW_ANN_BPS:.0f} bps/yr borrow")

    print("\n=== Legs vs SPY (next-month, annualised) ===")
    print(f"  Low-IVOL leg (Q1):     {s_low['mean'] * 100:+.2f}%/yr")
    print(f"  High-IVOL leg (Q5):    {s_high['mean'] * 100:+.2f}%/yr")
    print(f"  SPY (market):          {s_mkt['mean'] * 100:+.2f}%/yr")
    print(f"  Avg IVOL (low leg):    {res['avg_ivol_low'].mean() * 100:.1f}%/yr")
    print(f"  Avg IVOL (high leg):   {res['avg_ivol_high'].mean() * 100:.1f}%/yr")

    qm = st.quintile_means(prices, spy)
    print("\n=== Quintile mean returns (Q1 low IVOL ... Q5 high IVOL, annualised) ===")
    for q, v in qm.items():
        print(f"  {q}: {v * 100:+.2f}%/yr")
    monotone = bool(np.all(np.diff(qm.dropna().to_numpy()) < 0))
    print(f"  Strictly decreasing (the AHXZ puzzle shape)? {monotone}")

    print("\n=== Robustness: window sweep (gross low-minus-high HAC t) ===")
    for w in (126, 189, 252, 504):
        r_w = st.ivol_long_short(prices, spy, window=w, min_stocks=15)
        if not r_w.empty:
            s_w = st.summary(r_w["ls_gross"])
            print(f"  window {w:3d}d: mean {s_w['mean'] * 100:+6.2f}%/yr | HAC t {s_w['tstat']:+.2f}")

    print("\n=== Robustness: quantile-count sweep (gross HAC t) ===")
    for nq in (3, 5, 10):
        r_q = st.ivol_long_short(prices, spy, n_quantiles=nq, min_stocks=max(12, nq * 2))
        if not r_q.empty:
            s_q = st.summary(r_q["ls_gross"])
            print(f"  {nq:2d} quantiles: mean {s_q['mean'] * 100:+6.2f}%/yr | HAC t {s_q['tstat']:+.2f}")

    print("\nNOTE Survivorship bias: universe = current S&P 500 names projected backwards.")
    print("     Results are upper-bound estimates. IVOL = std of the CAPM RESIDUAL (market beta")
    print("     removed) -- distinct from total-vol low-vol (Study 330). One execution lag:")
    print("     enter one trading day after the signal date. RF = 3%/yr constant (no FRED).")


if __name__ == "__main__":
    main()
