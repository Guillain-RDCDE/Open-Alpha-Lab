"""Real-tape verification -- Study 260 (Margin-Debt). Regenerates docs/results.md numbers.

Joins the hardcoded NYSE/FINRA December margin-debt series with month-end S&P 500
closes (price-only, from the repo cache), then tests the contrarian claim that
year-over-year margin growth forecasts a *lower* next-12m return: a HAC OLS
regression, high/low YoY terciles, a fresh-record-high event test, sub-period
robustness, a tradable timing rule vs buy-and-hold, and a synthetic positive
control. Network is touched only with --fetch.

    python studies/260-margin-debt/examples/verify.py          # cache-only
    python studies/260-margin-debt/examples/verify.py --fetch  # refresh ^GSPC
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from margin_debt import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    panel = data.build_panel(fetch=fetch)
    print(f"Panel: {len(panel)} forward-year observations "
          f"({panel['year'].min()} -> {panel['year'].max()})")
    print(f"Fingerprint (fwd_ret): {data.fingerprint(panel)}\n")

    # ---- Predictive regression -------------------------------------------
    reg = st.predictive_regression(panel)
    print("=== Contrarian predictive regression (fwd 12m ret ~ std YoY margin) ===")
    print(f"  slope (per +1 sigma): {reg['beta']*100:+.2f}%   HAC t = {reg['beta_t']:+.2f}")
    print(f"  R^2 = {reg['r2']:.3f}   n = {reg['n']}")
    print(f"  -> contrarian sign: {'YES' if reg['beta'] < 0 else 'NO'};  "
          f"clears |t|>=2: {'YES' if abs(reg['beta_t']) >= 2 else 'NO'}")

    # ---- Terciles --------------------------------------------------------
    ter = st.tercile_forward_returns(panel)
    print("\n=== YoY margin-growth terciles (mean forward 12m return) ===")
    print(f"  Low (contracting):  {ter['low_mean']*100:+.1f}%  (n={ter['n_low']})")
    print(f"  Mid:                {ter['mid_mean']*100:+.1f}%  (n={ter['n_mid']})")
    print(f"  High (surging):     {ter['high_mean']*100:+.1f}%  (n={ter['n_high']})")
    print(f"  High - Low spread:  {ter['spread_high_minus_low']*100:+.1f}%  "
          f"Welch t = {ter['welch_t']:+.2f}")

    # ---- Record-high event ----------------------------------------------
    rec = st.record_high_event(panel)
    print("\n=== Fresh-record-high event test ===")
    print(f"  Record years:     {rec['record_mean']*100:+.1f}%  up-rate {rec['record_up_rate']:.1%}  (n={rec['n_record']})")
    print(f"  Non-record years: {rec['nonrecord_mean']*100:+.1f}%  (n={rec['n_nonrecord']})")
    print(f"  Unconditional up-rate: {rec['uncond_up_rate']:.1%}")
    print(f"  Difference (rec-non): {rec['diff']*100:+.1f}%  Welch t = {rec['welch_t']:+.2f}")

    # ---- Sub-periods -----------------------------------------------------
    print("\n=== Sub-period robustness (slope, HAC t) ===")
    for label, lo, hi in [("1960-1989", 1960, 1989),
                          ("1990-2009", 1990, 2009),
                          ("2010-2024", 2010, 2024)]:
        sub = panel[(panel["year"] >= lo) & (panel["year"] <= hi)]
        r = st.predictive_regression(sub)
        print(f"  {label}: slope={r['beta']*100:+.2f}%  HAC t={r['beta_t']:+.2f}  n={r['n']}")

    # ---- Tradable rule ---------------------------------------------------
    bt = st.contrarian_timing_backtest(panel, surge_z=1.0, one_way_bps=10.0)
    print("\n=== Tradable contrarian timing rule (cash after surge) vs buy-and-hold ===")
    print(f"  Buy-and-hold (price-only):     {bt['bah_ann']*100:+.2f}%/yr")
    print(f"  Contrarian rule (net 10bps):   {bt['strat_net_ann']*100:+.2f}%/yr")
    print(f"  Years in cash: {bt['years_in_cash']}/{bt['n_years']}   "
          f"Shortfall: {bt['shortfall_net_pp']:+.2f}pp/yr")

    # ---- Positive control ------------------------------------------------
    print("\n=== Positive control (synthetic planted contrarian signal) ===")
    for pb in [0.0, 0.03, 0.06]:
        df, _ = data.synthetic_series(n_years=400, predictive_beta=pb, seed=260)
        df = df.dropna(subset=["yoy"]).copy()
        df["fwd_ret"] = df["index_ret"].shift(-1)
        df = df.dropna(subset=["fwd_ret"])
        r = st.predictive_regression(df)
        print(f"  predictive_beta={pb:.2f}: slope={r['beta']:+.4f}  HAC t={r['beta_t']:+.2f}")

    print(f"\nFingerprint: {data.fingerprint(panel)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
