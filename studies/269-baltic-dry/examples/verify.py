"""Real-tape verification -- Study 269 (Baltic-Dry). Regenerates docs/results.md numbers.

Pairs the curated monthly Baltic Dry Index with monthly S&P 500 closes (Shiller
cache), builds the 3-month BDI momentum signal, runs the predictive regression
(HAC t on the slope), the long-when-BDI-rising timing overlay vs buy-and-hold,
sub-periods, an ex-2008/09 robustness cut, the sign hit-rate, and a synthetic
positive control.  Network is touched only with --fetch (yfinance ^GSPC, and
--proxy for the BDRY freight ETF).

    python studies/269-baltic-dry/examples/verify.py            # cache-only
    python studies/269-baltic-dry/examples/verify.py --fetch    # refresh ^GSPC
    python studies/269-baltic-dry/examples/verify.py --fetch --proxy  # + BDRY proxy
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from baltic_dry import data, strategy as st  # noqa: E402


def main(fetch: bool, use_proxy: bool) -> None:
    df = data.fetch_real(fetch=fetch, use_proxy=use_proxy)
    print(f"Tape: {df.shape[0]} months  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Fingerprint (last row): {data.fingerprint(df)}\n")

    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)

    # ---- Predictive regression -------------------------------------------
    reg = st.predictive_regression(sig, fwd)
    print("=== Predictive regression: next-month S&P return on 3M BDI momentum ===")
    print(f"  beta = {reg['beta']:+.4f}   HAC t = {reg['tstat']:+.2f}   R2 = {reg['r2']:.4f}   n = {reg['n']}")

    # ---- ex-2008/09 robustness -------------------------------------------
    mask = ~((df.index.year >= 2008) & (df.index.year <= 2009))
    rex = st.predictive_regression(sig[mask], fwd[mask])
    print(f"  ex-2008/09 crisis: beta = {rex['beta']:+.4f}  HAC t = {rex['tstat']:+.2f}  n = {rex['n']}")

    # ---- Sub-periods ------------------------------------------------------
    print("\n=== Sub-period regression (slope on BDI momentum) ===")
    for label, a, b in [("1985-1999", "1985", "1999"),
                        ("2000-2009", "2000", "2009"),
                        ("2010-2026", "2010", "2026")]:
        r = st.predictive_regression(sig[a:b], fwd[a:b])
        print(f"  {label}: beta={r['beta']:+.4f}  HAC t={r['tstat']:+.2f}  n={r['n']}")

    # ---- Timing overlay vs buy-and-hold ----------------------------------
    tim = st.timing_returns(sig, fwd, one_way_bps=5.0)
    s_bh = st.summarize(tim["buy_hold"])
    s_g = st.summarize(tim["timing_gross"])
    s_n = st.summarize(tim["timing_net"])
    exc = (tim["timing_net"] - tim["buy_hold"]).dropna()
    s_e = st.summarize(exc)
    n_switch = int((tim["position"].diff().abs() > 0).sum())

    print("\n=== Timing overlay (long S&P when BDI rising, else cash) vs buy-and-hold ===")
    print(f"  BUY-HOLD     : {s_bh['mean']*12*100:+.2f}%/yr  SR(ann)={s_bh['sharpe']*np.sqrt(12):+.2f}  HAC t={s_bh['tstat']:+.2f}  maxDD={s_bh['max_drawdown']*100:.1f}%")
    print(f"  TIMING gross : {s_g['mean']*12*100:+.2f}%/yr")
    print(f"  TIMING net   : {s_n['mean']*12*100:+.2f}%/yr  SR(ann)={s_n['sharpe']*np.sqrt(12):+.2f}  HAC t={s_n['tstat']:+.2f}  maxDD={s_n['max_drawdown']*100:.1f}%")
    print(f"  TIMING - BH  : {s_e['mean']*12*100:+.2f}%/yr  HAC t={s_e['tstat']:+.2f}  (negative = overlay LOSES)")
    print(f"  position switches: {n_switch}   avg cost: {tim['cost'].mean()*10000:.2f} bps/mo")

    # ---- Sign hit-rate ----------------------------------------------------
    hit = st.sign_hit_rate(sig, fwd)
    print(f"\n=== Sign hit-rate ===")
    print(f"  sign(BDI) == sign(next S&P): {hit['hit']:.3f}  vs base rate (market up): {hit['base_rate']:.3f}")

    # ---- Synthetic positive control --------------------------------------
    print("\n=== Synthetic positive control (machinery recovers a planted lead-lag) ===")
    for beta in [0.0, 0.3, 0.6]:
        d, _ = data.synthetic_series(beta=beta)
        s = st.bdi_signal(d, lookback=3)
        f = st.forward_sp_return(d)
        r = st.predictive_regression(s, f)
        print(f"  planted beta={beta:.1f} -> est={r['beta']:+.3f}  HAC t={r['tstat']:+.2f}  R2={r['r2']:.3f}")

    print(f"\nFingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv, use_proxy="--proxy" in sys.argv)
