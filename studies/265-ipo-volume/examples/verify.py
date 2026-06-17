"""Real-tape verification -- Study 265 (IPO-Volume). Regenerates docs/results.md numbers.

Joins the hardcoded annual US IPO-count table with Shiller S&P 500 calendar-year
*price* returns, builds the expanding-window standardized froth signal, regresses
the *next-year* S&P return on it (HAC t-stat), reports the contemporaneous-vs-
predictive correlation gap, sweeps sub-periods, and races a long/flat froth-timing
rule against buy-and-hold (net of costs). The Shiller cache is repo-level and
read-only; no network is touched.

    python studies/265-ipo-volume/examples/verify.py          # cache-only (default)
    python studies/265-ipo-volume/examples/verify.py --gspc   # also cross-check ^GSPC cache
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ipo_volume import data, strategy as st  # noqa: E402


def main(use_gspc: bool) -> None:
    if use_gspc and os.path.exists(data.GSPC_CACHE):
        df = data.fetch_gspc_annual(fetch=False)
        tape = "^GSPC (yfinance cache)"
    else:
        df = data.fetch_sp500_annual()
        tape = "Shiller S&P 500 (price-only)"

    print(f"Tape: {tape}")
    print(f"Panel: {len(df)} forecastable years  {int(df['year'].min())}->{int(df['year'].max())}")
    print(f"Fingerprint (fwd_return): {data.fingerprint(df)}\n")

    z = st.froth_signal(df, min_obs=5)

    # ---- Predictive regression: next-year return on froth z ----------------
    reg = st.predictive_regression(z, df["fwd_return"])
    print("=== Predictive regression: next-year S&P return ~ alpha + beta * froth_z ===")
    print(f"  beta = {reg['beta']:+.4f}   HAC t = {reg['beta_t']:+.2f}   R^2 = {reg['r2']:.3f}   n = {reg['n']}")
    print(f"  predictive corr(froth_z, next-yr ret) = {reg['corr']:+.3f}")
    print("  Folklore predicts beta < 0 (more IPOs -> lower forward return).")

    # ---- The coincidence trap: contemporaneous vs predictive ---------------
    pair = pd.concat([z.rename("z"), df["sp500_return"].rename("c")], axis=1).dropna()
    contemp = float(np.corrcoef(pair["z"], pair["c"])[0, 1])
    print(f"\n=== The coincidence trap ===")
    print(f"  CONTEMPORANEOUS corr(froth_z, same-year ret) = {contemp:+.3f}  (froth IS high in up years)")
    print(f"  PREDICTIVE     corr(froth_z, next-year ret)  = {reg['corr']:+.3f}  (~0 -- no forecast power)")

    # ---- Sub-period sweep --------------------------------------------------
    print("\n=== Sub-period predictive slope (data-mining warning: tiny n) ===")
    yr = df["year"]
    for label, a, b in [("1985-1999", 1985, 1999), ("2000-2012", 2000, 2012), ("2013-2024", 2013, 2024)]:
        m = (yr >= a) & (yr <= b)
        rsub = st.predictive_regression(z[m], df["fwd_return"][m])
        print(f"  {label}: beta={rsub['beta']:+.4f}  HAC t={rsub['beta_t']:+.2f}  n={rsub['n']}")
    print("  Recent sub-periods look negative, but on n~12 that is unstable -- the full")
    print("  sample (n=41) is the honest test, and it reads t ~ 0.")

    # ---- Tradable wrapper: long/flat vs buy-and-hold -----------------------
    tm = st.timing_returns(z, df["fwd_return"], threshold=0.0, one_way_bps=5.0)
    s_g = st.summarize(tm["strat_gross"])
    s_n = st.summarize(tm["strat_net"])
    s_bh = st.summarize(tm["buy_hold"])
    n_long = int(tm["position"].sum())
    switches = int((tm["position"] != tm["position"].shift(1).fillna(0.0)).sum())
    drag_bps = float((tm["strat_gross"] - tm["strat_net"]).sum() * 1e4)
    print("\n=== Long/flat froth-timing rule (long when z<0) vs buy-and-hold ===")
    print(f"  Years long: {n_long}/{len(tm)}   switches: {switches}   total cost drag: {drag_bps:.0f} bps")
    print(f"  STRAT gross : {s_g['mean']*100:+.2f}%/yr  vol={s_g['vol']*100:.1f}%  t={s_g['tstat']:+.2f}  hit={s_g['hit_rate']:.2f}")
    print(f"  STRAT net   : {s_n['mean']*100:+.2f}%/yr  vol={s_n['vol']*100:.1f}%  t={s_n['tstat']:+.2f}")
    print(f"  BUY & HOLD  : {s_bh['mean']*100:+.2f}%/yr  vol={s_bh['vol']*100:.1f}%  t={s_bh['tstat']:+.2f}  hit={s_bh['hit_rate']:.2f}")
    exc = st.summarize(tm["strat_net"] - tm["buy_hold"])
    print(f"  STRAT - B&H : {exc['mean']*100:+.2f}%/yr  t={exc['tstat']:+.2f}  (negative => the rule HURTS)")

    # ---- Positive control --------------------------------------------------
    print("\n=== Positive control (synthetic planted slope) ===")
    for beta in (0.0, -0.06):
        syn, _ = data.synthetic_annual(n_years=120, beta=beta, seed=265)
        zs = st.froth_signal(syn, min_obs=5)
        rs = st.predictive_regression(zs, syn["fwd_return"])
        tag = "null " if beta == 0.0 else "froth"
        print(f"  beta={beta:+.2f} ({tag}): recovered beta={rs['beta']:+.4f}  HAC t={rs['beta_t']:+.2f}")
    print("  The estimator recovers a planted slope -- so the flat real-tape result is")
    print("  a true null, not a broken engine.")

    print(f"\nFingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(use_gspc="--gspc" in sys.argv)
