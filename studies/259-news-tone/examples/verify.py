"""Real-tape verification -- Study 259 (News-Tone). Regenerates docs/results.md numbers.

Reads (or fetches) the ^GSPC daily series, joins the curated monthly news-tone
proxy, and runs the full battery: the seductive same-day / next-day fits, the
naive sign-timing strategy, and the two honesty checks (within-month demeaning
and prior-month tone) that expose the mirage. A synthetic positive control
confirms the engine is truthful. Network is touched only with --fetch.

    python studies/259-news-tone/examples/verify.py          # cache-only
    python studies/259-news-tone/examples/verify.py --fetch  # refresh ^GSPC
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from news_tone import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.build_real_panel(fetch=fetch)
    print(f"Panel: {len(df)} daily rows  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Fingerprint (last row): {data.fingerprint(df)}\n")

    # ---- The seductive fits ------------------------------------------------
    sd = st.predictive_regression(df, target_col="ret")
    nd = st.predictive_regression(df, target_col="fwd_ret")
    print("=== Seductive fits ===")
    print(f"SAME-DAY  (ret ~ tone):     beta={sd['beta']:+.5f}  HAC t={sd['tstat']:+.2f}  r2={sd['r2']:.4f}  (mechanical)")
    print(f"NEXT-DAY  (fwd_ret ~ tone): beta={nd['beta']:+.5f}  HAC t={nd['tstat']:+.2f}  r2={nd['r2']:.4f}")
    print(f"Contemporaneous corr tone~ret: {df['tone'].corr(df['ret']):+.3f}")

    # ---- The naive (mirage) strategy --------------------------------------
    sig = st.tone_sign_signal(df)
    gross = st.timing_returns(df, sig)
    net = st.net_of_costs(sig, gross, one_way_bps=1.0, borrow_bps_per_day=0.04)
    sg, sn = st.summarize(gross), st.summarize(net)
    bh = st.summarize(df["fwd_ret"])
    print("\n=== Naive sign-timing strategy (the mirage) ===")
    print(f"GROSS : {sg['mean_ann']*100:+.1f}%/yr  HAC t={sg['tstat']:+.2f}  SR={sg['sharpe_ann']:.2f}  hit={sg['hit_rate']:.3f}")
    print(f"NET   : {sn['mean_ann']*100:+.1f}%/yr  HAC t={sn['tstat']:+.2f}  SR={sn['sharpe_ann']:.2f}")
    print(f"B&H   : {bh['mean_ann']*100:+.1f}%/yr  HAC t={bh['tstat']:+.2f}  SR={bh['sharpe_ann']:.2f}")

    # ---- The two honesty checks -------------------------------------------
    print("\n=== Honesty checks ===")
    d2 = df.copy()
    d2["ym"] = d2.index.to_period("M")
    d2["fwd_dm"] = d2["fwd_ret"] - d2.groupby("ym")["fwd_ret"].transform("mean")
    dm = st.predictive_regression(
        d2[["tone", "fwd_dm"]].rename(columns={"fwd_dm": "fwd_ret"}), target_col="fwd_ret"
    )
    print(f"Within-month demeaned next-day reg: beta={dm['beta']:+.8f}  HAC t={dm['tstat']:+.2f}  r2={dm['r2']:.8f}")

    monthly_lag = data.news_tone_monthly().shift(1)
    keys = df.index + pd.offsets.MonthEnd(0)
    d3 = df.copy()
    d3["tone_lag"] = monthly_lag.reindex(keys).to_numpy()
    d3 = d3.dropna(subset=["tone_lag", "fwd_ret"])
    g_lag = (np.sign(d3["tone_lag"]) * d3["fwd_ret"]).dropna()
    s_lag = st.summarize(g_lag)
    print(f"Prior-month tone sign strategy    : {s_lag['mean_ann']*100:+.1f}%/yr  HAC t={s_lag['tstat']:+.2f}  SR={s_lag['sharpe_ann']:.2f}")
    print("  -> Demean kills it (t~0); prior-month tone kills it (t<2). The edge is hindsight.")

    # ---- Sub-periods of the mirage ----------------------------------------
    print("\n=== Sub-period analysis (naive strategy) ===")
    for a, b in [("2007", "2012"), ("2013", "2018"), ("2019", "2025")]:
        s = st.summarize(gross[a:b])
        print(f"  {a}-{b}: {s['mean_ann']*100:+.1f}%/yr  HAC t={s['tstat']:+.2f}  n={s['n']}")

    # ---- Synthetic positive control ---------------------------------------
    print("\n=== Synthetic positive control ===")
    for gamma in [0.0, 0.002, 0.004]:
        d, _ = data.synthetic_daily(n_days=2000, gamma=gamma, seed=data.SEED)
        reg = st.predictive_regression(d, target_col="fwd_ret")
        ss = st.summarize(st.timing_returns(d, st.tone_sign_signal(d)))
        tag = "NULL " if gamma == 0 else "planted"
        print(f"  gamma={gamma:.3f} [{tag}]: reg t={reg['tstat']:+.2f}  | sign SR={ss['sharpe_ann']:.2f}  t={ss['tstat']:+.2f}")

    print(f"\nFingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
