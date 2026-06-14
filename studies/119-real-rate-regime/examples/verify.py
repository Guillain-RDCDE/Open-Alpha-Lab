"""Reproduce the real-data headline run — Study 119 (Real-Rate-Regime).

Reads the pre-staged Shiller parquet from the repo's shared _cache/ folder
(cache-only; no network). Prints every number that appears in docs/results.md.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from real_rate_regime import data, strategy as st  # noqa: E402


def main() -> None:
    # ------------------------------------------------------------------
    # Load Shiller data
    # ------------------------------------------------------------------
    df = data.fetch_shiller(start_year=1873)
    fp = data.fingerprint(df)
    n_months = len(df)
    date_start = df.index[0].strftime("%Y-%m")
    date_end = df.index[-1].strftime("%Y-%m")

    print(f"\nStudy 119 — Real-Rate-Regime  ({date_start} .. {date_end}, n={n_months} months)")
    print(f"Inputs fingerprint: {fp}\n")

    # ------------------------------------------------------------------
    # Compute monthly real returns (needed for timing overlay)
    # ------------------------------------------------------------------
    df["monthly_real_ret"] = np.log(df["sp500_real"] / df["sp500_real"].shift(1))

    # ------------------------------------------------------------------
    # 1. Level sort: forward 12m real return by real-rate quartile
    # ------------------------------------------------------------------
    clean = df.dropna(subset=["real_rate", "real_rate_chg", "fwd12_real"])
    clean2 = clean.copy()
    clean2["bucket"] = pd.qcut(clean2["real_rate"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    print("Forward 12-month real return by real-rate level quartile:")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = clean2[clean2["bucket"] == q]["fwd12_real"]
        s = st.summarize(sub)
        rr_med = clean2[clean2["bucket"] == q]["real_rate"].median()
        print(f"  {q} (median rr={rr_med*100:+.1f}%): fwd12={sub.mean()*100:+.1f}%, "
              f"HAC t={s['tstat']:+.2f}, n={len(sub)}")

    # ------------------------------------------------------------------
    # 2. Momentum sort: rising vs falling regime
    # ------------------------------------------------------------------
    print("\nForward 12-month real return by real-rate direction:")
    ms = st.momentum_sort(clean)
    for reg, row in ms.iterrows():
        mask = (clean["real_rate_chg"] > 0) if reg == "rising" else (clean["real_rate_chg"] <= 0)
        s = st.summarize(clean.loc[mask, "fwd12_real"])
        print(f"  {reg}: {row['fwd_mean']*100:+.1f}%/yr, HAC t={s['tstat']:+.2f}, n={row['n']:.0f}")

    corr_level = float(np.corrcoef(clean["real_rate"], clean["fwd12_real"])[0, 1])
    clean_chg = clean.dropna(subset=["real_rate_chg"])
    corr_chg = float(np.corrcoef(clean_chg["real_rate_chg"], clean_chg["fwd12_real"])[0, 1])
    print(f"\n  Corr(real rate level, fwd12): {corr_level:+.3f}")
    print(f"  Corr(real rate 12m chg, fwd12): {corr_chg:+.3f}")

    # ------------------------------------------------------------------
    # 3. Timing overlay: risk-off when real rates rising
    # ------------------------------------------------------------------
    df_m = df.dropna(subset=["monthly_real_ret", "real_rate_chg"]).copy()
    overlay = st.timing_overlay(df_m)

    strat = st.summarize(overlay["strat_ret"])
    bh = st.summarize(overlay["bh_ret"])
    diff_t = st.diff_tstat(overlay["strat_ret"], overlay["bh_ret"])
    frac_in = float(overlay["signal"].mean())

    print(f"\nTiming overlay (risk-off when real rates rising, lag=1 month):")
    print(f"  Strategy: mean {strat['mean_ann']*100:+.1f}%/yr, Sharpe {strat['sharpe']:+.3f}, "
          f"HAC t={strat['tstat']:+.2f} (n={strat['n']})")
    print(f"  Buy&hold: mean {bh['mean_ann']*100:+.1f}%/yr, Sharpe {bh['sharpe']:+.3f}, "
          f"HAC t={bh['tstat']:+.2f}")
    print(f"  Diff HAC t (strat - BH): {diff_t:+.2f}")
    print(f"  Fraction in market: {frac_in:.1%}")

    # ------------------------------------------------------------------
    # Verdict summary
    # ------------------------------------------------------------------
    q1_fwd = clean2[clean2["bucket"] == "Q1"]["fwd12_real"].mean()
    q4_fwd = clean2[clean2["bucket"] == "Q4"]["fwd12_real"].mean()
    rising_fwd = ms.loc["rising", "fwd_mean"]
    falling_fwd = ms.loc["falling", "fwd_mean"]

    print("\n" + "=" * 60)
    print("VERDICT:")
    print(f"  Level:    Q1 (cheap money) {q1_fwd*100:+.1f}%/yr  "
          f"Q4 (dear money) {q4_fwd*100:+.1f}%/yr")
    print(f"  Direction: rising {rising_fwd*100:+.1f}%/yr  falling {falling_fwd*100:+.1f}%/yr")
    print(f"  Timing strategy HAC t={strat['tstat']:+.2f}, BH HAC t={bh['tstat']:+.2f}")
    print(f"  Strat vs BH diff t={diff_t:+.2f}  (timing rule SIGNIFICANTLY underperforms BH)")
    print("  Signal: NONE  |  Tradability: MIRAGE")
    print("=" * 60)


if __name__ == "__main__":
    main()
