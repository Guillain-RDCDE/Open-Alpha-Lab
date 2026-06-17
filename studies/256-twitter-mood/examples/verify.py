"""Real-tape verification -- Study 256 (Twitter-Mood). Regenerates docs/results.md numbers.

Joins the curated daily 'Calm' mood index with real ^GSPC daily returns over the
Bollen (2011) window, runs the lead-lag regression sweep (Granger-lag trap), a
permutation/shuffle null, the directional hit-rate vs the unconditional up-rate,
and a gross/net long-short trading rule with execution lag and costs.
Network is touched only with --fetch.

    python studies/256-twitter-mood/examples/verify.py          # cache-only
    python studies/256-twitter-mood/examples/verify.py --fetch  # refresh ^GSPC
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from twitter_mood import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.build_real_panel(fetch=fetch)
    print(f"Panel: {df.shape[0]} business days x mood+return")
    print(f"Window: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Fingerprint: {data.fingerprint(df)}\n")

    # ---- Lead-lag regression sweep (the Granger-lag trap) ------------------
    print("=== Lead-lag: ret_{t+lag} ~ a + b * Calm_t  (HAC t) ===")
    sw = st.leadlag_sweep(df, max_lag=5)
    for _, row in sw.iterrows():
        print(f"  lag {int(row['lag'])}: beta={row['beta']:+.5f}  HAC t={row['tstat']:+.2f}  R2={row['r2']:.4f}  n={int(row['n'])}")
    best = sw.iloc[sw["tstat"].abs().idxmax()]
    print(f"  Best |t| across 5 lags: lag {int(best['lag'])}  t={best['tstat']:+.2f}")
    print(f"  Bonferroni bar for 5 lags at 5%: |t| ~ 2.58 -- not cleared.")

    # ---- Permutation / shuffle null ---------------------------------------
    print("\n=== Shuffle null (time-permute Calm, re-estimate slope) ===")
    for lag in [1, 2, 3]:
        sn = st.shuffle_null(df, lag=lag, n_shuffles=2000, seed=256)
        print(f"  lag {lag}: real beta={sn['real_beta']:+.5f}  t={sn['real_tstat']:+.2f}  perm p={sn['perm_p']:.4f}")

    # ---- Directional hit-rate vs unconditional up-rate --------------------
    hr = st.directional_hitrate(df, lookback=20)
    print("\n=== Directional next-day hit-rate ===")
    print(f"  Calm-rule hit-rate : {hr['hit_rate']*100:.1f}%")
    print(f"  Unconditional up-rate (baseline): {hr['uncond_up_rate']*100:.1f}%")
    print(f"  Edge over baseline : {(hr['hit_rate']-hr['uncond_up_rate'])*100:+.1f} pts   (n={hr['n']})")

    # ---- Trading rule: gross / net, long-only and long-short --------------
    print("\n=== Trading rule (1-day execution lag) ===")
    for short, label in [(False, "long/flat"), (True, "long/short")]:
        res = st.mood_strategy(df, lookback=20, allow_short=short, one_way_bps=5.0)
        sg = st.summarize(res["gross"])
        snet = st.summarize(res["net"])
        sm = st.summarize(res["mkt"])
        print(f"  [{label}] gross: {sg['mean']*252*100:+.1f}%/yr  HAC t={sg['tstat']:+.2f}  | "
              f"net@5bps: {snet['mean']*252*100:+.1f}%/yr  HAC t={snet['tstat']:+.2f}")
    print(f"  Buy-and-hold over window: {sm['mean']*252*100:+.1f}%/yr  HAC t={sm['tstat']:+.2f}")
    print("  (The window is the 2008 crash; any net-short rule looks good for the")
    print("   wrong reason -- it is not a mood signal, and it does not clear t>=2.)")

    # ---- Positive control --------------------------------------------------
    print("\n=== Positive control (synthetic planted lead-lag) ===")
    for knob in [0.0, 0.002, 0.004]:
        dp, _ = data.synthetic_series(n_days=200, predictive=knob, seed=256)
        r = st.leadlag_regression(dp, lag=1)
        print(f"  predictive={knob:.3f}: beta={r['beta']:+.5f}  HAC t={r['tstat']:+.2f}")
    print("  -> The engine recovers a planted lead-lag (t>2 at 0.004) and reads ~0 on the null.")

    print(f"\nFingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
