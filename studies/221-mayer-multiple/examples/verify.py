"""Real-tape verification -- Study 221 (Mayer-Multiple). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the BTC-USD daily tape, computes the Mayer Multiple, runs
the buy-below-1 / trim-above-2.4 strategy, and prints the headline metrics together with
the 30-day forward return comparison across bands.

    python studies/221-mayer-multiple/examples/verify.py            # cache-only
    python studies/221-mayer-multiple/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mayer_multiple import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_btc(fetch=fetch)
    print(f"BTC-USD: {df.index[0].date()} -> {df.index[-1].date()}  "
          f"n={len(df)}  fp={data.fingerprint(df)}")
    print()

    # --- Mayer Multiple distribution ---
    mm_stats = st.mm_distribution(df)
    print("=== MAYER MULTIPLE DISTRIBUTION ===")
    print(f"  mean:  {mm_stats['mean']:.3f}   median: {mm_stats['median']:.3f}")
    print(f"  min:   {mm_stats['min']:.3f}   max:    {mm_stats['max']:.3f}")
    print(f"  % days < 1.0 (cheap):     {mm_stats['pct_below_1']:.1f}%")
    print(f"  % days > 2.4 (expensive): {mm_stats['pct_above_2p4']:.1f}%")
    print(f"  % days neutral 1.0-2.4:   {mm_stats['pct_neutral']:.1f}%")
    print()

    # --- Long-only backtest (MM < 1) ---
    sig = st.mayer_signal(df, buy_threshold=1.0, sell_threshold=2.4, allow_short=False)

    print("=== LONG-ONLY STRATEGY (long when MM < 1.0) ===")
    bt_gross = st.backtest(df, sig, cost_bps=0.0)
    s_gross = st.summarize(bt_gross)
    print(f"  Gross:  ann={s_gross['ann_ret_pct']:+.1f}%  BH={s_gross['ann_bh_pct']:+.1f}%  "
          f"Sharpe={s_gross['sharpe']:+.3f}  HAC t={s_gross['tstat']:+.3f}")
    print(f"  Time in market: {s_gross['time_in_market_pct']:.1f}%")

    bt_net = st.backtest(df, sig, cost_bps=10.0)
    s_net = st.summarize(bt_net)
    print(f"  Net 10bps: ann={s_net['ann_ret_pct']:+.1f}%  HAC t={s_net['tstat']:+.3f}")
    print()

    # --- Cost sweep ---
    print("=== COST SWEEP (long-only, MM < 1) ===")
    for c in [0, 10, 50, 100]:
        bt = st.backtest(df, sig, cost_bps=c)
        s = st.summarize(bt)
        print(f"  cost={c:3d} bps:  ann={s['ann_ret_pct']:+6.1f}%  HAC t={s['tstat']:+.3f}")
    print()

    # --- Forward returns by band ---
    print("=== 30D FORWARD RETURN BY BAND ===")
    fwd = st.forward_return_by_band(df)
    for regime in ["cheap", "neutral", "expensive"]:
        grp = fwd.loc[fwd["regime"] == regime, "fwd_30d"]
        if len(grp) > 0:
            print(f"  {regime:12s}: n={len(grp):5d}  mean={grp.mean()*100:+6.2f}%  "
                  f"std={grp.std()*100:5.2f}%")

    t_cn = st.band_tstat(fwd, "cheap", "neutral")
    t_en = st.band_tstat(fwd, "expensive", "neutral")
    print(f"  t-stat cheap vs neutral:     {t_cn:+.3f}")
    print(f"  t-stat expensive vs neutral: {t_en:+.3f}")
    print()

    # --- Verdict ---
    print("=== VERDICT ===")
    t_strat = s_gross["tstat"]
    print(f"  Strategy HAC t = {t_strat:+.3f}  (bar: |t| >= 2)")
    print(f"  Signal = {'REAL' if abs(t_strat) >= 2 else 'NONE'}  |  "
          f"Tradability = {'INVESTABLE' if s_gross['ann_ret_pct'] > s_gross['ann_bh_pct'] * 0.5 else 'MIRAGE'}")
    if abs(t_cn) >= 2:
        direction = "SAME as claimed" if t_cn > 0 else "INVERTED — cheap underperforms neutral"
        print(f"  Band effect (cheap vs neutral): t={t_cn:+.3f} — {direction}")
    print(f"\n  Fingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
