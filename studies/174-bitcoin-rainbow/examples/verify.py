"""Real-tape verification -- Study 174 (Bitcoin-Rainbow). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the BTC-USD daily tape, runs the Rainbow Chart strategy
in both its in-sample (look-ahead-biased) and walk-forward (honest) forms, and prints
the headline metrics. Shows that the gorgeous in-sample t-stat collapses to noise the
moment future data is withheld.

    python studies/174-bitcoin-rainbow/examples/verify.py            # cache-only
    python studies/174-bitcoin-rainbow/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bitcoin_rainbow import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_btc(fetch=fetch)
    print(f"BTC-USD: {df.index[0].date()} -> {df.index[-1].date()}  "
          f"n={len(df)}  fp={data.fingerprint(df)}")
    print()

    # --- In-sample fit (look-ahead biased) ---
    df_is = st.fit_rainbow_insample(df)
    sig_is = st.rainbow_signal(df_is, "band_sigma_insample")
    bt_is = st.backtest(df_is, sig_is, cost_bps=0.0)
    s_is = st.summarize(bt_is)
    print("=== IN-SAMPLE (look-ahead biased) ===")
    print(f"  ann return:  {s_is['ann_ret_pct']:+.1f}%  (BH: {s_is['ann_bh_pct']:+.1f}%)")
    print(f"  Sharpe:      {s_is['sharpe']:+.2f}")
    print(f"  HAC t-stat:  {s_is['tstat']:+.2f}")
    print(f"  Time in mkt: {s_is['time_in_market_pct']:.1f}%")
    print()

    # --- Walk-forward fit (honest) ---
    df_wf = st.fit_rainbow_walkforward(df)
    sig_wf = st.rainbow_signal(df_wf, "band_sigma_wf")
    bt_wf = st.backtest(df_wf, sig_wf, cost_bps=0.0)
    s_wf = st.summarize(bt_wf)
    print("=== WALK-FORWARD (honest) ===")
    print(f"  ann return:  {s_wf['ann_ret_pct']:+.1f}%  (BH: {s_wf['ann_bh_pct']:+.1f}%)")
    print(f"  Sharpe:      {s_wf['sharpe']:+.2f}")
    print(f"  HAC t-stat:  {s_wf['tstat']:+.2f}")
    print(f"  Time in mkt: {s_wf['time_in_market_pct']:.1f}%")
    print()

    # --- Cost sweep (walk-forward) ---
    print("=== COST SWEEP (walk-forward) ===")
    for c in [0, 10, 50, 100]:
        bt = st.backtest(df_wf, sig_wf, cost_bps=c)
        s = st.summarize(bt)
        print(f"  cost={c:3.0f} bps:  ann={s['ann_ret_pct']:+6.1f}%  t={s['tstat']:+.2f}")
    print()

    # --- In-sample vs walk-forward sigma comparison ---
    print("=== BAND SIGMA STATS ===")
    bs_is = df_is["band_sigma_insample"].dropna()
    bs_wf = df_wf["band_sigma_wf"].dropna()
    print(f"  IS band sigma: mean={bs_is.mean():+.3f}  std={bs_is.std():.3f}  "
          f"min={bs_is.min():.2f}  max={bs_is.max():.2f}")
    print(f"  WF band sigma: mean={bs_wf.mean():+.3f}  std={bs_wf.std():.3f}  "
          f"min={bs_wf.min():.2f}  max={bs_wf.max():.2f}")
    print()

    print("=== VERDICT ===")
    t_diff = s_is["tstat"] - s_wf["tstat"]
    print(f"  Look-ahead bonus: IS t={s_is['tstat']:+.2f}  WF t={s_wf['tstat']:+.2f}  "
          f"Delta t={t_diff:+.2f}")
    print(f"  WF beats BH: {s_wf['ann_ret_pct']:+.1f}% vs {s_wf['ann_bh_pct']:+.1f}% "
          f"({'YES' if s_wf['ann_ret_pct'] > s_wf['ann_bh_pct'] else 'NO, badly'}) ")
    print(f"  Signal = {'REAL' if abs(s_wf['tstat']) >= 2 else 'NONE'}  |  "
          f"Tradability = {'INVESTABLE' if s_wf['ann_ret_pct'] > s_wf['ann_bh_pct'] * 0.5 else 'MIRAGE'}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
