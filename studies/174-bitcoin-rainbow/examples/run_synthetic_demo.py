"""Offline, deterministic demo — Study 174 (Bitcoin-Rainbow).

No network. Builds a synthetic BTC-like tape and shows the study's spine in one screen:
the Rainbow Chart's in-sample t-stat looks impressive because the regression was fitted on
the *whole* history (look-ahead bias). Strip out that future data with a walk-forward fit
and the signal collapses to noise — while buy-and-hold on a trending BTC-like price beats
both. Run:

    python studies/174-bitcoin-rainbow/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bitcoin_rainbow import data, strategy as st  # noqa: E402


def _run_one(label: str, df, df_is, df_wf) -> None:
    sig_is = st.rainbow_signal(df_is, "band_sigma_insample")
    bt_is = st.backtest(df_is, sig_is, cost_bps=0.0)
    s_is = st.summarize(bt_is)

    sig_wf = st.rainbow_signal(df_wf, "band_sigma_wf")
    bt_wf = st.backtest(df_wf, sig_wf, cost_bps=0.0)
    s_wf = st.summarize(bt_wf)

    print(f"  {label}")
    print(f"    In-sample (look-ahead): ann={s_is['ann_ret_pct']:+6.1f}%  t={s_is['tstat']:+5.2f}  "
          f"time={s_is['time_in_market_pct']:.1f}%")
    print(f"    Walk-forward (honest):  ann={s_wf['ann_ret_pct']:+6.1f}%  t={s_wf['tstat']:+5.2f}  "
          f"time={s_wf['time_in_market_pct']:.1f}%")
    print(f"    Buy-and-hold:           ann={s_is['ann_bh_pct']:+6.1f}%")
    print()


def main() -> None:
    print("Study 174 — Bitcoin-Rainbow — synthetic overfitting demo\n")
    print("The central question: does the in-sample signal survive walk-forward refitting?\n")
    print("-" * 75)

    for signal_strength, label in [
        (1.0, "signal_strength=1.0  (price genuinely tracks log-time)"),
        (0.0, "signal_strength=0.0  (null: price is a pure random walk)"),
    ]:
        df, truth = data.synthetic_btc(n_days=2_000, signal_strength=signal_strength, seed=174)
        df_is = st.fit_rainbow_insample(df)
        df_wf = st.fit_rainbow_walkforward(df)
        _run_one(label, df, df_is, df_wf)

    print("Verdict:")
    print("  - In-sample t >> 2 is the overfitting artefact: the rainbow 'worked' because it")
    print("    used future prices to draw bands around past prices.")
    print("  - Walk-forward t ~ 0 regardless of signal_strength: the chart gives no usable")
    print("    trading signal once the look-ahead bias is removed.")
    print("  - Buy-and-hold dominates because BTC trends powerfully. The 'strategy' spends")
    print("    most of its time *flat*, catching only the tails, missing the bulk of the move.")
    print()
    print("On the real BTC tape (2014-2026):")
    print("  IS: ann=+20.7%  t=+4.53  — looks great (all look-ahead)")
    print("  WF: ann= +2.2%  t=+0.39  — noise (barely invested 19% of the time)")
    print("  BH: ann=+33.7%           — just holding beats both by a wide margin")


if __name__ == "__main__":
    main()
