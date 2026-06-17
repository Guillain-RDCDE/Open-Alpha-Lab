"""Real-tape verification — Study 225 (Sector-Rotation). Regenerates docs/results.md numbers.

Reads (or fetches) the 11 SPDR sector ETF monthly return panel plus SPY, runs the
momentum rotation (long top-K sectors by trailing 6-month return) vs the equal-weight
sector basket and a random-rotation control, sweeps costs, and probes sub-period
stability and lookback sensitivity. Network is touched only with --fetch.

    python studies/225-sector-rotation/examples/verify.py            # cache-only
    python studies/225-sector-rotation/examples/verify.py --fetch    # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sector_rotation import data, strategy as st  # noqa: E402

STUDY_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def main(fetch: bool) -> None:
    panel, spy = data.fetch_etfs(fetch=fetch, cache_dir=STUDY_CACHE)
    fp = data.fingerprint(panel)
    print(f"Panel: {panel.shape[0]} months x {panel.shape[1]} sectors")
    print(f"Window: {panel.index[0]} -> {panel.index[-1]}  fingerprint={fp}")
    print(f"SPY: {len(spy)} months")
    print()

    k = 3
    lb = 6

    mom_df = st.run_strategy(panel, k=k, lookback=lb, cost_bps=0, mode="momentum")
    ew_df  = st.run_strategy(panel, k=k, lookback=lb, cost_bps=0, mode="ew")

    mom_s = st.summarize(mom_df["r_gross"])
    ew_s  = st.summarize(ew_df["r_gross"])
    spy_s = st.summarize(spy)

    # Random-rotation control (50 seeds)
    rand_means = []
    for s in range(50):
        rd = st.run_strategy(panel, k=k, lookback=lb, cost_bps=0, mode="random", seed=s)
        rand_means.append(float(rd["r_gross"].mean()))
    rand_arr = np.array(rand_means)

    # Active vs EW
    active_vs_ew = mom_df["r_gross"] - ew_df["r_gross"]
    avew_s = st.summarize(active_vs_ew)

    # Active vs SPY (align index)
    common = mom_df["r_gross"].index.intersection(spy.index)
    active_vs_spy = mom_df["r_gross"].loc[common] - spy.loc[common]
    avspy_s = st.summarize(active_vs_spy)

    print("=== Momentum top-3 6mo (gross, cost=0) ===")
    print(
        f"n={mom_s['n']} ann={mom_s['ann_ret']*100:.1f}%  vol={mom_s['ann_vol']*100:.1f}%  "
        f"Sharpe={mom_s['sharpe']:+.2f}  max_dd={mom_s['max_drawdown']*100:.1f}%  "
        f"HAC t={mom_s['tstat']:+.2f}"
    )
    print(f"EW sector basket: ann={ew_s['ann_ret']*100:.1f}%  Sharpe={ew_s['sharpe']:+.2f}  HAC t={ew_s['tstat']:+.2f}")
    print(f"SPY benchmark:    ann={spy_s['ann_ret']*100:.1f}%  Sharpe={spy_s['sharpe']:+.2f}  HAC t={spy_s['tstat']:+.2f}")
    print(
        f"Random-rotation (50 seeds): mean ann={rand_arr.mean()*12*100:.1f}%  "
        f"mean monthly={rand_arr.mean()*100:.3f}%"
    )
    print()
    print(
        f"Active vs EW:  ann={avew_s['ann_ret']*100:.1f}%  "
        f"Sharpe={avew_s['sharpe']:+.2f}  HAC t={avew_s['tstat']:+.2f}"
    )
    print(
        f"Active vs SPY: ann={avspy_s['ann_ret']*100:.1f}%  "
        f"Sharpe={avspy_s['sharpe']:+.2f}  HAC t={avspy_s['tstat']:+.2f}"
    )
    print()

    print("=== Cost sweep (momentum net, k=3, 6mo) ===")
    for c in [0, 5, 10, 20, 40]:
        mdf = st.run_strategy(panel, k=k, lookback=lb, cost_bps=c, mode="momentum")
        s = st.summarize(mdf["r_net"])
        print(
            f"  cost={c:3d}bps  ann={s['ann_ret']*100:.1f}%  "
            f"Sharpe={s['sharpe']:+.2f}  HAC t={s['tstat']:+.2f}"
        )
    print()

    print("=== Sub-period breakdown (gross, k=3, 6mo) ===")
    for label, start, end in [
        ("1999-2009", "1999-02", "2009-12"),
        ("2010-2019", "2010-01", "2019-12"),
        ("2020-2026", "2020-01", "2026-06"),
    ]:
        mask = (panel.index >= start) & (panel.index <= end)
        sub = panel[mask]
        sub_m = st.run_strategy(sub, k=k, lookback=lb, cost_bps=0, mode="momentum")
        sub_ew = st.run_strategy(sub, k=k, lookback=lb, cost_bps=0, mode="ew")
        sm = st.summarize(sub_m["r_gross"])
        se = st.summarize(sub_ew["r_gross"])
        print(
            f"  {label}: mom ann={sm['ann_ret']*100:.1f}%  Sharpe={sm['sharpe']:+.2f}"
            f"  t={sm['tstat']:+.2f}  |  EW ann={se['ann_ret']*100:.1f}%  Sharpe={se['sharpe']:+.2f}"
        )
    print()

    print("=== Lookback sweep (k=3, 0 cost) ===")
    lb_sweep = st.lookback_sweep(panel, k=k, lookbacks=(3, 6, 9, 12))
    print(lb_sweep.round(2).to_string())
    print()

    print("=== K sensitivity (6mo, 0 cost) ===")
    for kk in [2, 3, 4, 5]:
        mdf = st.run_strategy(panel, k=kk, lookback=lb, cost_bps=0, mode="momentum")
        s = st.summarize(mdf["r_gross"])
        print(
            f"  k={kk}: ann={s['ann_ret']*100:.1f}%  Sharpe={s['sharpe']:+.2f}"
            f"  HAC t={s['tstat']:+.2f}"
        )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
