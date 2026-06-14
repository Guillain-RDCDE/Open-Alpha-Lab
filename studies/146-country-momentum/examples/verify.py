"""Real-tape verification — Study 146 (Country-Momentum). Regenerates docs/results.md numbers.

Reads (or fetches) the 23-country ETF monthly return panel, runs the 12-1 momentum rotation
(long top-4) vs the equal-weight basket and a random-rotation control, sweeps costs, and
probes sub-period stability. Network is touched only with --fetch.

    python studies/146-country-momentum/examples/verify.py            # cache-only
    python studies/146-country-momentum/examples/verify.py --fetch    # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from country_momentum import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    panel = data.fetch_etfs(fetch=fetch)
    fp = data.fingerprint(panel)
    print(f"Panel: {panel.shape[0]} months × {panel.shape[1]} countries")
    print(f"Window: {panel.index[0]} -> {panel.index[-1]}  fingerprint={fp}")
    print()

    k = 4
    mom_df = st.run_strategy(panel, k=k, cost_bps=0, mode="momentum")
    ew_df  = st.run_strategy(panel, k=k, cost_bps=0, mode="ew")

    mom_s = st.summarize(mom_df["r_gross"])
    ew_s  = st.summarize(ew_df["r_gross"])

    # Random-rotation control (50 seeds)
    rand_means = []
    for s in range(50):
        rd = st.run_strategy(panel, k=k, cost_bps=0, mode="random", seed=s)
        rand_means.append(float(rd["r_gross"].mean()))
    rand_arr = np.array(rand_means)

    # Active vs EW
    active_vs_ew = mom_df["r_gross"] - ew_df["r_gross"]
    avew_s = st.summarize(active_vs_ew)

    print("=== Momentum top-4 (gross, cost=0) ===")
    print(
        f"n={mom_s['n']} ann={mom_s['ann_ret']*100:.1f}%  vol={mom_s['ann_vol']*100:.1f}%  "
        f"Sharpe={mom_s['sharpe']:+.2f}  max_dd={mom_s['max_drawdown']*100:.1f}%  "
        f"HAC t={mom_s['tstat']:+.2f}"
    )
    print(f"EW basket: ann={ew_s['ann_ret']*100:.1f}%  Sharpe={ew_s['sharpe']:+.2f}  HAC t={ew_s['tstat']:+.2f}")
    print(
        f"Random-rotation (50 seeds): mean ann={rand_arr.mean()*12*100:.1f}%  "
        f"mean monthly={rand_arr.mean()*100:.3f}%"
    )
    print()
    print(
        f"Active vs EW: ann={avew_s['ann_ret']*100:.1f}%  "
        f"Sharpe={avew_s['sharpe']:+.2f}  HAC t={avew_s['tstat']:+.2f}"
    )
    print()

    print("=== Cost sweep (momentum net, k=4) ===")
    for c in [0, 5, 10, 20, 40]:
        mdf = st.run_strategy(panel, k=k, cost_bps=c, mode="momentum")
        s = st.summarize(mdf["r_net"])
        print(
            f"  cost={c:3d}bps  ann={s['ann_ret']*100:.1f}%  "
            f"Sharpe={s['sharpe']:+.2f}  HAC t={s['tstat']:+.2f}"
        )
    print()

    print("=== Sub-period breakdown (gross, k=4) ===")
    for label, start, end in [
        ("1996-2009", "1996-04", "2009-12"),
        ("2010-2019", "2010-01", "2019-12"),
        ("2020-2026", "2020-01", "2026-06"),
    ]:
        mask = (panel.index >= start) & (panel.index <= end)
        sub = panel[mask]
        sub_m = st.run_strategy(sub, k=k, cost_bps=0, mode="momentum")
        sub_ew = st.run_strategy(sub, k=k, cost_bps=0, mode="ew")
        sm = st.summarize(sub_m["r_gross"])
        se = st.summarize(sub_ew["r_gross"])
        print(
            f"  {label}: mom ann={sm['ann_ret']*100:.1f}%  Sharpe={sm['sharpe']:+.2f}"
            f"  t={sm['tstat']:+.2f}  |  EW ann={se['ann_ret']*100:.1f}%  Sharpe={se['sharpe']:+.2f}"
        )
    print()

    print("=== k sensitivity (gross, 0 cost) ===")
    for kk in [2, 3, 4, 5, 6]:
        mdf = st.run_strategy(panel, k=kk, cost_bps=0, mode="momentum")
        s = st.summarize(mdf["r_gross"])
        print(
            f"  k={kk}: ann={s['ann_ret']*100:.1f}%  Sharpe={s['sharpe']:+.2f}"
            f"  HAC t={s['tstat']:+.2f}"
        )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
