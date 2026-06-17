"""Real-tape verification -- Study 222 (Altseason-Rotation). Regenerates docs/results.md numbers.

    python studies/222-altseason-rotation/examples/verify.py            # cache-only
    python studies/222-altseason-rotation/examples/verify.py --fetch    # refresh the panel
"""
from __future__ import annotations
import os
import sys
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from altseason_rotation import data, strategy as st  # noqa: E402

def main(fetch):
    panel = data.fetch_panel(fetch=fetch)
    print("Panel: {} .. {}".format(panel.index[0].date(), panel.index[-1].date()))
    print("Fingerprint: {}".format(data.fingerprint(panel)))
    print()
    dom = data.compute_dominance(panel)
    print("BTC dominance proxy: mean={:.3f}  min={:.3f}  max={:.3f}  std={:.4f}".format(
        dom.mean(), dom.min(), dom.max(), dom.std()))
    print()
    sig1 = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    pnl1 = st.rotation_pnl(panel, sig1, cost_bps=40)
    ev1 = st.count_rotations(pnl1)
    p1g = st.performance_summary(pnl1["gross_ret"])
    p1n = st.performance_summary(pnl1["net_ret"])
    print("=== dom_change_20d signal ===")
    print("  Events: n={}, avg_hold={:.1f}d, active={:.1f}%".format(
        ev1["n_rotations"], ev1["avg_hold_days"], ev1["pct_days_active"]))
    print("  Gross: ann_ret={:.2f}% Sharpe={:.2f} t={:+.2f} MDD={:.2f}%".format(
        p1g["ann_return_pct"], p1g["sharpe"], p1g["tstat"], p1g["max_dd_pct"]))
    print("  Net:   ann_ret={:.2f}% Sharpe={:.2f} t={:+.2f} MDD={:.2f}%".format(
        p1n["ann_return_pct"], p1n["sharpe"], p1n["tstat"], p1n["max_dd_pct"]))
    sig2 = st.dominance_level_signal(panel, window=252)
    pnl2 = st.rotation_pnl(panel, sig2, cost_bps=40)
    ev2 = st.count_rotations(pnl2)
    p2g = st.performance_summary(pnl2["gross_ret"])
    p2n = st.performance_summary(pnl2["net_ret"])
    print()
    print("=== dom_level signal ===")
    print("  Events: n={}, avg_hold={:.1f}d, active={:.1f}%".format(
        ev2["n_rotations"], ev2["avg_hold_days"], ev2["pct_days_active"]))
    print("  Gross: ann_ret={:.2f}% Sharpe={:.2f} t={:+.2f} MDD={:.2f}%".format(
        p2g["ann_return_pct"], p2g["sharpe"], p2g["tstat"], p2g["max_dd_pct"]))
    print("  Net:   ann_ret={:.2f}% Sharpe={:.2f} t={:+.2f} MDD={:.2f}%".format(
        p2n["ann_return_pct"], p2n["sharpe"], p2n["tstat"], p2n["max_dd_pct"]))
    bm = st.benchmark_pnl(panel)
    spread_bm = bm["alt_ew_ret"] - bm["btc_ret"]
    pb = st.performance_summary(bm["btc_ret"])
    pa = st.performance_summary(bm["alt_ew_ret"])
    ps = st.performance_summary(spread_bm)
    print()
    print("=== Benchmarks ===")
    print("  BTC B&H:       ann_ret={:.2f}% Sharpe={:.2f}".format(pb["ann_return_pct"], pb["sharpe"]))
    print("  Alt EW B&H:    ann_ret={:.2f}% Sharpe={:.2f}".format(pa["ann_return_pct"], pa["sharpe"]))
    print("  Always spread: ann_ret={:.2f}% Sharpe={:.2f} t={:+.2f}".format(
        ps["ann_return_pct"], ps["sharpe"], ps["tstat"]))

if __name__ == "__main__":
    main(fetch="--fetch" in __import__("sys").argv)
