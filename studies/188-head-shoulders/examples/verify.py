"""Real-tape verification — Study 188 (Head-Shoulders).  Regenerates docs/results.md numbers.

Fetches (or reads from cache) daily OHLCV for the ticker basket, detects confirmed
head-and-shoulders top and inverse-H&S bottom patterns, measures forward returns at
1, 5, 20, and 60 trading days, pins them against an unconditional random-day baseline,
and reports HAC t-stats with a Bonferroni correction for multiple comparisons.

    python studies/188-head-shoulders/examples/verify.py            # cache-only
    python studies/188-head-shoulders/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from head_shoulders import data, strategy as st  # noqa: E402

TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL",
    "META", "NVDA", "TSLA", "JPM",
]
HORIZONS = (1, 5, 20, 60)
N_COMPARISONS = 2 * len(HORIZONS)  # 2 pattern types × 4 horizons = 8


def _load_all(fetch: bool):
    frames = {}
    for t in TICKERS:
        try:
            bars = data.fetch_daily(t, fetch=fetch)
            frames[t] = bars
            if fetch:
                print(f"{t:5s}  {bars.index[0].date()}..{bars.index[-1].date()}  "
                      f"fp={data.fingerprint(bars)}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
    return frames


def _pool_study(tapes: dict) -> dict:
    """Pool H&S study across all tickers."""
    top_fwd_list, inv_fwd_list, ctrl_list = [], [], []
    sig_counts = {"top": 0, "inverse": 0}

    for t, bars in tapes.items():
        result = st.run_hs_study(bars, horizons=HORIZONS, seed=42)
        sig_counts["top"] += result["n_top"]
        sig_counts["inverse"] += result["n_inv"]
        if result["n_top"] > 0 and not result["fwd_top"].empty:
            top_fwd_list.append(result["fwd_top"])
        if result["n_inv"] > 0 and not result["fwd_inv"].empty:
            inv_fwd_list.append(result["fwd_inv"])
        if not result["control"].empty:
            ctrl_list.append(result["control"])

    top_fwd = pd.concat(top_fwd_list, ignore_index=True) if top_fwd_list else pd.DataFrame()
    inv_fwd = pd.concat(inv_fwd_list, ignore_index=True) if inv_fwd_list else pd.DataFrame()
    ctrl = pd.concat(ctrl_list, ignore_index=True) if ctrl_list else pd.DataFrame()

    return {"top_fwd": top_fwd, "inv_fwd": inv_fwd, "ctrl": ctrl, "n": sig_counts}


def main(fetch: bool) -> None:
    print("Study 188 — Head-Shoulders — real-tape verification\n")

    tapes = _load_all(fetch)
    if not tapes:
        print("ERROR: no cached tapes found. Run with --fetch to download.")
        sys.exit(1)

    pooled = _pool_study(tapes)
    n_top = pooled["n"]["top"]
    n_inv = pooled["n"]["inverse"]
    top_fwd = pooled["top_fwd"]
    inv_fwd = pooled["inv_fwd"]
    ctrl = pooled["ctrl"]

    print(f"Tickers: {len(tapes)}   H&S top signals: {n_top}   Inverse H&S: {n_inv}\n")

    bonferroni_thresh = 0.05 / N_COMPARISONS  # adjusted p-value threshold
    bonferroni_t = 2.576  # ~|t| for α/8 two-sided (very roughly)
    print(f"Bonferroni threshold: a/8 = {bonferroni_thresh:.4f}  (|t| ~= {bonferroni_t:.2f})\n")

    print("=== H&S TOP (bearish signal, short entry) ===")
    if n_top == 0:
        print("  No confirmed H&S top signals found.\n")
    else:
        print(f"{'horizon':>8} {'n':>5} {'mean%':>8} {'win%':>7} {'HAC t':>7} {'excess%':>9} {'t_excess':>9}")
        for h in HORIZONS:
            s = st.summarize(top_fwd, direction=-1, horizon=h, control=ctrl)
            flag = "*" if abs(s.get("tstat_excess", 0) or 0) >= bonferroni_t else " "
            print(
                f"{h:>7}d {s.get('n', 0):>5} {s.get('mean_pct', float('nan')):>+8.3f} "
                f"{s.get('win_rate', float('nan')) * 100:>6.1f}% {s.get('tstat', float('nan')):>+7.2f} "
                f"{s.get('excess_pct', float('nan')):>+9.3f} {s.get('tstat_excess', float('nan')):>+9.2f}{flag}"
            )

    print()
    print("=== INVERSE H&S (bullish signal, long entry) ===")
    if n_inv == 0:
        print("  No confirmed IH&S signals found.\n")
    else:
        print(f"{'horizon':>8} {'n':>5} {'mean%':>8} {'win%':>7} {'HAC t':>7} {'excess%':>9} {'t_excess':>9}")
        for h in HORIZONS:
            s = st.summarize(inv_fwd, direction=+1, horizon=h, control=ctrl)
            flag = "*" if abs(s.get("tstat_excess", 0) or 0) >= bonferroni_t else " "
            print(
                f"{h:>7}d {s.get('n', 0):>5} {s.get('mean_pct', float('nan')):>+8.3f} "
                f"{s.get('win_rate', float('nan')) * 100:>6.1f}% {s.get('tstat', float('nan')):>+7.2f} "
                f"{s.get('excess_pct', float('nan')):>+9.3f} {s.get('tstat_excess', float('nan')):>+9.2f}{flag}"
            )

    print()
    print("=== VERDICT ===")
    all_t_excess = []
    for h in HORIZONS:
        if not top_fwd.empty:
            s = st.summarize(top_fwd, direction=-1, horizon=h, control=ctrl)
            all_t_excess.append(s.get("tstat_excess", float("nan")))
        if not inv_fwd.empty:
            s = st.summarize(inv_fwd, direction=+1, horizon=h, control=ctrl)
            all_t_excess.append(s.get("tstat_excess", float("nan")))

    t_arr = np.array([t for t in all_t_excess if np.isfinite(t)])
    if len(t_arr) == 0:
        print("  Signal: NONE (no finite t-stats — too few signals or insufficient history)")
    elif (np.abs(t_arr) >= bonferroni_t).any():
        print(f"  Signal: WEAK (some |t| > Bonferroni threshold but small N, "
              f"max |t| = {np.max(np.abs(t_arr)):.2f})")
    else:
        print(
            f"  Signal: NONE  (max |t_excess| = {np.max(np.abs(t_arr)):.2f}, "
            f"Bonferroni threshold = {bonferroni_t:.2f})"
        )
    print("  Tradability: MIRAGE  (rare patterns, small sample, selection bias unavoidable)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
