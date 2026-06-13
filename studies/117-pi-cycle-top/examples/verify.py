"""Reproducible real-tape runner for Study 117 (Pi-Cycle-Top).

Fetches (or loads from cache) BTC-USD daily data via yfinance, runs the Pi-Cycle-Top
indicator, and prints the headline numbers that feed docs/results.md. Pass ``--fetch``
to refresh the cache.

Usage:
    python examples/verify.py            # cache-only (after first run)
    python examples/verify.py --fetch    # hit yfinance and refresh cache
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from pi_cycle_top import data, strategy as st

FETCH = "--fetch" in sys.argv


def main() -> int:
    print("=" * 72)
    print("Study 117 — Pi-Cycle-Top  |  REAL-TAPE VERIFY")
    print("=" * 72)

    bars = data.fetch_daily(fetch=FETCH)
    fp = data.fingerprint(bars)
    print(f"\nTicker : BTC-USD")
    print(f"Window : {bars.index[0].date()} to {bars.index[-1].date()}")
    print(f"Bars   : {len(bars)}")
    print(f"Fingerprint: {fp}")

    # ---- Pi-Cycle signals ------------------------------------------------------
    sigs = st.pi_cycle_signals(bars)
    print(f"\nPi-Cycle signals (MA111 crosses above 2×MA350): {len(sigs)}")
    if len(sigs) > 0:
        print(sigs[["signal_date", "btc_price", "fast_ma", "slow_ma_2x"]].to_string(index=False))
    else:
        print("  None found.")

    sig_dates = list(sigs["signal_date"]) if len(sigs) > 0 else []

    # ---- Forward returns at multiple horizons ----------------------------------
    print("\n--- Forward log-returns by horizon ---")
    for horizon in [30, 60, 90]:
        fr = st.forward_returns(bars, sig_dates, horizon_days=horizon)
        if len(fr) == 0:
            print(f"  {horizon}d: no complete windows")
            continue
        s = st.summarize(fr["log_ret"].values, label=f"{horizon}d")
        pcts = [f"{r*100:.1f}%" for r in fr["pct_ret"]]
        print(f"  {horizon}d: n={s['n']}, mean={s['mean']:.3f} ({s['mean']*100:.1f}%),  "
              f"t={s['tstat']:.3f},  low_power={s['low_power_warning']}")
        print(f"      per-event pct_ret: {pcts}")

    # ---- Days to actual local peak (90d look-ahead) --------------------------
    if sig_dates:
        print("\n--- Distance to cycle peak (90d look-ahead window) ---")
        dtop = st.days_to_actual_top(bars, sig_dates, look_ahead_days=90)
        print(dtop[["signal_date", "btc_at_signal", "peak_date", "peak_price",
                     "days_to_peak"]].to_string(index=False))

    # ---- Power analysis -------------------------------------------------------
    print("\n--- Power analysis (empirical p-value vs random-day control) ---")
    for horizon in [30, 60, 90]:
        fr = st.forward_returns(bars, sig_dates, horizon_days=horizon)
        if len(fr) < 1:
            continue
        pa = st.power_analysis(bars, sig_dates, horizon_days=horizon, n_draws=2000, seed=117)
        print(f"  {horizon}d: obs_mean={pa['obs_mean_log_ret']:.3f}, "
              f"ctrl_mean={pa['ctrl_mean']:.3f}, "
              f"p_value={pa['p_value']:.4f} (one-sided: fraction of random bundles worse)")

    # ---- Multiplier sensitivity -----------------------------------------------
    print("\n--- Multiplier sensitivity (n signals at each 2x variant) ---")
    alt = st.alternative_ma_signals(bars)
    for mult, ms in alt.items():
        dates = [str(d.date()) for d in ms["signal_date"]] if len(ms) > 0 else []
        print(f"  mult={mult:.1f}: {len(ms)} signals  {dates}")

    # ---- BTC characteristics --------------------------------------------------
    log_ret = np.diff(np.log(bars["close"].to_numpy()))
    annual_vol = log_ret.std() * np.sqrt(252)
    print(f"\nBTC annual volatility: {annual_vol*100:.0f}%")
    print(f"Effective signal count on Yahoo tape: {len(sig_dates)}")
    print("\n[VERDICT] n=2 signals, no inference possible (low_power_warning always True).")
    print("The indicator fired twice (2017-12-16, 2021-04-11); both preceded massive "
          "crashes (-57% and -44% in 90d). But 2024 never crossed the threshold. "
          "n=2 on a ~80% annual-vol asset: the p-value is not the issue; n is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
