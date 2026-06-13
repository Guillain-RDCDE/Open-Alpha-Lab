"""Offline deterministic demo for Study 117 (Pi-Cycle-Top).

Runs entirely on the synthetic tape: no network, no cache required. Prints
the study's spine (signal count on null tape, forward returns on planted tape,
power analysis) and exits 0.

Usage:
    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from pi_cycle_top import data, strategy as st


def main() -> int:
    print("=" * 65)
    print("Study 117 — Pi-Cycle-Top  |  SYNTHETIC DEMO (offline)")
    print("=" * 65)

    # ---- 1. Null tape: nothing to find ----------------------------------------
    null_bars, null_truth = data.synthetic_daily(n_years=10, top_sharpness=0.0, seed=117)
    null_sigs = st.pi_cycle_signals(null_bars)
    null_dates = list(null_sigs["signal_date"]) if len(null_sigs) > 0 else []
    print(f"\n[Null tape] Pi-Cycle signals: {len(null_sigs)}")
    print(f"  n_days={null_truth['n_days']}, top_sharpness={null_truth['top_sharpness']}")
    if null_dates:
        fr = st.forward_returns(null_bars, null_dates, horizon_days=90)
        s = st.summarize(fr["log_ret"].values, label="null 90d")
        print(f"  90d forward mean: {s['mean']:.3f}, t={s['tstat']:.3f}, low_power={s['low_power_warning']}")
    else:
        print("  No signals on null tape (expected).")

    # ---- 2. Planted tape: effect is there, but n=3 kills power ---------------
    sig_bars, sig_truth = data.synthetic_daily(n_years=10, top_sharpness=0.80, seed=117)
    planted = sig_truth["planted_signals"]
    fr90 = st.forward_returns(sig_bars, planted, horizon_days=90)
    s90 = st.summarize(fr90["log_ret"].values, label="planted 90d")
    print(f"\n[Planted tape] top_sharpness={sig_truth['top_sharpness']}, n_signals={len(planted)}")
    print(f"  90d forward returns:")
    for i, row in fr90.iterrows():
        print(f"    signal {i+1}: log_ret={row['log_ret']:.3f} ({row['pct_ret']*100:.1f}%)")
    print(f"  Summary: n={s90['n']}, mean={s90['mean']:.3f}, t={s90['tstat']:.3f}")
    print(f"  low_power_warning={s90['low_power_warning']}  <- n=3 kills inference even here")

    # ---- 3. Multiplier sensitivity -------------------------------------------
    print("\n[Multiplier sweep] How many signals at each 2x variant?")
    alt = st.alternative_ma_signals(sig_bars)
    for mult, sigs in alt.items():
        print(f"  mult={mult:.1f}: {len(sigs)} signals")

    # ---- 4. Power analysis ---------------------------------------------------
    if len(planted) >= 2:
        pa = st.power_analysis(sig_bars, planted, horizon_days=90, n_draws=500, seed=117)
        print(f"\n[Power analysis] obs_mean={pa['obs_mean_log_ret']:.3f}, "
              f"ctrl_mean={pa['ctrl_mean']:.3f}, p_value={pa['p_value']:.3f}")
        print(f"  -> Even on a planted tape, n={pa['n_events']} events gives p={pa['p_value']:.2f}")

    # ---- 5. Fingerprint -------------------------------------------------------
    fp = data.fingerprint(null_bars)
    print(f"\nFingerprint (null tape): {fp}")

    print("\n[VERDICT (synthetic)] n=3 events; the indicator scores well on "
          "planted tops but inference is impossible. On the null tape it may fire "
          "spuriously — the structure of the market, not pi, drives the MA cross.")
    print("\nDemo complete. Exit 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
