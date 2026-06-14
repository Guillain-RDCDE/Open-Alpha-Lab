"""Offline, deterministic demo — Study 141 (Turnover-Anomaly).

No network. Builds two synthetic panels (null and signal) and shows the study's
spine in one screen: the low-turnover Q1 portfolio only beats the market and the
shuffled control when a genuine negative premium is planted in the tape.

    python studies/141-turnover-anomaly/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turnover_anomaly import data, strategy as st  # noqa: E402


def run_one(label: str, premium: float, n_firms: int = 120, n_years: int = 14, seed: int = 141):
    sig, fwd, truth = data.synthetic_panel(
        n_firms=n_firms, n_years=n_years, premium=premium, seed=seed
    )
    real_res = st.quintile_returns(sig, fwd)
    ctrl = st.shuffled_control(sig, fwd, n_shuffles=200, seed=0)

    real_hedge = st.summarize(real_res["hedge"])
    real_spread = st.summarize(real_res["spread"])
    shuffle_mean = float(ctrl.mean(axis=None).mean())

    print(f"\n{'='*60}")
    print(f"  {label}  (premium={premium:+.2f}, n={n_firms} firms, {n_years} years)")
    print(f"{'='*60}")
    print(f"  Q1−market (hedge):  mean={real_hedge['mean']:+.3f}  t={real_hedge['tstat']:+.5f}"
          f"  hit={real_hedge['hit_rate']:.0%}")
    print(f"  Q1−Q5 (spread):     mean={real_spread['mean']:+.3f}  t={real_spread['tstat']:+.5f}"
          f"  hit={real_spread['hit_rate']:.0%}")
    print(f"  Shuffle Q1−market:  mean={shuffle_mean:+.3f}  (null distribution)")
    beats = "YES" if real_hedge["mean"] > shuffle_mean + 0.005 else "no"
    print(f"  Signal beats shuffle? {beats}")


def main() -> None:
    print("Study 141 - Turnover-Anomaly - synthetic positive control")
    print("Datar-Naik-Radcliffe (1998): high turnover stocks underperform")
    print()

    print("Profile across planted premium strengths:")
    print(f"{'premium':>10} | {'Q1-mkt mean':>12} {'Q1-mkt t':>10} | {'Q1-Q5 mean':>11} {'Q1-Q5 t':>8} | beats shuffle?")
    print("-" * 75)

    for premium in (0.00, -0.02, -0.05, -0.10):
        sig, fwd, _ = data.synthetic_panel(
            n_firms=150, n_years=15, premium=premium, seed=141
        )
        real_res = st.quintile_returns(sig, fwd)
        ctrl = st.shuffled_control(sig, fwd, n_shuffles=200, seed=0)

        hedge = st.summarize(real_res["hedge"])
        spread = st.summarize(real_res["spread"])
        shuffle_mean = float(ctrl.mean(axis=None).mean())
        beats = "yes" if hedge["mean"] > shuffle_mean + 0.005 else "no"

        print(f"{premium:>10.2f} | {hedge['mean']:>+12.4f} {hedge['tstat']:>+10.4f} | "
              f"{spread['mean']:>+11.4f} {spread['tstat']:>+8.4f} | {beats}")

    print()
    print("Quintile profile (premium = -0.06):")
    sig, fwd, _ = data.synthetic_panel(n_firms=150, n_years=15, premium=-0.06, seed=141)
    profile = st.quintile_profile(sig, fwd)
    print(f"  {'Quintile':>8} {'Mean turnover':>14} {'Mean fwd ret':>14} {'n_obs':>7}")
    for q, row in profile.iterrows():
        print(f"  Q{q:1d} (low to high) {row['mean_turnover']:>14.3f} {row['mean_fwd_ret']:>+14.4f} {row['n_obs']:>7.0f}")

    print()
    print("The engine is faithful: Q1 outperforms the shuffle only when a premium is planted.")
    print("On the real tape, check docs/results.md for the honest verdict.")


if __name__ == "__main__":
    main()
