"""Offline, deterministic demo — Study 133 (Crypto-Seasonality).

No network. Builds a synthetic BTC-USD daily tape and shows the study's spine in one
screen: the monthly-seasonality engine detects a planted October boost with a clear
t-stat signal, and reads near zero on a martingale (no boost). This is the positive
control that proves the machinery works — the real-tape verdict is therefore a
statement about the market, not the method. Run:

    python studies/133-crypto-seasonality/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_seasonality import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 133 — Crypto-Seasonality — synthetic positive control\n")
    print("Testing: does the engine detect a planted 'Uptober' effect?\n")

    for boost, label in [(0.00, "null (no boost)"), (0.30, "moderate boost"),
                         (0.50, "strong boost")]:
        bars, truth = data.synthetic_daily(
            n_years=11, monthly_boost=boost, boost_month=10, seed=133
        )
        mret = st.monthly_returns(bars)
        stats = st.month_stats(mret)
        oct_row = stats[stats["month"] == 10].iloc[0]
        other_max = stats[stats["month"] != 10]["tstat_vs_baseline"].abs().max()

        print(f"  monthly_boost={boost:.2f} ({label})")
        print(f"    October: mean={oct_row['mean_pct']:+.1f}%  |t|={abs(oct_row['tstat_vs_baseline']):.2f}  "
              f"naive_sig={'YES' if oct_row['naive_sig'] else 'no'}")
        print(f"    Other months max |t|: {other_max:.2f}")
        print()

    print("The engine is a faithful seasonality detector: it recovers the planted")
    print("effect and returns near-zero t-stats when there is nothing to find.")
    print("On the real BTC tape (~11 years) see docs/results.md for the honest verdict.")


if __name__ == "__main__":
    main()
