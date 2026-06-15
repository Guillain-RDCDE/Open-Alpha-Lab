"""Offline, deterministic demo — Study 175 (Crypto-Weekend).

No network. Builds synthetic daily BTC-like tapes and shows the study's spine in one
screen: the engine detects a planted weekend-return boost when one exists, and reads
nothing when the tape is a pure random walk. Run:

    python studies/175-crypto-weekend/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_weekend import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 175 — Crypto-Weekend — synthetic positive control\n")
    print(
        f"{'planted weekend boost':>22} | "
        f"{'wknd mean%':>10} {'wkday mean%':>11} {'premium%':>9} {'t':>6} | "
        f"vol ratio | detected?"
    )
    print("-" * 90)

    for boost in (0.0, 0.02, 0.05, -0.02):
        bars, _ = data.synthetic_daily(n_years=10, weekend_boost=boost, seed=175)
        daily = st.daily_returns(bars)
        rc = st.return_comparison(daily)
        vc = st.volatility_comparison(daily)

        wknd_m = rc["weekend"]["mean_pct"]
        wkday_m = rc["weekday"]["mean_pct"]
        prem = rc["premium_pct"]
        t = rc["diff_tstat"]
        vol_ratio = vc["vol_ratio"]
        detected = "YES" if abs(t) >= 2.0 and (prem > 0) == (boost > 0) else "no"

        print(
            f"{boost:>22.3f} | "
            f"{wknd_m:>+10.4f} {wkday_m:>+11.4f} {prem:>+9.4f} {t:>+6.2f} | "
            f"{vol_ratio:>9.3f} | {detected}"
        )

    print()
    print("At boost=0 (the null) the premium t-stat is small — no spurious detection.")
    print("A planted boost of +0.05/day shows up; the engine is honest about direction.")
    print("On the REAL BTC tape there is no significant weekend premium — see docs/results.md.")


if __name__ == "__main__":
    main()
