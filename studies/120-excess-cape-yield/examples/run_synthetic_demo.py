"""Offline, deterministic demo -- Study 120 (Excess-CAPE-Yield).

No network. Builds synthetic ECY worlds with varying predictive power and shows
the study's spine in one screen: ECY forecasts 10-year excess returns *when
predictive power is planted* (predicts=1.0) and has no such ability *when it
isn't* (predicts=0.0). Run:

    python studies/120-excess-cape-yield/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from excess_cape_yield import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 120 -- Excess-CAPE-Yield -- synthetic positive control\n")
    print(
        "  predicts | n_nonoverlap | R2(10yr)   slope       t |  R2(1yr) | OOS R2"
    )
    print("-" * 78)

    for predicts in (0.0, 0.5, 1.0):
        frame, truth = data.synthetic_world(n_years=130, predicts=predicts, seed=120)
        s = st.summarize(frame)
        print(
            f"{predicts:>10.1f} | {s['n_nonoverlap_10yr']:>13d} | "
            f"{s['r2_ecy_nonoverlap']:>9.3f} "
            f"{s['slope_ecy_nonoverlap']:>7.3f} "
            f"{s['tstat_ecy_nonoverlap']:>7.2f} | "
            f"{s['r2_ecy_1yr_monthly']:>8.3f} | "
            f"{s['oos_r2_ecy']:>+.3f}"
        )

    print()
    print("Interpretation:")
    print("  predicts=0.0 -> ECY is pure noise; regression finds nothing, OOS R2 ~ 0 or negative.")
    print("  predicts=1.0 -> ECY is the true signal; R2(10yr) is high, t >> 2, OOS R2 > 0.")
    print("  predicts=0.5 -> partial predictability; intermediate R2, weaker t.")
    print()
    print("The horizon effect is stark at all predicts values:")
    for predicts in (0.0, 1.0):
        frame, _ = data.synthetic_world(n_years=130, predicts=predicts, seed=120)
        s = st.summarize(frame)
        print(
            f"  predicts={predicts}: R2(1yr)={s['r2_ecy_1yr_monthly']:.3f}  "
            f"R2(10yr,monthly)={s['r2_ecy_monthly']:.3f}"
        )
    print()
    print("At the 1-year horizon the signal is swamped by noise -- ECY is a tide table,")
    print("not a stopwatch. This matches the real data and is the key tradability caveat.")


if __name__ == "__main__":
    main()
