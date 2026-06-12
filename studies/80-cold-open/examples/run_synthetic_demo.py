"""Offline, deterministic demo — Study 80 (Cold-Open).

No network.  Builds a synthetic annual tape and shows the study's spine: the
January Barometer (sign(Jan) predicts Feb-Dec direction) only shows a real hit-rate
edge when the signal is planted, and on the null tape (signal_strength = 0) it
behaves like a biased coin driven by equity drift rather than genuine prediction.
Run:

    python studies/80-cold-open/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cold_open import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 80 — Cold-Open — synthetic January Barometer sweep\n")
    print(f"{'signal_strength':>16} | {'hit_rate':>9} {'pval/coin':>10} {'contrast%':>10}"
          f" {'HAC-t':>7} | signal?")
    print("-" * 72)

    for strength in (0.0, 0.15, 0.30, 0.50, -0.20):
        df, _ = data.synthetic_annual(n_years=75, signal_strength=strength, seed=80)
        result = st.summarize(df)
        sig = (
            "yes (planted)" if strength > 0 and result["pval_vs_coin"] < 0.10
            else "no" if strength == 0.0
            else "weak" if result["pval_vs_coin"] < 0.10
            else "no"
        )
        print(
            f"{strength:>16.2f} | {result['hit_rate']:>9.3f} {result['pval_vs_coin']:>10.4f}"
            f" {result['contrast_pct']:>10.2f} {result['tstat_hac']:>7.2f} | {sig}"
        )

    print()
    print("Key insight: on the null tape, a hit-rate > 50% can emerge from equity drift")
    print("(markets go up most years regardless of January) — not from any predictive signal.")
    print("The pval_vs_base_rate test exposes this: does Jan beat *knowing markets trend up*?")
    print()
    print("On the real tape (fetch_gspc) the contrast (t~3.9) is real but post-1985 it")
    print("weakens (t~2.0), and the barometer strategy doesn't beat buy-and-hold.")
    print("See docs/results.md for the full real-tape run.")


if __name__ == "__main__":
    main()
