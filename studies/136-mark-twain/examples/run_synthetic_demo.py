"""Offline, deterministic demo -- Study 136 (Mark-Twain).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the analysis detects an October penalty only when one is planted -- and on a flat null
(no monthly seasonality) October looks just like any other month. Run:

    python studies/136-mark-twain/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import io
import os
import sys

# Force UTF-8 output on Windows so the print statements do not trip on special chars.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mark_twain import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 136 -- Mark-Twain -- synthetic positive control\n")
    print("Testing: do we find an October effect only when one is planted?\n")

    configs = [
        ("null (no penalty)",        0.0),
        ("mild drag (-50 bps/d)",   -50.0),
        ("strong drag (-150 bps/d)", -150.0),
    ]

    print(
        f"{'Scenario':32s} | {'Oct':>8} {'Rest':>8} {'Diff':>8} {'Welch-t':>8} | Detected?"
    )
    print("-" * 82)
    for label, penalty in configs:
        ret, _ = data.synthetic_daily(n_years=80, october_penalty_bp=penalty, seed=136)
        cmp = st.compare_october_vs_rest(ret)
        detected = "YES" if cmp["welch_t"] < -2.0 else "no"
        print(
            f"{label:32s} | "
            f"{cmp['oct_mean_bps']:>+8.2f} {cmp['rest_mean_bps']:>+8.2f} "
            f"{cmp['diff_bps']:>+8.2f} {cmp['welch_t']:>+8.2f} | {detected}"
        )

    print()
    print("The engine is a faithful detector: only a planted penalty yields a significant Welch-t.")
    print("On the real ^GSPC tape (1928-2026), October is POSITIVE (+2.9 bps/day, t=+0.95).")
    print("September is the true worst month (-5.5 bps/day, t=-2.0). October is a myth.")
    print()
    print("Run examples/verify.py with a real cache to regenerate docs/results.md numbers.")


if __name__ == "__main__":
    main()
