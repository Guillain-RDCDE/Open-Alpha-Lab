# -*- coding: utf-8 -*-
"""Offline, deterministic demo -- Study 162 (Rosh-Hashanah).

No network.  Builds a synthetic annual tape and shows the study's spine in one screen:
the holiday-window drag is detectable only when genuinely planted -- and when nothing is
planted (the null), the holiday window looks indistinguishable from any other 8-day slice.
This mirrors what the real tape shows: HAC t near zero, verdict NONE.

Run:
    python studies/162-rosh-hashanah/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rosh_hashanah import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 162 — Rosh-Hashanah — synthetic positive control\n")
    print(
        f"{'planted drag (bps)':>20} | {'window mean (bps)':>17} {'t':>6} | "
        f"{'null mean':>9} {'detectable?':>11}"
    )
    print("-" * 75)

    null_df, _ = data.synthetic_annual(n_years=80, holiday_drag_bp=0.0, seed=162)
    null_stat = st.summarize(null_df, col="window_ret")

    for drag in (0.0, -50.0, -100.0, -200.0):
        df, _ = data.synthetic_annual(n_years=80, holiday_drag_bp=drag, seed=162)
        s = st.summarize(df, col="window_ret")
        detectable = "yes" if abs(s["tstat"]) > 2.0 and s["mean_bps"] < 0 else "no"
        print(
            f"{drag:>20.0f} | {s['mean_bps']:>+17.2f} {s['tstat']:>+6.2f} | "
            f"{null_stat['mean_bps']:>+9.2f} {detectable:>11}"
        )

    print()
    print("On the null (drag = 0) the holiday window is indistinguishable from any 8-day slice.")
    print("The drag must be >100 bps before it becomes reliably detectable with n=80 years.")
    print("The REAL tape: mean = -0.87 bps, HAC t = -0.02 => NONE (see docs/results.md).")
    print()
    print("The 2008 effect: the entire measured 'holiday weakness' is one observation.")
    print("Excluding 2008, the mean flips to +38.6 bps — a positive season.")


if __name__ == "__main__":
    main()
