"""Offline, deterministic demo — Study 195 (Monthly-OpEx).

No network.  Builds synthetic daily tapes and shows the study's spine:
the vol/range detector only finds an elevated-volume effect when we *plant* it,
and the return detector only fires when we plant a directional drift.  On the null
tape (both premiums = 0) the t-stats sit inside the noise band.  Run:

    python studies/195-monthly-opex/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from monthly_opex import data, strategy as st  # noqa: E402


def _t_vol_week(bars: pd.DataFrame) -> float:
    idx = pd.DatetimeIndex(bars.index)
    mask = data.opex_week_mask(idx)
    lv = st.log_volume_demeaned(bars)
    return st.compare_opex_vs_baseline(lv, mask)["tstat"]


def _t_ret_week(bars: pd.DataFrame) -> float:
    idx = pd.DatetimeIndex(bars.index)
    mask = data.opex_week_mask(idx)
    r = st.daily_return(bars).dropna()
    return st.compare_opex_vs_baseline(r, mask)["tstat"]


def main() -> None:
    print("Study 195 — Monthly-OpEx — synthetic positive control\n")

    # ---- Vol/volume sweep -----------------------------------------------
    print("=== Volume uplift sweep (planted vol_premium) ===")
    print(f"{'planted vol_premium':>22} | {'vol t-stat':>10} | clearly elevated?")
    print("-" * 58)
    for vp in (0.0, 0.10, 0.20, 0.40):
        bars, _ = data.synthetic_daily(n_years=30, vol_premium=vp, ret_premium_bps=0.0, seed=195)
        t = _t_vol_week(bars)
        flag = "yes" if t > 2.0 else "no"
        print(f"{vp:>22.2f} | {t:>+10.2f} | {flag}")

    print()

    # ---- Return sweep -----------------------------------------------
    print("=== Return drift sweep (planted ret_premium_bps) ===")
    print(f"{'planted ret_premium_bps':>24} | {'ret t-stat':>10} | clearly elevated?")
    print("-" * 60)
    for rp in (0.0, 5.0, 10.0, 20.0):
        bars, _ = data.synthetic_daily(n_years=30, vol_premium=0.0, ret_premium_bps=rp, seed=195)
        t = _t_ret_week(bars)
        flag = "yes" if abs(t) > 2.0 else "no"
        print(f"{rp:>24.1f} | {t:>+10.2f} | {flag}")

    print()
    print("The engine finds elevated volume and elevated return only when we plant them.")
    print("On the real SPY tape (see docs/results.md):")
    print("  - Volume effect on OpEx DAY: t=+4.44 (REAL, but driven by quarterly triple-witching)")
    print("  - Monthly-only (non-quarterly) vol effect: t=-0.65 (NONE)")
    print("  - Return effect on OpEx week: t=-0.07 (NONE)")
    print("  => Signal=NONE for the monthly-only effect; Tradability=MIRAGE")


if __name__ == "__main__":
    main()
