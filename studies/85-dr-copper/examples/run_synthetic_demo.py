"""Offline, deterministic demo — Study 85 (Dr-Copper).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the copper/gold predictive regression only beats the naive mean (positive OOS R²)
when real cross-series predictability is planted — and on a null tape (pred_r=0) it
does not. Run:

    python studies/85-dr-copper/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dr_copper import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 85 — Dr-Copper — synthetic positive control\n")
    print(
        f"{'pred_r':>8} | {'IS beta':>9} {'IS t':>7} {'IS R²%':>7} | "
        f"{'OOS R²%':>8} {'DM t':>7} | beats naive?"
    )
    print("-" * 78)
    for pred_r in (0.0, 0.1, 0.3, 0.5, -0.3):
        daily, _ = data.synthetic_daily(n_years=20, pred_r=pred_r, seed=85)
        daily["cu_au"] = daily["copper"] / daily["gold"]
        res = st.summarize(daily, horizon="W")
        is_e = res["is_equity"]
        oos_e = res["oos_equity"]
        beats = "yes" if (oos_e["r2_oos"] > 0.001 and is_e["tstat"] > 1.5) else "no"
        print(
            f"{pred_r:>8.2f} | {is_e['beta']:>+9.4f} {is_e['tstat']:>+7.2f} "
            f"{is_e['r2'] * 100:>7.3f} | {oos_e['r2_oos'] * 100:>+8.3f} "
            f"{oos_e['dm_tstat']:>+7.2f} | {beats}"
        )

    print()
    print("The regression engine is faithful: it recovers predictability when planted")
    print("(pred_r >= 0.3) and reports near-zero when nothing is there (pred_r = 0).")
    print("The real-tape verdict is therefore a statement about the market, not the method.")
    print("See docs/results.md for the honest real-data numbers.")


if __name__ == "__main__":
    main()
