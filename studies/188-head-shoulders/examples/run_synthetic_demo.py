"""Offline, deterministic demo — Study 188 (Head-Shoulders).

No network.  Builds synthetic daily tapes with and without injected head-and-shoulders
patterns and shows the study's spine in one screen:

- On a pure random walk, the detector fires at its background rate and forward
  returns after a confirmed neckline break are indistinguishable from random.
- When a real H&S structure is planted, the detector fires more and (by construction)
  the forward return in the predicted direction is mechanically positive.

This is the *positive-control check*: the engine works when there is something to
find.  On the real tape (see ``examples/verify.py``) the pattern exists in the data
but its directional prediction is not statistically distinguishable from noise.

Run:
    python studies/188-head-shoulders/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from head_shoulders import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 188 — Head-Shoulders — synthetic positive control\n")
    print(
        f"{'hs_strength':>12} | {'n_top':>5} {'n_inv':>5} | "
        f"{'top_1d%':>8} {'top_t':>6} | {'inv_1d%':>8} {'inv_t':>6}"
    )
    print("-" * 72)

    for strength in (0.0, 0.05, 0.10, 0.15, 0.20):
        bars, truth = data.synthetic_daily(
            n_days=1000,
            hs_strength=strength,
            n_patterns=4,
            pattern_width=80,
            seed=188,
        )
        result = st.run_hs_study(bars, horizons=(1, 5, 20))

        n_top = result["n_top"]
        n_inv = result["n_inv"]
        ctrl = result["control"]

        if n_top > 0:
            s_top = st.summarize(result["fwd_top"], direction=-1, horizon=1, control=ctrl)
            top_mean = s_top.get("mean_pct", float("nan"))
            top_t = s_top.get("tstat", float("nan"))
        else:
            top_mean = float("nan")
            top_t = float("nan")

        if n_inv > 0:
            s_inv = st.summarize(result["fwd_inv"], direction=+1, horizon=1, control=ctrl)
            inv_mean = s_inv.get("mean_pct", float("nan"))
            inv_t = s_inv.get("tstat", float("nan"))
        else:
            inv_mean = float("nan")
            inv_t = float("nan")

        print(
            f"{strength:>12.2f} | {n_top:>5} {n_inv:>5} | "
            f"{top_mean:>+8.3f} {top_t:>+6.2f} | "
            f"{inv_mean:>+8.3f} {inv_t:>+6.2f}"
        )

    print()
    print("The engine detects patterns when they're present.")
    print("On the real tape, patterns exist but carry no reliable directional edge.")
    print("See docs/results.md for the full verdict on historical SPY/QQQ data.")


if __name__ == "__main__":
    main()
