"""Offline, deterministic demo — Study 184 (Williams-Fractals).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the fractal *breakout* rule only beats a random-direction coin when the tape actually
carries bar-level momentum — and on a martingale (momentum=0) it is a fair die.  Run:

    python studies/184-williams-fractals/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from williams_fractals import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 184 — Williams-Fractals — synthetic positive control\n")
    print(f"{'momentum':>12} | {'n_frac_brk':>10} | {'brk bps':>9} {'win':>6} {'t':>6} | "
          f"{'rand bps':>9} {'t':>6} | beats coin?")
    print("-" * 82)
    for mom in (0.00, 0.10, 0.20, 0.30, -0.10):
        bars, _ = data.synthetic_daily(n_days=1200, momentum=mom, seed=184)
        f = st.detect_fractals(bars)
        ent = st.breakout_entries(bars, f)
        n_brk = len(ent)
        brk = st.summarize(
            st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, hold_days=5, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if brk["mean_bps"] > rand["mean_bps"] + 0.05 else "no"
        wr = brk["win_rate"] if brk["win_rate"] == brk["win_rate"] else 0.0
        print(f"{mom:>12.2f} | {n_brk:>10d} | "
              f"{brk['mean_bps']:>+9.2f} {wr:>6.3f} {brk['tstat']:>+6.2f} | "
              f"{rand['mean_bps']:>+9.2f} {rand['tstat']:>+6.2f} | {beats}")

    print()
    print("The breakout rule harvests momentum: it only edges the coin when persistence")
    print("is planted (+0.10/+0.20/+0.30). On a martingale (0.00) it is a fair die.")
    print("On the real daily tape there is no such momentum — see docs/results.md.")


if __name__ == "__main__":
    main()
