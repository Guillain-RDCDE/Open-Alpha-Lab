"""Offline, deterministic demo — Study 178 (CCI).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the CCI oversold/overbought rule only beats a random-direction coin when the tape carries
mean-reversion — and on a martingale (mean_rev = 0) it is a fair die.  Run:

    python studies/178-cci/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cci import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 178 — CCI — synthetic positive control\n")
    print(f"{'planted mean-rev':>18} | {'CCI bps':>9} {'win':>6} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | CCI beats coin?")
    print("-" * 82)
    for mr in (0.00, 0.10, 0.25, 0.40, -0.10):
        bars, _ = data.synthetic_daily(n_days=1000, mean_rev=mr, seed=178)
        c = st.cci(bars, period=20)
        ent = st.breach_entries(c)
        if len(ent) < 5:
            print(f"{mr:>18.2f} | (too few entries on this tape)")
            continue
        cci_s = st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross")
        rand_s = st.summarize(
            st.run_trades(
                bars, ent, hold_days=5, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if cci_s["mean_bps"] > rand_s["mean_bps"] + 0.5 else "no"
        print(f"{mr:>18.2f} | {cci_s['mean_bps']:>+9.2f} {cci_s['win_rate']:>6.3f} "
              f"{cci_s['tstat']:>+6.2f} | {rand_s['mean_bps']:>+10.2f} {rand_s['tstat']:>+6.2f} "
              f"| {beats}")

    print("\nNote: at very high mean-reversion (>=0.60) CCI consistently edges a coin.")
    print("The real daily equity tape has little mean-reversion at this horizon.")
    print("On equities, 'oversold' often just means 'in a downtrend' — see docs/results.md.")


if __name__ == "__main__":
    main()
