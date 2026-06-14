"""Offline, deterministic demo — Study 137 (Mansfield-RS).

No network.  Builds a set of synthetic weekly tapes and shows the study's spine
in one screen: the Stage-2 Mansfield-RS filter only beats a random-entry control
when the tape actually carries RS momentum — and on a martingale (rs_momentum = 0)
it adds nothing.  Run:

    python studies/137-mansfield-rs/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mansfield_rs import data, strategy as st  # noqa: E402


def _pool(tapes: dict, truth: dict, hold: int, seed: int, use_s2: bool) -> dict:
    bm = tapes["SPY"]
    all_leds = []
    for tk in truth["tickers"]:
        stock = tapes[tk]
        if use_s2:
            entries = st.stage2_entries(stock, bm, sma_n=30)
        else:
            entries = st.stage2_entries(stock, bm, sma_n=30)  # same entry times
            entries_rnd = st.random_entries(stock, n_entries=max(len(entries), 5), seed=seed)
            entries = entries_rnd
        if entries.empty:
            continue
        led = st.run_forward_returns(stock, bm, entries, hold_weeks=hold, cost_bps=0)
        all_leds.append(led)
    pooled = pd.concat(all_leds, ignore_index=True) if all_leds else pd.DataFrame()
    return st.summarize(pooled, "ret_gross") if not pooled.empty else {}


def main() -> None:
    print("Study 137 — Mansfield-RS — synthetic positive control\n")
    print(f"{'planted RS momentum':>22} | {'stage2 bps':>10} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | filter beats random?")
    print("-" * 78)

    for mom in (0.00, 0.15, 0.30, 0.40):
        tapes, truth = data.synthetic_weekly(n_weeks=500, rs_momentum=mom, seed=137)
        s2 = _pool(tapes, truth, hold=13, seed=137, use_s2=True)
        rnd = _pool(tapes, truth, hold=13, seed=137, use_s2=False)
        if not s2 or not rnd:
            continue
        beats = "yes" if s2["mean_bps"] > rnd["mean_bps"] + 1 else "no"
        print(f"{mom:>22.2f} | {s2['mean_bps']:>+10.1f} {s2['tstat']:>+6.2f} | "
              f"{rnd['mean_bps']:>+10.1f} {rnd['tstat']:>+6.2f} | {beats}")

    print("\nThe Stage-2 filter is a faithful RS-momentum harvester: it only beats")
    print("a random-entry control when RS momentum is actually planted in the tape.")
    print("On a martingale (0.00) the timing adds nothing.")
    print("On the real tape (see docs/results.md), signal t = +0.72 — a coin.")


if __name__ == "__main__":
    main()
