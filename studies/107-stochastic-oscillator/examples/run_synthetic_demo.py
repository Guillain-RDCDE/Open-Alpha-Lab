"""Offline, deterministic demo — Study 107 (Stochastic-Oscillator).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the Stochastic %K/%D oversold/overbought crossover only beats a random-direction coin
when the tape actually carries short-term mean-reversion — and on a martingale
(mean_rev = 0) it is a fair die.  Run:

    python studies/107-stochastic-oscillator/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stochastic_oscillator import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 107 — Stochastic-Oscillator — synthetic positive control\n")
    print(
        f"{'planted mean_rev':>18} | {'n sigs':>6} | {'stoch bps':>9} {'win':>6} {'t':>6} | "
        f"{'random bps':>10} {'t':>6} | stoch beats coin?"
    )
    print("-" * 88)
    for mean_rev in (0.00, 0.15, 0.25, -0.15):
        bars, _ = data.synthetic_daily(n_days=600, mean_rev=mean_rev, seed=107)
        osc = st.stochastic(bars)
        sigs = st.crossover_signals(osc["pct_k"], osc["pct_d"], zone_filter=True)
        if len(sigs) == 0:
            print(f"{mean_rev:>18.2f} | {'0':>6} | (no signals)")
            continue
        cross = st.summarize(
            st.run_trades(bars, sigs, hold_days=5, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, sigs, hold_days=5, cost_bps=0,
                directions=st.random_directions(len(sigs), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if cross["mean_bps"] > rand["mean_bps"] + 0.5 else "no"
        print(
            f"{mean_rev:>18.2f} | {len(sigs):>6} | "
            f"{cross['mean_bps']:>+9.2f} {cross['win_rate']:>6.3f} {cross['tstat']:>+6.2f} | "
            f"{rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} | {beats}"
        )

    print(
        "\nThe stochastic is a mean-reversion detector: on tapes with planted negative")
    print("autocorrelation (+0.15/+0.25) it consistently beats the coin; on a trending")
    print("(-0.15) tape it cannot. The real tape verdict is in docs/results.md.")


if __name__ == "__main__":
    main()
