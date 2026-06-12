"""Offline, deterministic demo — Study 77 (Golden-Mean).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
Fibonacci retracement levels only beat placebo (randomly-placed) levels when the tape
actually carries mean reversion — and on a martingale (mean_rev=0) both are fair coins.
Run:

    python studies/77-golden-mean/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from golden_mean import data, strategy as st  # noqa: E402


def _run_arm(bars, use_placebo, swings, fwd_bars=5, cost=0.0):
    ledger = st.run_bounces(bars, swings, fwd_bars=fwd_bars, cost_bps=cost,
                            use_placebo=use_placebo)
    return st.summarize(ledger, col="ret_gross")


def main() -> None:
    print("Study 77 — Golden-Mean — synthetic positive control\n")
    print(f"{'mean_rev':>10} | {'Fib bps':>9} {'win':>6} {'t':>6} | "
          f"{'Plac bps':>9} {'win':>6} {'t':>6} | Fib beats placebo?")
    print("-" * 85)
    for mean_rev in (0.00, 0.15, 0.30, -0.15):
        bars, _ = data.synthetic_daily(n_days=500, mean_rev=mean_rev, seed=77)
        swings = st.find_swings(bars, lookback=40, min_swing_pct=0.03)

        fib = _run_arm(bars, use_placebo=False, swings=swings)
        plac = _run_arm(bars, use_placebo=True, swings=swings)

        beats = ("yes" if (fib["mean_bps"] > plac["mean_bps"] + 0.5
                           and fib["n_touches"] > 10)
                 else "no")
        fib_win = fib["bounce_rate"] if fib["n_touches"] else float("nan")
        pl_win = plac["bounce_rate"] if plac["n_touches"] else float("nan")
        print(f"{mean_rev:>10.2f} | {fib['mean_bps']:>+9.2f} {fib_win:>6.3f} "
              f"{fib['tstat']:>+6.2f} | {plac['mean_bps']:>+9.2f} {pl_win:>6.3f} "
              f"{plac['tstat']:>+6.2f} | {beats}")

    print("\nThe Fibonacci engine is a faithful mean-reversion probe:")
    print("  it only edges the placebo when the tape has real mean reversion planted,")
    print("  and on a martingale (0.00) both arms are fair coins.")
    print("On the real daily tape there is no such effect — see docs/results.md.")


if __name__ == "__main__":
    main()
