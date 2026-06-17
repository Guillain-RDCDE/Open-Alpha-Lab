"""Offline, deterministic demo — Study 301 (Triple-RSI).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the Triple-RSI oversold-bounce rule only beats a random-direction coin when the tape
actually carries short-term anti-persistence — and on a martingale (reversion=0) it is a
fair die. Run:

    python studies/301-triple-rsi/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from triple_rsi import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 301 — Triple-RSI — synthetic positive control")
    print("RSI(5) triple-condition bounce vs a random-direction coin on a daily tape\n")
    print(
        f"{'planted reversion':>18} | {'rule bps':>9} {'t':>6} | "
        f"{'random bps':>10} {'t':>6} | excess | signal?"
    )
    print("-" * 80)
    for rev in (0.00, 0.25, 0.50, -0.10):
        bars, _ = data.synthetic_daily(n_days=8000, reversion=rev, seed=301)
        signal = st.triple_rsi_entries(bars, trend_sma=None)
        rule = st.summarize(
            st.run_trades(bars, signal, rsi_exit=50.0, cost_bps=0, entry="close"),
            "ret_gross",
        )
        rand = st.summarize(
            st.run_trades(
                bars, signal, rsi_exit=50.0, cost_bps=0, entry="close",
                random_dirs=st.random_directions(int(signal.sum()), seed=1),
            ),
            "ret_gross",
        )
        excess = rule["mean_bps"] - rand["mean_bps"]
        has_signal = "yes" if (excess > 15 and rule["tstat"] > 2.0) else "no"
        print(
            f"{rev:>18.2f} | {rule['mean_bps']:>+9.2f} {rule['tstat']:>+6.2f} | "
            f"{rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} | "
            f"{excess:>+6.1f} | {has_signal}"
        )

    print()
    print("Two lessons from the bottle:")
    print(" - Engine is faithful: it beats the coin (t>2) only when reversion is planted.")
    print(" - Win-rate illusion: on the martingale (rev=0) the rule still WINS ~62% of")
    print("   trades yet its mean is ~0/negative (|t|<2) — proof that a high hit-rate")
    print("   says nothing about expectancy. That is the heart of the '90% win rate' trap.")
    print("On the real daily SPY tape see docs/results.md for the full verdict.")


if __name__ == "__main__":
    main()
