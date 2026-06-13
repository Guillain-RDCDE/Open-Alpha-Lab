"""Offline, deterministic demo — Study 113 (Gold-Silver-Ratio).

No network. Builds a synthetic daily pair and shows the study's spine in one screen:
the z-score mean-reversion strategy only beats a random-direction coin when the ratio
actually carries genuine mean-reversion — and on a random walk (mean_rev_speed=0) it
is a fair die. Run:

    python studies/113-gold-silver-ratio/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gold_silver_ratio import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 113 — Gold-Silver-Ratio — synthetic positive control\n")
    print(f"{'mean_rev_speed':>16} | {'half_life':>10} | {'strat bps':>10} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | strat beats coin?")
    print("-" * 90)
    for speed in (0.00, 0.05, 0.10, 0.15, 0.20):
        gold, silver, truth = data.synthetic_daily(
            n_days=2000, mean_rev_speed=speed, seed=113
        )
        ratio = st.compute_ratio(gold, silver)
        entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
        hl = truth["planted_half_life"]
        hl_str = f"{hl:.1f}d" if hl != float("inf") else "  inf"

        if len(entries) == 0:
            print(f"{speed:>16.2f} | {hl_str:>10} | {'(no entries)':>10} {'':>6} | "
                  f"{'(no entries)':>10} {'':>6} |")
            continue

        ledger = st.run_trades(gold, silver, entries, cost_bps=0.0)
        rand_ent = st.random_entries(entries, seed=42)
        rand_led = st.run_trades(gold, silver, rand_ent, cost_bps=0.0)
        s_strat = st.summarize(ledger, "ret_gross")
        s_rand = st.summarize(rand_led, "ret_gross")
        beats = "yes" if s_strat["tstat"] > 2.0 and s_strat["mean_bps"] > s_rand["mean_bps"] else "no"
        print(
            f"{speed:>16.2f} | {hl_str:>10} | "
            f"{s_strat['mean_bps']:>+10.1f} {s_strat['tstat']:>+6.2f} | "
            f"{s_rand['mean_bps']:>+10.1f} {s_rand['tstat']:>+6.2f} | {beats}"
        )

    print("\nWith planted mean-reversion (speed>0) the strategy earns significant t-stats;")
    print("without it (speed=0.00) the result is noisy and the t-stat is well below 2.")
    print("On the real tape see docs/results.md for the honest verdict.")


if __name__ == "__main__":
    main()
