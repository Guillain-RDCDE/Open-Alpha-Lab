"""Real-data verification — the headline tables behind the README verdict.

Downloads ^GSPC + ^VIX (cached after the first run) and prints, in one place, the
numbers that earn each stamp: the random-day null for the level and the spike, the
cross-study control vs a -3% price day, the clustering bootstrap, the window-
selection test, and the martingale ruin sim on the full sample vs the chart window.

    python examples/verify_real.py            # S&P 500 (^GSPC)
    python examples/verify_real.py ndx        # cross-check on the Nasdaq-100

Network required on first run (Yahoo! Finance); offline thereafter via ../_cache/.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fear_gauge import benchmark, data, robustness, triggers

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

MARKET = {"spx": "^GSPC", "ndx": "^NDX"}.get(
    (sys.argv[1].lower() if len(sys.argv) > 1 else "spx"), "^GSPC"
)


def main():
    mkt, vix = data.aligned(MARKET)
    print(f"{MARKET}: {len(mkt):,} sessions  {mkt.index.min().date()} -> {mkt.index.max().date()}")

    fam = {
        "V1_level_30": triggers.first_crossings(triggers.level(vix, 30), cooldown=21),
        "V1_level_50": triggers.first_crossings(triggers.level(vix, 50), cooldown=21),
        "V2_spike": triggers.first_crossings(triggers.spike(vix), cooldown=21),
        "V3_high_base": triggers.first_crossings(triggers.spike_from_base(vix, base_min=30), cooldown=21),
    }
    print("\nfresh events:", {k: int(v.sum()) for k, v in fam.items()})

    for name in ("V1_level_30", "V2_spike"):
        print(f"\n[random-day null] {name}")
        print(benchmark.conditional_vs_unconditional(mkt, fam[name]).round(4))

    price = triggers.first_crossings(mkt["r_cc"] <= -0.03, cooldown=21)
    for name in ("V1_level_30", "V2_spike"):
        print(f"\n[cross-study control] {name} vs price -3%")
        print(benchmark.excess_vs_alternative(mkt, fam[name], price).round(4))

    for name in ("V1_level_30", "V2_spike"):
        print(f"\n[block bootstrap h=21] {name}:",
              {k: round(v, 4) for k, v in robustness.block_bootstrap_excess(mkt, fam[name]).items()})

    print("\n[window sensitivity] V2_spike, h=21")
    print(robustness.window_sensitivity(mkt, fam["V2_spike"]).round(4))

    win = mkt.index >= "2016-06-01"
    print("\n[martingale ruin] full sample vs chart window (2016+)")
    print(pd.DataFrame({
        "full": robustness.martingale_ruin(mkt, vix),
        "window 2016+": robustness.martingale_ruin(mkt[win], vix[win]),
    }).round(4))


if __name__ == "__main__":
    main()
