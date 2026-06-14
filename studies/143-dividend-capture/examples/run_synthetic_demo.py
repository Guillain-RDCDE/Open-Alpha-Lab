"""Offline, deterministic demo — Study 143 (Dividend-Capture).

No network. Generates synthetic ex-dividend events and shows the study's spine
in one screen: the capture trade earns nothing on a fully efficient market
(drop ratio = 1.0), but the same machinery correctly finds a gross profit
when we plant partial efficiency. Run:

    python studies/143-dividend-capture/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dividend_capture import data, strategy as st  # noqa: E402


def _fmt(v: float, fmt: str = "+.2f") -> str:
    return format(v, fmt) if v == v else "  nan"


def main() -> None:
    print("Study 143 — Dividend-Capture — synthetic positive control\n")
    print("The efficient-markets null: price drops by exactly the dividend on the ex-date.")
    print("The capture trade earns zero gross unless the drop is systematically < the dividend.\n")

    header = (
        f"{'efficiency':>12} | {'n_events':>8} | "
        f"{'mean drop/div':>14} | {'t vs 1.0':>10} | "
        f"{'gross bps':>10} | {'t gross':>8} | "
        f"{'net bps':>9} | profitable?"
    )
    print(header)
    print("-" * len(header))

    for eff in (0.0, 0.25, 0.50, 0.75, 1.0):
        events, truth = data.synthetic_events(
            n_events=300, efficiency=eff, seed=143
        )
        drop_s = st.drop_ratio_stats(events)
        cap_s = st.capture_trade_stats(events, cost_bps=5.0)
        profitable = "yes" if cap_s["mean_net_bps"] > 0 and cap_s["tstat_net"] > 1.5 else "no"
        print(
            f"{eff:>12.2f} | {drop_s['n']:>8d} | "
            f"{drop_s['mean_ratio']:>14.4f} | {drop_s['tstat_vs_one']:>+10.2f} | "
            f"{cap_s['mean_gross_bps']:>+10.2f} | {cap_s['tstat_gross']:>+8.2f} | "
            f"{cap_s['mean_net_bps']:>+9.2f} | {profitable}"
        )

    print()
    print("At efficiency=0.0 (the EMH null), the price drops by exactly the dividend.")
    print("The capture trade earns ~0 gross and is a net loser once costs are charged.")
    print("The machinery correctly recovers a gross profit only when we plant partial")
    print("market inefficiency (efficiency > 0) — the price drop is less than the div.")
    print()
    print("On the REAL tape (see docs/results.md), the drop ratio is ~1.1 (price drops")
    print("slightly MORE than the div on average) and capture earns -12 bps gross.")
    print("Verdict: NONE / MIRAGE.")


if __name__ == "__main__":
    main()
