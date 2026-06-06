"""Reproduce the world-indices result on REAL data (needs Internet / Yahoo).

Run:  python examples/verify_world_indices.py

For each index ETF it prints the overnight vs intraday split, runs the artefact
detector, and saves a grid figure. Two things to look for, straight from the
critique:
  * the CHINA test (FXI): Knuteson's pattern should look weak/inverted vs the
    West, consistent with the T+1 microstructure explanation (Qiao and Dam 2020);
  * the INDIA counter (INDA): how many days the artefact detector flags — a
    proxy for the data-quality problems behind Figure 8.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: never pop a window from a script

from overnight import backtest, data, decompose, diagnostics, plots  # noqa: E402

MODE = "split_only"  # document this choice in any published figure!


def main() -> None:
    print(f"Fetching {len(data.WORLD_INDICES)} indices (mode={MODE}). First run hits the network.\n")
    decs = {}
    for ticker, label in data.WORLD_INDICES.items():
        try:
            ohlc = data.fetch(ticker, mode=MODE)
        except Exception as exc:  # network / ticker issues shouldn't kill the run
            print(f"  {ticker:5s} {label:28s}  FAILED: {exc}")
            continue
        dec = decompose.decompose(ohlc)
        s = decompose.summary(dec)
        flags = diagnostics.flag_suspicious_returns(dec, threshold=0.40)
        decs[f"{ticker} — {label}"] = dec
        print(
            f"  {ticker:5s} {label:28s}  "
            f"overnight {s.loc['overnight','cum_return']*100:>+10,.0f}%  "
            f"intraday {s.loc['intraday','cum_return']*100:>+9,.0f}%  "
            f"Sharpe(on) {s.loc['overnight','sharpe']:+.2f}  "
            f"flags {len(flags)}"
        )

    if not decs:
        print("\nNo data fetched — check your internet connection / Yahoo availability.")
        return

    fig = plots.plot_grid(decs, linear=False)
    out = "out_world_indices.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nGrid figure saved -> {out}")

    # Cost reality on the US flagship.
    if any(k.startswith("SPY") for k in decs):
        spy = next(v for k, v in decs.items() if k.startswith("SPY"))
        print("\nSPY cost sweep (round-trip bps -> net CAGR / Sharpe / maxDD):")
        print(backtest.cost_sweep(spy).round(3).to_string())


if __name__ == "__main__":
    main()
