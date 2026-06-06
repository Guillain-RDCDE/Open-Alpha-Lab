"""Is -3% special? Sweep the drop threshold and the look-back N.

The project is named after -3%, but if the (non-)edge is real it must vary
smoothly as we move the knob. A lone threshold that lights up while its neighbours
are flat is data-mined. Run:

    python examples/sweep_thresholds.py

Needs cached data from a prior `verify_ndx.py` run (or network on first call).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falling_knife import data, triggers, sweeps, plots

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 140)


def run_for(label, ohlc):
    print(f"\n{'#'*72}\n#  {label}\n{'#'*72}")
    ret = data.daily_returns(ohlc)

    # 1) Threshold sweep on the classic close-to-close trigger.
    sw = sweeps.threshold_sweep(ohlc, ret, trigger_fn=triggers.close_to_close)
    print("\n[Threshold sweep | T1 close-to-close]  excess vs random day (%):")
    print((sw["excess"] * 100).to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\n  p_greater (smaller = more likely a real edge):")
    print(sw["p_greater"].to_string(float_format=lambda v: f"{v:.3f}"))

    # 2) Window sweep on the drawdown trigger (which scored best earlier).
    ws = sweeps.window_sweep(ohlc, ret, kind="drawdown", horizon=5)
    print("\n[Window sweep | T3 drawdown look-back, +5d horizon]:")
    print(ws.to_string(float_format=lambda v: f"{v:.4f}"))

    plots.plot_threshold_heatmap(
        sw, title=f"{label}: excess vs random — threshold x horizon",
        path=os.path.join(OUT_DIR, f"out_threshold_{label.split()[0].lower()}.png"))
    return sw, ws


def main():
    print("FALLING-KNIFE — threshold & window robustness sweeps")
    for label, (spot, _etf) in data.INDEX_PAIRS.items():
        ohlc = data.fetch(spot, mode="split_only")
        run_for(f"{label} ({spot})", ohlc)

    print("\nReading guide: a genuine effect changes gradually across thresholds. "
          "If only -3% pops while -2%/-4% are flat, that's the round-number trap.")


if __name__ == "__main__":
    main()
