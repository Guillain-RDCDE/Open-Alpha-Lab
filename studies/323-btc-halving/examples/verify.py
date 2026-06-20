"""Reproducible headline run for Study 323 (BTC-Halving) on the real BTC-USD tape.

    python examples/verify.py

Reads the cached Yahoo BTC-USD daily parquet under ``_cache/`` (no network),
runs the cycle-extrema, phase-bucket and timing analyses, prints an as-of stamp
and content fingerprint, and emits the numbers mirrored into docs/results.md and
the notebooks' frozen ``R`` dict.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_halving import data, strategy as st  # noqa: E402


def main() -> int:
    bars = data.load_real()
    # Drop any partial final bar / pin an explicit as-of.
    asof = bars.index[-1]
    print(f"BTC-USD daily: {bars.index[0].date()} -> {asof.date()} "
          f"({len(bars)} days)  fingerprint={data.fingerprint(bars)}")
    print(f"as-of: {asof.date()}\n")

    print("== Halvings covered by this tape ==")
    for h in data.HALVINGS:
        covered = (bars.index.min() <= h <= bars.index.max())
        print(f"  {h.date()}  covered={covered}")

    print("\n== Cycle extrema (offset in days after each halving) ==")
    ext = st.cycle_extrema(bars)
    print(ext.to_string(index=False))

    print("\n== Phase-bucket daily returns (quarters of the cycle) ==")
    pb = st.phase_bucket_returns(bars, n_buckets=4)
    print(pb.round(3).to_string(index=False))

    print("\n== Halving-timing rule (long first half of cycle) vs buy-and-hold ==")
    book = st.timing_rule(bars, phase_lo=0.0, phase_hi=0.5, cost_bps=10.0)
    s = st.summarize_timing(book)
    for k, v in s.items():
        print(f"  {k:18s} {v}")

    lo, hi = st.block_bootstrap_ci(book["ret_strat"].to_numpy() -
                                   book["ret_bh"].to_numpy())
    print(f"  excess mean 95% block-bootstrap CI (bps/day): [{lo:+.2f}, {hi:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
