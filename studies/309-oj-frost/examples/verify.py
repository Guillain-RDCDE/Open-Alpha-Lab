#!/usr/bin/env python
"""Reproduce the pinned headline run for Study 309 (OJ-Frost).

Reads the cached OJ=F tape (or fetches once with --fetch), drops the partial final month,
and prints the freeze event study, the perfect-foresight ceiling, and the winter
seasonality — the numbers mirrored in docs/results.md. Deterministic; no network unless
--fetch is passed.

    python examples/verify.py            # from cache
    python examples/verify.py --fetch    # populate the cache once
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oj_frost import data, strategy as st  # noqa: E402

ASOF = pd.Timestamp("2026-05-31")  # drop the partial June 2026 month


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="fetch OJ=F once and cache it")
    args = ap.parse_args()

    f = data.fetch_oj("OJ=F", fetch=args.fetch)
    f = f[f.index <= ASOF]
    print(f"OJ=F  {f.index.min().date()} -> {f.index.max().date()}  "
          f"n={len(f)}  fingerprint={data.fingerprint(f)}")

    fz = data.freeze_dates()
    in_tape = fz[(fz >= f.index.min()) & (fz <= f.index.max())]
    print(f"freezes: {len(fz)} total, {len(in_tape)} in tape -> {list(in_tape.date)}")
    print(f"pre-tape (untestable): {list(fz[fz < f.index.min()].date)}\n")

    print("Freeze event study (reactive, lag=1, gross):")
    for w in (5, 10, 21):
        led = st.window_returns(f, in_tape, window=w, lag=1)
        ctrl = st.random_control_windows(f, n_events=len(led), window=w, n_draws=5000)
        s = st.summarize_events(led, "ret_gross", control_means=ctrl)
        print(f"  w={w:2d}  n={s['n_events']}  win={s['win_rate']*100:.0f}%  "
              f"mean={s['mean_bps']:+.0f}bps  t={s['tstat']:+.2f}  "
              f"placebo_pct={s['placebo_pct']:.2f}  excess={s['excess_bps']:+.0f}")

    led0 = st.window_returns(f, in_tape, window=5, lag=0)
    s0 = st.summarize_events(led0, "ret_gross")
    print(f"\nPerfect-foresight ceiling (lag=0, w=5): mean={s0['mean_bps']:+.0f}bps "
          f"t={s0['tstat']:+.2f}")

    ws = st.winter_seasonality(f)
    print(f"\nWinter (DJF) {ws['winter_mean_bps']:+.1f} bps/day vs rest "
          f"{ws['other_mean_bps']:+.1f} bps/day  diff={ws['diff_bps']:+.1f} "
          f"t={ws['tstat']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
