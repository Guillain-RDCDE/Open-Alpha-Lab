"""Reproduce the real-data headline run (docs/results.md) — crude oil seasonality.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download daily CL=F + XLE + ^IRX from Yahoo, then run

The sample is pinned to the desk's as-of (quantlab.repro). Sharpe convention for the timer race:
excess of the rolled 13-week T-bill (^IRX) on both legs, like-for-like.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from crude_seasonality import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_data(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    d = repro.as_of(d)
    rf = d["tbill"]

    print(f"\nCrude seasonality, {d.index.min().date()}..{d.index.max().date()} "
          f"({len(d)} months, CL=F + XLE, daily closes resampled to month-end)\n")

    # Per-month statistics for CL=F
    ms = st.month_stats(d["crude"])
    print("CL=F per-month mean returns (one-sample t vs 0, n~26 each):")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        row = ms.loc[m]
        print(f"  {month_names[m-1]:4s}: mean {row['mean']*100:+5.1f}%  "
              f"t={row['tstat']:+.2f}  n={int(row['n'])}")

    # Spring vs Autumn spread
    sa = st.spring_autumn_tstat(d["crude"])
    print(f"\nSpring (Apr-Jun) mean: {sa['spring_mean']*100:.2f}%  n={sa['n_spring']}")
    print(f"Autumn (Aug-Nov) mean: {sa['autumn_mean']*100:.2f}%  n={sa['n_autumn']}")
    print(f"Spring vs Autumn: t={sa['tstat']:.2f}  "
          "(house bar: |t| >= 2; Bonferroni-adjusted bar for 12 tests: |t| > ~3)")

    # Calendar timer vs buy-and-hold
    timer_cl = st.seasonal_timer(d["crude"], tbill=rf)
    bh_cl = st.buy_hold(d["crude"])
    timer_xle = st.seasonal_timer(d["energy"], tbill=rf)
    bh_xle = st.buy_hold(d["energy"])

    print("\nCalendar timer (long Apr-Jun, short Aug-Nov, T-bill otherwise) vs buy-and-hold:")
    print("Sharpe = excess of T-bill, both legs")
    for nm, r in [
        ("crude timer (CL=F)", timer_cl),
        ("buy & hold CL=F", bh_cl),
        ("energy timer (XLE)", timer_xle),
        ("buy & hold XLE", bh_xle),
    ]:
        s = st.summary(r, rf=rf)
        print(f"  {nm:30s}  CAGR {s['cagr']:+.1%}  Sharpe {s['sharpe']:.2f}  maxDD {s['max_drawdown']:.0%}")

    # Sub-period split
    print("\nSub-period spring vs autumn t-stat (CL=F):")
    d.index = pd.DatetimeIndex(d.index)
    for lab, yr_range in [("2000-2012", (2000, 2012)), ("2013-on", (2013, 2030))]:
        sl = d[(d.index.year >= yr_range[0]) & (d.index.year <= yr_range[1])]
        r = st.spring_autumn_tstat(sl["crude"])
        print(f"  {lab}: spring {r['spring_mean']*100:.2f}%  autumn {r['autumn_mean']*100:.2f}%  "
              f"t={r['tstat']:.2f}  n_sp={r['n_spring']}  n_au={r['n_autumn']}")

    print(f"\n{repro.data_stamp('CL=F + XLE monthly', d)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
