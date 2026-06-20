"""Reproduce the real-data headline run (docs/results.md) — coffee seasonality.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download daily KC=F + ^IRX from Yahoo, then run

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

from coffee_seasonality import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_data(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    d = repro.as_of(d)
    rf = d["tbill"]

    print(f"\nCoffee seasonality, {d.index.min().date()}..{d.index.max().date()} "
          f"({len(d)} months, KC=F daily closes resampled to month-end)\n")

    ms = st.month_stats(d["coffee"])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print("KC=F per-month mean returns (one-sample t vs 0; naive and HAC):")
    for m in range(1, 13):
        row = ms.loc[m]
        print(f"  {month_names[m-1]:4s}: mean {row['mean']*100:+6.2f}%  "
              f"t={row['tstat']:+.2f}  t_HAC={row['tstat_hac']:+.2f}  n={int(row['n'])}")

    fh = st.frost_harvest_tstat(d["coffee"])
    print(f"\nFrost (Jun-Aug) mean: {fh['frost_mean']*100:.2f}%  n={fh['n_frost']}")
    print(f"Harvest (May/Sep) mean: {fh['harvest_mean']*100:.2f}%  n={fh['n_harvest']}")
    print(f"Frost vs Harvest spread: {fh['spread']*100:.2f}%  t={fh['tstat']:.2f}")

    ci = st.spread_bootstrap_ci(d["coffee"], n_boot=5000, seed=307)
    print(f"Block-bootstrap 95% CI on spread: [{ci['lo']*100:.2f}%, {ci['hi']*100:.2f}%]  "
          f"(point {ci['point']*100:.2f}%, n_boot={ci['n_boot']})")

    timer = st.seasonal_timer(d["coffee"], tbill=rf)
    bh = st.buy_hold(d["coffee"])
    net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=10)
    print("\nCalendar timer (long Jun-Aug, short May/Sep, T-bill otherwise) vs buy-and-hold:")
    print("Sharpe = excess of T-bill, both legs")
    for nm, r in [("frost/harvest timer (gross)", timer), ("timer (net 10bp/leg)", net), ("buy & hold KC=F", bh)]:
        s = st.summary(r, rf=rf)
        print(f"  {nm:28s}  CAGR {s['cagr']:+.1%}  Sharpe {s['sharpe']:.2f}  maxDD {s['max_drawdown']:.0%}")

    print("\nSub-period frost vs harvest t-stat (KC=F):")
    d.index = pd.DatetimeIndex(d.index)
    for lab, yr in [("2000-2012", (2000, 2012)), ("2013-on", (2013, 2030))]:
        sl = d[(d.index.year >= yr[0]) & (d.index.year <= yr[1])]
        r = st.frost_harvest_tstat(sl["coffee"])
        print(f"  {lab}: frost {r['frost_mean']*100:.2f}%  harvest {r['harvest_mean']*100:.2f}%  "
              f"t={r['tstat']:.2f}  n_fr={r['n_frost']}  n_ha={r['n_harvest']}")

    print(f"\n{repro.data_stamp('KC=F monthly', d)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
