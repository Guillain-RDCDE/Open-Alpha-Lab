"""Reproduce the real-data headline run (docs/results.md) — chicken-wing / Super-Bowl seasonality.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download daily WING + SPY + ^IRX from Yahoo, then run

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

from chicken_wing_index import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_data(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    d = repro.as_of(d)
    rf = d["tbill"]

    n_jan = int((pd.DatetimeIndex(d.index).month == 1).sum())
    print(f"\nChicken-wing / Super-Bowl seasonality, {d.index.min().date()}..{d.index.max().date()} "
          f"({len(d)} months, {n_jan} completed Januaries, WING daily closes resampled to month-end)\n")

    ms = st.month_stats(d["wing"])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print("WING per-month mean returns (one-sample t vs 0; naive and HAC — HAC unreliable at n~11):")
    for m in range(1, 13):
        row = ms.loc[m]
        print(f"  {month_names[m-1]:4s}: mean {row['mean']*100:+6.2f}%  "
              f"t={row['tstat']:+.2f}  t_HAC={row['tstat_hac']:+.2f}  n={int(row['n'])}")

    w = st.superbowl_window_test(d["wing"])
    print(f"\nSuper-Bowl run-up window (January) mean: {w['window_mean']*100:.2f}%  n={w['n_window']}")
    print(f"Every other month mean: {w['rest_mean']*100:.2f}%  n={w['n_rest']}")
    print(f"Window - rest spread: {w['spread']*100:.2f}%  Welch t={w['tstat']:.2f}")

    ci = st.spread_bootstrap_ci(d["wing"], n_boot=5000, seed=726)
    print(f"Block-bootstrap 95% CI on spread: [{ci['lo']*100:.2f}%, {ci['hi']*100:.2f}%]  "
          f"(point {ci['point']*100:.2f}%, n_boot={ci['n_boot']})")

    feb = st.superbowl_window_test(d["wing"], window=[2])
    print(f"\nFebruary (the game month) mean: {feb['window_mean']*100:.2f}%  "
          f"vs-rest spread {feb['spread']*100:.2f}%  t={feb['tstat']:.2f}  <- sell-the-news check")

    al = st.window_alpha_vs_market(d["wing"], d["spy"])
    print(f"\nJanuary alpha vs SPY: {al['alpha_m']*100:.2f}%/mo  t={al['t_alpha']:.2f}  beta={al['beta']:.2f}  "
          f"(mean Jan excess of SPY {al['mean_excess']*100:.2f}%, n={al['n']})")

    print("\n12-month placebo — long-one-month timer, excess-of-T-bill Sharpe (rank):")
    pl = st.placebo_months(d["wing"], tbill=rf)
    for m, row in pl.iterrows():
        print(f"  {month_names[int(m)-1]:4s}: sharpe {row['sharpe']:+.2f}  t={row['tstat']:+.2f}  "
              f"mean {row['mean']*100:+.2f}%")

    timer = st.superbowl_timer(d["wing"], tbill=rf)
    bh = st.buy_hold(d["wing"])
    net = st.apply_costs(timer, n_trades_per_year=2, cost_bps_one_way=10)
    print("\nCalendar timer (long January, T-bill otherwise) vs buy-and-hold:")
    print("Sharpe = excess of T-bill, both legs")
    for nm, r in [("Super-Bowl timer (gross)", timer), ("timer (net 10bp/leg)", net), ("buy & hold WING", bh)]:
        s = st.summary(r, rf=rf)
        print(f"  {nm:26s}  CAGR {s['cagr']:+.1%}  Sharpe {s['sharpe']:.2f}  maxDD {s['max_drawdown']:.0%}")

    print("\nSub-period Super-Bowl-window t-stat (WING):")
    d.index = pd.DatetimeIndex(d.index)
    for lab, yr in [("2016-2020", (2016, 2020)), ("2021-on", (2021, 2030))]:
        sl = d[(d.index.year >= yr[0]) & (d.index.year <= yr[1])]
        r = st.superbowl_window_test(sl["wing"])
        print(f"  {lab}: Jan {r['window_mean']*100:.2f}%  rest {r['rest_mean']*100:.2f}%  "
              f"t={r['tstat']:.2f}  n_Jan={r['n_window']}")

    wp = data.load_wing_price()
    print(f"\nWholesale wing price proxy (labelled, cited, NOT tradable): "
          f"2021 spike ${wp.loc['2021-01-31']:.2f}/lb  ->  2023 low ${wp.loc['2023-01-31']:.2f}/lb")

    print(f"\n{repro.data_stamp('WING+SPY+^IRX monthly', d)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
