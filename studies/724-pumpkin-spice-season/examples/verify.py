"""Reproduce the real-data headline run (docs/results.md) — pumpkin-spice-season.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download daily SBUX + SPY + ^IRX from Yahoo, then run

The sample is pinned to the desk's as-of (quantlab.repro). Object of study: SBUX **minus SPY** monthly
total-return excess ("beats the market"). Sharpe convention for the rotation race: excess of the
rolled 13-week T-bill (^IRX).
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

from pumpkin_spice_season import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_data(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    d = repro.as_of(d)
    rf = d["tbill"]
    excess = d["excess"]

    print(f"\nPumpkin-spice season, {d.index.min().date()}..{d.index.max().date()} "
          f"({len(d)} months, SBUX-minus-SPY total-return excess)\n")

    ms = st.month_stats(excess)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print("SBUX-minus-SPY per-month excess (one-sample t vs 0; naive and HAC):")
    for m in range(1, 13):
        row = ms.loc[m]
        tag = " (PSL)" if m in (8, 9, 10, 11) else ""
        print(f"  {month_names[m-1]+tag:6s}: mean {row['mean']*100:+6.2f}%  "
              f"t={row['tstat']:+.2f}  t_HAC={row['tstat_hac']:+.2f}  n={int(row['n'])}")

    se = st.season_tstat(excess)
    print(f"\nSeason (Aug-Nov) mean: {se['season_mean']*100:.2f}%/mo  n={se['n_season']}")
    print(f"Off-season mean:       {se['off_mean']*100:.2f}%/mo  n={se['n_off']}")
    print(f"Season vs Off spread:  {se['spread']*100:.2f}%  Welch t={se['tstat']:.2f}")

    ci = st.spread_bootstrap_ci(excess, n_boot=5000, seed=724)
    print(f"Block-bootstrap 95% CI on spread: [{ci['lo']*100:.2f}%, {ci['hi']*100:.2f}%]  "
          f"(point {ci['point']*100:.2f}%, n_boot={ci['n_boot']})")

    print("\nWindow placebo — every 4-month window's excess spread vs rest-of-year (best first):")
    wp = st.window_placebo(excess)
    for rk, row in wp.iterrows():
        flag = "  <-- PUMPKIN SPICE (Aug-Nov)" if row["is_psl"] else ""
        print(f"  #{rk+1}: {row['months']:16s} spread {row['spread']*100:+.2f}%  t={row['tstat']:+.2f}{flag}")
    print(f"  PSL window rank = {int(wp[wp['is_psl']].index[0]) + 1} of {len(wp)}")

    rot = st.seasonal_rotation(d["sbux"], d["spy"])
    rot_net = st.apply_costs(rot, n_trades_per_year=2, cost_bps_one_way=5)
    timer = st.spread_timer(excess, tbill=rf)
    timer_net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=5)
    bh_spy, bh_sbux = st.buy_hold(d["spy"]), st.buy_hold(d["sbux"])
    print("\nRotation / timer race (Sharpe = excess of T-bill):")
    for nm, r, kw in [("rotation (long SBUX Aug-Nov, else SPY)", rot, dict(rf=rf)),
                      ("rotation (net 5bp)", rot_net, dict(rf=rf)),
                      ("market-neutral pair, season only", timer, dict(rf=rf)),
                      ("market-neutral pair, ALL YEAR", st.buy_hold(excess), dict()),
                      ("buy & hold SPY", bh_spy, dict(rf=rf)),
                      ("buy & hold SBUX", bh_sbux, dict(rf=rf))]:
        s = st.summary(r, **kw)
        print(f"  {nm:40s}  CAGR {s['cagr']:+.1%}  Sharpe {s['sharpe']:.2f}  maxDD {s['max_drawdown']:.0%}")

    print("\nSub-period season vs off (SBUX-SPY excess):")
    d.index = pd.DatetimeIndex(d.index)
    for lab, yr in [("1993-2009", (1993, 2009)), ("2010-on", (2010, 2030))]:
        sl = d[(d.index.year >= yr[0]) & (d.index.year <= yr[1])]
        r = st.season_tstat(sl["excess"])
        print(f"  {lab}: season {r['season_mean']*100:+.2f}%  off {r['off_mean']*100:+.2f}%  "
              f"spread {r['spread']*100:+.2f}%  t={r['tstat']:.2f}")

    print(f"\n{repro.data_stamp('SBUX/SPY/^IRX monthly', d)}")

    b = data.fetch_basket(fetch=fetch)
    if not b.empty:
        b = repro.as_of(b)
        rb = st.season_tstat(b["excess"])
        print(f"\nQSR basket (SBUX/MCD/YUM/CMG) excess over SPY, {len(b)} months: "
              f"season {rb['season_mean']*100:+.2f}%  off {rb['off_mean']*100:+.2f}%  "
              f"spread {rb['spread']*100:+.2f}%  t={rb['tstat']:.2f}")
        print(repro.data_stamp("QSR basket monthly", b))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
