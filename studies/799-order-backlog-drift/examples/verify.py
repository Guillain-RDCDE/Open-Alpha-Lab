"""Reproducible headline run for Study 799 — Order-Backlog Drift (RPO).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached RPO panel under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from order_backlog import data, strategy as st  # noqa: E402

print("# Order-Backlog Drift — does YoY growth in RPO (contracted backlog) lead returns?")

if not data.have_real():
    print("(cache miss — fetching RPO panel once)")
    data.fetch_panel()

prices, events = data.load_real()
print(data_stamp("RPO panel prices (adj close)", prices, asof=data.AS_OF))
print(f"names: {events['ticker'].nunique()} | RPO-growth events: {len(events)} | "
      f"filings {events['filed'].min().date()} -> {events['filed'].max().date()}")

# --- PRIMARY: calendar-time long-short -------------------------------------------------
print("\n# PRIMARY — calendar-time tercile long-short (long high-RPO-growth, short low),")
print("#   monthly rebalance, one execution lag (signal known at month t, return of t+1).")
ls = st.calendar_ls(prices, events, signal_col="rpo_yoy", n_buckets=3, min_names=6,
                    staleness_days=200)
s = st.calendar_ls_stats(ls)
print(f"  months={s['n_months']}  avg cross-section={s['avg_n']:.1f}  "
      f"avg one-way turnover={s['avg_turnover']:.2f}")
print(f"  gross LS mean {s['mean_bps']:+.1f} bps/mo  (~{s['ann_pct']:+.1f}%/yr)  "
      f"hit {s['hit']*100:.0f}%  Sharpe {s['sharpe']:.2f}")
print(f"  long {s['long_bps']:+.1f} bps/mo  short {s['short_bps']:+.1f} bps/mo")
print(f"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   <-- DECISIVE")
for cb in (10.0, 20.0):
    net = st.calendar_ls_net(ls, cost_bps=cb, borrow_bps_ann=100.0)
    print(f"  net @ {cb:>4.1f} bps + 100bps borrow: {net['net_mean_bps']:+.1f} bps/mo "
          f"(~{net['net_ann_pct']:+.1f}%/yr, NW t = {net['net_t_nw']:+.2f}, "
          f"Sharpe {net['net_sharpe']:.2f})")

# --- CROSS-CHECK: pooled event drift ---------------------------------------------------
print("\n# CROSS-CHECK — pooled event-drift sort (534-style), top-minus-bottom RPO-growth")
print("#   tercile forward drift, entered 1 session after the filing.")
for h in st.HORIZONS:
    e = st.event_summary(prices, events, horizon=h, n_buckets=3, lag=1,
                         placebo=True, n_draws=10_000)
    print(f"  horizon {h:>3}d: n={e['n_events']}  top {e['top_mean']*100:+.2f}%  "
          f"bot {e['bot_mean']*100:+.2f}%  LS {e['ls_mean']*100:+.2f}%  "
          f"win {e['ls_win']*100:.0f}%  t = {e['t']:+.2f}  placebo p = {e['p_placebo']:.3f}")

# --- 3rd axis: does RPO growth lead future sales? --------------------------------------
print("\n# MECHANISM — does this quarter's RPO growth predict NEXT quarter's revenue growth?")
lead = st.leads_sales(events)
print(f"  n={lead['n']}  OLS slope next_rev_yoy ~ rpo_yoy = {lead['slope']:+.3f} "
      f"(iid t = {lead['t']:+.2f}, R2 = {lead['r2']:.3f}, corr = {lead['corr']:+.2f})")
print(f"  future-sales-growth spread top vs bottom RPO tercile: "
      f"{lead['top_sales']*100:+.1f}% vs {lead['bot_sales']*100:+.1f}% "
      f"= {lead['spread']*100:+.1f} pp")

# --- Synthetic positive control --------------------------------------------------------
print("\n# Synthetic positive control — deterministic, no network")
print("  the calendar long-short NW t must NOT fire on a null world (edge=0) across seeds")
print("  and must recover a planted forward-return edge.")
null_ts = []
for sd in range(20):
    pr, ev = data.synthetic_panel(edge=0.0, seed=799 + sd)
    null_ts.append(st.synthetic_detect(pr, ev)["t_nw"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean NW t = {np.nanmean(null_ts):+.2f} "
      f"(sd {np.nanstd(null_ts, ddof=1):.2f}), |t|>=2 in "
      f"{int((np.abs(null_ts) >= 2).sum())}/20 seeds")
pr, ev = data.synthetic_panel(edge=0.15, seed=799)
sy = st.synthetic_detect(pr, ev)
print(f"  planted edge=0.15 (seed 799): LS {sy['mean_bps']:+.1f} bps/mo, "
      f"NW t = {sy['t_nw']:+.2f}")
