"""Reproducible headline run for Study 783 — IPO-Deal-Of-Year.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached marquee-IPO / SPY tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from ipo_deal_of_year import data as dt, strategy as st  # noqa: E402

print("# IPO-Deal-Of-Year — do the banks' celebrated 'IPO of the year' then underperform?")

print(f"calendar: {len(dt.EVENTS)} marquee US IPOs {dt.EVENTS[0][1][:4]}->{dt.EVENTS[-1][1][:4]}, "
      f"hardcoded real tickers + first-trade dates (SELECTION EX POST BY DESIGN)")

if not dt.have_real():
    print("(cache miss — fetching marquee names + SPY once)")
    dt.fetch()

prices = dt.load_real()
spy = prices[dt.BENCHMARK]
n_rows = sum(len(prices[t]) for t, _, _ in dt.EVENTS) + len(spy)
print(f"tapes loaded: {len(prices)} series ({len(dt.EVENTS)} names + SPY), "
      f"{n_rows:,} total name-day rows; SPY {spy.index[0].date()}->{spy.index[-1].date()}")

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} marquee IPOs have full 12-month "
      f"forward coverage")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n:2d}x {reason}")

print("\n# FORWARD ABNORMAL RETURN (name minus SPY) from the first trading close")
for label, col in (("3-month (k=63)", "fwd_3m"), ("6-month (k=126)", "fwd_6m"),
                   ("12-month (k=252)", "fwd_12m")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<18s} n={s['n']:2d}  mean={s['mean']*100:+.2f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Per-name 12-month forward abnormal return (the raw ledger)")
for _, r in inc.sort_values("fwd_12m").iterrows():
    print(f"  {r['ticker']:<5s} {r['anchor_date']}  12m AR {r['fwd_12m']*100:+7.1f}%   {r['label']}")

print("\n# Random-window placebo (20 seeds x 200 draws per name, drawn from each name's own tape)")
for label, col, k in (("3-month", "fwd_3m", 63), ("12-month", "fwd_12m", 252)):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail="left")
    print(f"  {label:<10s} (left-tail): observed {pl['obs']*100:+.2f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.2f}% (sd {pl['placebo_sd']*100:.2f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — 12-month forward AR t-stat")
x = inc["fwd_12m"].values
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — 12-month forward AR net of costs (descriptive: crown is awarded ex post)")
for base, label in (("fwd_3m", "3-month"), ("fwd_12m", "12-month")):
    g = st.one_sample_t(inc[base].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<10s} gross {g['mean']*100:+.2f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.2f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.2f}% (t={n10['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=802 + s)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for bump in (-0.10, -0.20):
    planted = st.synthetic_detect(bump=bump, seed=802)
    print(f"  planted 12m drift {bump*100:+.0f}% (seed 802): mean AR {planted['mean']*100:+.2f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
