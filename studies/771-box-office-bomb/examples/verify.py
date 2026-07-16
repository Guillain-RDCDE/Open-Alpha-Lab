"""Reproducible headline run for Study 771 — Box-Office-Bomb.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached DIS / SPY tapes under
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

from box_office_bomb import data as dt, strategy as st  # noqa: E402

print("# Box-Office-Bomb — should you sell Disney (DIS) after a notorious flop weekend?")

print(f"calendar: {len(dt.EVENTS)} notorious Disney box-office bombs "
      f"{dt.EVENTS[0][1][:4]}->{dt.EVENTS[-1][1][:4]}, hardcoded opening dates "
      f"(anchor = first session on/after the Monday after opening)")

if not dt.have_real():
    print("(cache miss — fetching DIS + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("Box-office-bomb panel (DIS + SPY, adjusted/total-return)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} flops have DIS+SPY coverage "
      f"around the anchor")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n:2d}x {reason}")

print("\n# THE POST-FLOP (the 'sell the news') — abnormal return, K sessions AFTER the "
      "anchor, DIS minus SPY (gross). Folklore says this should be NEGATIVE.")
for label, col in (("2-week post (k=10)", "post_s"), ("1-month post (k=21)", "post_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"up {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# THE PRE-RELEASE DRIFT (context) — abnormal return, K sessions BEFORE the anchor")
for label, col in (("2-week pre (k=10)", "pre_s"), ("1-month pre (k=21)", "pre_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"up {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event)")
for label, col, k, tail in (("2wk post", "post_s", 10, "left"),
                            ("1mo post", "post_l", 21, "left")):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail=tail)
    print(f"  {label:<12s} ({tail}-tail): observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — 2-week post-flop t-stat")
x = inc["post_s"].values
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — the SHORT (sell the studio) net of costs, calendar-known entry")
for base, label in (("post_s", "2-week post"), ("post_l", "1-month post")):
    g = st.one_sample_t(inc[base].values)
    short_col = "short_s_net" if base == "post_s" else "short_l_net"
    n5 = st.one_sample_t(inc[short_col].values)
    n10 = st.one_sample_t(inc10[short_col].values)
    print(f"  {label:<14s} gross DIS-SPY {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"SHORT net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"SHORT net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(drop=0.0, seed=773 + s, k=10)["t"] for s in range(20)])
print(f"  null (drop=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for drop in (0.01, 0.02):
    planted = st.synthetic_detect(drop=drop, seed=773, k=10)
    print(f"  planted post-flop drop=-{drop*100:.1f}% (seed 773): mean AR {planted['mean']*100:+.3f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
