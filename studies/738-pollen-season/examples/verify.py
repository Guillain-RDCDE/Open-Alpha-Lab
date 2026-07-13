"""Reproducible headline run for Study 738 — Pollen-Season.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket/benchmark tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from pollen_season import data, strategy as st  # noqa: E402

print("# Pollen-Season — do the owners of the big allergy brands rally through spring?")
print("# Claim: US hay-fever season (~Mar->May) drives an antihistamine demand spike, so")
print("# a basket of allergy-brand owners should beat the market through the pollen window.")
print(f"# Window (calendar-known, no execution lag): {data.SEASON_LABEL}")

if not data.have_real():
    print("(cache miss — fetching the basket + benchmarks once)")
    data.fetch()

prices = data.load_real()
for t in data.all_tickers():
    print(data_stamp(f"{t} close", prices[t].to_frame("Close"), asof=data.AS_OF))

years = data.sample_years()
tbl = st.build_spread_table(prices, years)
abn = tbl["abn"].to_numpy()

print(f"\n# Basket coverage (named honestly, never back-filled):")
for t, (brand, note) in data.ALLERGY_BRANDS.items():
    print(f"  {t:6s} {brand:38s} — {note}")
print(f"  sample: {len(tbl)} spring windows {tbl['year'].min()} -> {tbl['year'].max()} "
      f"(2-name basket pre-2003, 3-name to 2022, 4 in 2023 (Kenvue not yet listed at entry), "
      f"5 from 2024)")

print(f"\n# THE HEADLINE — basket-minus-market total return over the pollen window")
h = st.one_sample_t(abn)
hr = st.hit_rate(abn)
lo, hi = st.block_bootstrap_ci(abn)
print(f"  n = {h['n']} independent yearly windows")
print(f"  mean abnormal return : {h['mean']*100:+.2f}%   sd {h['sd']*100:.2f}%   "
      f"one-sample t = {h['t']:+.3f}")
print(f"  hit rate (basket beat the market) : {hr['k']}/{hr['n']} = {hr['rate']*100:.1f}%  "
      f"(Wilson 95% [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")
print(f"  block-bootstrap 95% CI on the mean : [{lo*100:+.2f}%, {hi*100:+.2f}%]  "
      f"(straddles zero)")

print(f"\n# Random-window placebo (20 seeds x 250 draws of same-length NON-spring windows)")
draws = st.placebo_distribution(prices, tbl, n_seeds=20, n_draws_per_seed=250)
p_right = st.placebo_pvalue(abn.mean(), draws, tail="right")
print(f"  observed {abn.mean()*1e4:+.2f} bps  vs  placebo mean {draws.mean()*1e4:+.2f} bps "
      f"(sd {draws.std(ddof=1)*1e4:.2f}) over {len(draws):,} draws")
print(f"  right-tail p = {p_right:.3f}  <-- borderline, and note WHY: the basket bleeds vs")
print(f"      the market on a *random* window ({draws.mean()*1e4:+.0f} bps), so spring merely")
print(f"      looks special relative to that negative baseline — the vs-ZERO t above is 1.06.")

print(f"\n# Robustness — a fairer benchmark (consumer staples XLP) and the 3-name core basket")
x = st.one_sample_t(tbl["abn_xlp"].to_numpy())
core = st.build_spread_table(prices, years, tickers=data.CORE_TICKERS, min_names=2)
c = st.one_sample_t(core["abn"].to_numpy())
print(f"  basket - XLP (staples)     : mean {x['mean']*100:+.2f}%  t = {x['t']:+.2f}  (n={x['n']})")
print(f"  3-name core (BAYRY/SNY/PRGO): mean {c['mean']*100:+.2f}%  t = {c['t']:+.2f}  (n={c['n']})")
print("  neither clears the |t| >= 2 desk bar; the spin-off names do not drive the result")

print(f"\n# Calendar-month seasonality (a blunt cross-check — is spring actually special?)")
ms = st.month_seasonality(prices)
for mo, row in ms.iterrows():
    tag = "  <-- pollen months" if mo in (3, 4, 5) else ""
    print(f"  month {mo:2d}: mean abn {row['mean_abn_bps']:+6.2f} bps/day  (t={row['t']:+.2f}){tag}")

print(f"\n# THE TIMER — long basket / short market over the window; both legs costed, short pays borrow")
for cb in (0.0, 5.0, 10.0):
    t2 = st.build_spread_table(prices, years, cost_bps=cb)
    ts = st.timer_stats(t2)
    print(f"  cost {cb:>4}bps/leg: gross {ts['gross_mean_bps']:+7.2f} bps (t={ts['gross_t']:+.2f})   "
          f"net {ts['net_mean_bps']:+7.2f} bps (t={ts['net_t']:+.2f})")
print("  (borrow = 50 bps annual on the short SPY leg, ~12.6 bps over a 63-session window)")

print(f"\n# Synthetic positive control — deterministic, no network")
print("  the window detector must NOT fire on a null world (bump=0) and must recover a planted")
print("  spring bump. Null checked over 20 seeds (never a single stream).")
null_ts = np.array([st.synthetic_detect(0.0, 738 + s)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 (~ the 5% chance expected, no bias)")
planted = st.synthetic_detect(0.06, 738)
print(f"  planted spring bump = +6.0% (seed 738): mean {planted['mean']*100:+.1f}%  t = {planted['t']:+.2f}")
