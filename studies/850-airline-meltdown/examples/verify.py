"""Reproducible headline run for Study 850 — Airline Operational Meltdown.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / LUV / DAL / UAL /
AAL / BA tapes under ``_cache/`` (fetching once on a cache miss), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from airline_meltdown import data, strategy as st  # noqa: E402

print("# Airline Operational Meltdown — does a public operational collapse dent the")
print("# implicated carrier's own stock? Single-name market-model event study.")

allev = data.events_table()
ev = data.coverable_events()
print(f"\nmeltdown table: {len(allev)} curated operational meltdowns "
      f"{allev['date'].min().date()} -> {allev['date'].max().date()} "
      f"({len(ev)} with fetchable-ticker coverage; SAVE/Spirit delisted -> dropped)")

if not data.have_real():
    print("(cache miss — fetching SPY + LUV/DAL/UAL/AAL/BA once)")
    data.fetch()

spy, stocks = data.load_prices()
print(data_stamp("SPY close", spy.to_frame("Close"), asof=data.AS_OF))
for t, s in stocks.items():
    print(data_stamp(f"{t} close", s.to_frame("Close"), asof=data.AS_OF))

cars = st.stack_event_cars(ev, spy, stocks)
print(f"\n# Events scored on the tape: {len(cars)}")
print("  per-event market-model CAR (bps): day0 / week[0..4] / month[0..21] / drift[1..21]")
for d, row in cars.iterrows():
    print(f"  {d.date()} {row['ticker']:4s}: "
          f"{row['day0']*1e4:+8.1f} / {row['week']*1e4:+8.1f} / "
          f"{row['month']*1e4:+8.1f} / {row['drift']*1e4:+8.1f}")

print("\n# THE HEADLINE — cross-event mean CAR by horizon (one-sample t across events)")
for h in ("day0", "week", "month", "drift"):
    s = st.car_stats(cars, h)
    wlo, whi = s["wilson"]
    print(f"  {h:6s}: n={s['n']}  mean {s['mean_bps']:+8.1f} bps  t={s['t']:+.2f}  "
          f"(NW t={s['t_nw']:+.2f})  down {s['down']}/{s['n']} "
          f"(Wilson [{wlo*100:.0f}%, {whi*100:.0f}%])")

print("\n# Same-ticker random-date permutation placebo (5,000 draws)")
for h in ("day0", "month", "drift"):
    pb = st.permutation_placebo(ev, spy, stocks, horizon=h, n_draws=5000, seed=850)
    print(f"  {h:6s}: observed {pb['obs_bps']:+.1f} bps vs placebo mean "
          f"{pb['placebo_mean_bps']:+.1f} (sd {pb['placebo_sd_bps']:.1f}) over "
          f"{pb['n_draws']:,} -> left-tail p = {pb['p_left']:.3f}")

print("\n# ROBUSTNESS — the aggregate is driven by the two Boeing MAX groundings")
air = cars[cars["ticker"] != "BA"]
boe = cars[cars["ticker"] == "BA"]
for tag, sub in (("airlines-only (no BA, n=%d)" % len(air), air),
                 ("Boeing-only (n=%d)" % len(boe), boe)):
    sm = st.car_stats(sub, "month")
    sd = st.car_stats(sub, "day0")
    print(f"  {tag:26s} day0 {sd['mean_bps']:+8.1f} bps (t={sd['t']:+.2f})  "
          f"month {sm['mean_bps']:+8.1f} bps (t={sm['t']:+.2f})")
print("  leave-one-out on the full-sample MONTH one-sample t:")
x = cars["month"].to_numpy()
for i, (d, row) in enumerate(cars.iterrows()):
    _, t = st.one_sample_t(np.delete(x, i))
    flag = "  <-- load-bearing" if abs(t) < 2 and abs(st.one_sample_t(x)[1]) >= 2 else ""
    print(f"    drop {d.date()} {row['ticker']:4s}: t={t:+.2f}{flag}")

print("\n# THE TIMER — short the implicated stock at the meltdown close, hold, cover")
print("  (net = 2x one-way 5 bps + 300 bps/yr borrow on the short leg)")
for hold in (5, 10, 21):
    g = st.summarize_short(st.short_the_meltdown(ev, stocks, hold=hold, cost_bps=0.0,
                                                 borrow_bps_yr=0.0), "short_gross")
    n = st.summarize_short(st.short_the_meltdown(ev, stocks, hold=hold, cost_bps=5.0,
                                                 borrow_bps_yr=300.0), "short_net")
    na = st.summarize_short(st.short_the_meltdown(ev[ev["ticker"] != "BA"], stocks,
                                                  hold=hold, cost_bps=5.0,
                                                  borrow_bps_yr=300.0), "short_net")
    print(f"  hold {hold:>2d}d: gross {g['mean_bps']:+8.1f}  net {n['mean_bps']:+8.1f} bps  "
          f"t={n['t']:+.2f}  win {n['win_rate']*100:.0f}% (n={n['n']})  |  "
          f"airlines-only net {na['mean_bps']:+8.1f} (t={na['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null0 = np.array([st.synthetic_detect(*data.synthetic_world(edge=0.0, seed=850 + s),
                                      horizon="day0")["t"] for s in range(20)])
print(f"  null (edge=0), 20 seeds, day0: mean t = {null0.mean():+.2f} "
      f"(sd {null0.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null0) >= 2).sum()}/20")
m, sk, ee = data.synthetic_world(edge=0.03, seed=850)
sp = st.synthetic_detect(m, sk, ee, horizon="day0")
print(f"  planted edge=0.03 (seed 850), day0: mean {sp['mean_bps']:+.1f} bps  t = {sp['t']:+.2f}")
