"""Reproducible headline run for Study 609 — VIX Weekend Arithmetic.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; real-tape numbers come from the cached ^VIX and
VIXY closes under ``_cache/`` (cache-first — refetched only if missing), and the synthetic
control always runs with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402
from vix_weekend_arithmetic import data, strategy as st  # noqa: E402

print("# VIX Weekend Arithmetic — is the day-of-week seesaw in ^VIX pure calendar arithmetic?")

if not data.have_real():
    print("(cache miss — fetching ^VIX/VIXY once)")
    data.fetch()

vix = data.load_vix()
vixy = data.load_vixy()
print(data_stamp("^VIX close", vix.to_frame(), asof=data.AS_OF))
print(data_stamp("VIXY close (adj)", vixy.to_frame(), asof=data.AS_OF))

d = st.dlog_pct(vix)
print(f"sample: {len(d):,} daily close-to-close log changes, "
      f"{d.index.min().date()} -> {d.index.max().date()}")

print("\n# Day-of-week table — mean daily change of ln(VIX), in % (one-sample t per weekday)")
for r in st.weekday_table(d):
    print(f"  {r['weekday']}: {r['mean_pct']:+.3f}%/day   t = {r['t']:+.2f}   (n = {r['n']:,})")

print("\n# The headline contrast — Monday vs Friday")
c = st.mon_fri_contrast(d, lags=10)
print(f"  Monday mean {c['mon_mean']:+.3f}%  (n={c['n_mon']:,})  |  "
      f"Friday mean {c['fri_mean']:+.3f}%  (n={c['n_fri']:,})")
print(f"  Mon - Fri spread = {c['spread']:+.3f}%/day   Welch t = {c['welch_t']:+.2f}   "
      f"Newey-West(10) t = {c['hac_t']:+.2f}")
print(f"  HAC t vs mid-week base: Monday {c['hac_mon_t']:+.2f}, Friday {c['hac_fri_t']:+.2f}")
pl = st.placebo_spread(d, n_draws=20_000, seed=609)
print(f"  label-shuffle placebo (20,000 draws): p = {pl['p_value']:.4f}"
      f"{'  (< 1/20,000 — no draw reached the observed spread)' if pl['p_value'] == 0 else ''}")

print("\n# The arithmetic race — variance-day-count model vs the tape")
print("  model: quoted VIX ~ true vol x sqrt(Neff/30), Neff = trading days + f x weekend days")
print("  window day counts (trading, weekend) by quote day: "
      + ", ".join(f"{data.WEEKDAYS[k]}={data.window_day_counts(k)}" for k in range(5)))
imp = st.implied_weekend_fraction(d)
print(f"  full-arithmetic bound (f = 0): "
      + "  ".join(f"{data.WEEKDAYS[k]} {imp['model_full_arithmetic'][k]:+.2f}%" for k in range(5)))
print(f"  observed weekday means      : "
      + "  ".join(f"{data.WEEKDAYS[k]} {imp['obs'][k]:+.2f}%" for k in range(5)))
print(f"  model at fitted f           : "
      + "  ".join(f"{data.WEEKDAYS[k]} {imp['model_at_fit'][k]:+.2f}%" for k in range(5)))
print(f"  implied weekend fraction f = {imp['f']:.3f}  (a weekend day is priced at "
      f"~{imp['f']*100:.0f}% of a trading day's variance)   fit RMSE = {imp['rmse_pct']:.2f}%/day")

print("\n# Robustness — the seesaw by decade (Monday vs Friday Welch t)")
for r in st.by_decade(d):
    print(f"  {r['decade']}: Mon {r['mon_mean']:+.3f}%  Fri {r['fri_mean']:+.3f}%  "
          f"spread {r['spread']:+.3f}%  Welch t = {r['welch_t']:+.2f}  (n = {r['n']:,})")

print("\n# Robustness — calendar gaps, not weekday labels (holidays included)")
g = st.gap_table(d)
print(f"  post-gap days (prev close >= 3 cal days back): {g['post_mean']:+.3f}%/day (n={g['n_post']:,})")
print(f"  pre-gap days (next close >= 3 cal days ahead): {g['pre_mean']:+.3f}%/day (n={g['n_pre']:,})")
print(f"  mid-week days                                : {g['mid_mean']:+.3f}%/day (n={g['n_mid']:,})")
print(f"  post-gap vs pre-gap Welch t = {g['welch_t_post_vs_pre']:+.2f}   "
      f"post-gap vs mid-week Welch t = {g['welch_t_post_vs_mid']:+.2f}")

print("\n# Third axis — does the Monday pop leak into a tradable vehicle? (VIXY, 2011+)")
print("  trade: buy VIXY Friday close, sell Monday close (calendar known in advance =")
print("  the one documented execution lag); one round trip per weekend, one-way costs below")
for cb in (2.0, 5.0, 10.0):
    v = st.vixy_weekend(vixy, vix, cost_bps=cb)
    if cb == 2.0:
        print(f"  VIXY Monday mean {v['mon_mean_pct']:+.3f}%/weekend (t = {v['mon_t']:+.2f}) vs "
              f"other days {v['rest_mean_pct']:+.3f}%/day   Welch t = {v['welch_t']:+.2f}")
        print(f"  matched ^VIX index Monday mean over the same window: {v['vix_mon_mean_pct']:+.3f}%")
        print(f"  {v['n_mon']} weekends over {v['years']:.1f} years "
              f"(~{v['trades_per_year']:.0f} trades/yr)")
    print(f"  cost={cb:>4.1f} bps/leg: gross {v['gross_ann_pct']:+.1f}%/yr -> "
          f"net {v['net_ann_pct']:+.1f}%/yr")

print("\n# Synthetic positive control — deterministic, no network")
print("  quoted = true vol x sqrt(Neff(weekday, f)/30); f = 1.0 is the null (no weekday")
print("  pattern), f < 1 plants the seesaw; the estimator must recover the pattern AND f.")
for f in (1.0, 0.3):
    syn = data.synthetic_tape(f=f, seed=609)
    ds = st.dlog_pct(syn)
    cs = st.mon_fri_contrast(ds, lags=10)
    im = st.implied_weekend_fraction(ds)
    print(f"  planted f={f:.1f}: Mon-Fri spread {cs['spread']:+.3f}%/day  "
          f"Welch t = {cs['welch_t']:+.2f}  NW t = {cs['hac_t']:+.2f}  "
          f"recovered f = {im['f']:.3f}")
