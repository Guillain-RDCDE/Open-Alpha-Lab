"""Reproducible headline run for Study 644 — CPI-Day-Drift.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / TLT tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from cpi_day_drift import data, strategy as st  # noqa: E402

print("# CPI-Day-Drift — do stocks/bonds move systematically on CPI-print mornings?")

cpi = data.cpi_calendar()
print(f"calendar: {len(cpi)} actual CPI release days {cpi.min().date()} -> {cpi.max().date()} "
      f"(hardcoded BLS release-date table, identical to sibling study 602's CPI_DATES)")

if not data.have_real():
    print("(cache miss — fetching SPY / TLT once)")
    data.fetch()

spy_px, tlt_px = data.load_real()
print(data_stamp("SPY OHLC", spy_px, asof=data.AS_OF))
print(data_stamp("TLT OHLC", tlt_px, asof=data.AS_OF))

spy_sessions, spy_mapped = data.map_to_sessions(spy_px.index, cpi)
tlt_sessions, tlt_mapped = data.map_to_sessions(tlt_px.index, cpi)
print(f"CPI sessions mapped onto the SPY tape: {len(spy_sessions)} of {len(cpi)} "
      f"({spy_mapped} forward-mapped off a holiday)")
print(f"CPI sessions mapped onto the TLT tape: {len(tlt_sessions)} "
      f"(TLT inception {data.TLT_START}, so fewer events)")

spy = st.day_frame(spy_px, spy_sessions)
tlt = st.day_frame(tlt_px, tlt_sessions)

print("\n# THE HEADLINE — CPI-day return vs all other days")
for name, df in (("SPY", spy), ("TLT", tlt)):
    s = st.cpi_day_stats(df)
    print(f"  [{name}] CPI-day {s['cpi_bps']:+.2f} bps  vs other-day {s['rest_bps']:+.2f} bps "
          f"  gap {s['gap_bps']:+.2f} bps   Welch t = {s['welch_t']:+.2f}   NW(5) t = {s['nw_t']:+.2f}")
    print(f"         hit rate (up): {s['hit_up']}/{s['n_cpi']} = {s['hit_rate']*100:.1f}%  "
          f"(Wilson 95% [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])   n_rest={s['n_rest']:,}")

print("\n# Random-calendar placebo (20 seeds x 1,000 draws)")
pl = st.placebo_pvalue(spy, col="ret", two_sided=True)
print(f"  [SPY return, two-sided] observed {pl['obs']*1e4:+.2f} bps vs placebo mean "
      f"{pl['placebo_mean']*1e4:+.2f} bps (sd {pl['placebo_sd']*1e4:.2f}) over "
      f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.5f}")
pl_rng = st.placebo_pvalue(tlt, col="range_pct", two_sided=False)
print(f"  [TLT range, one-sided \"louder\"] observed {pl_rng['obs']*100:.3f}% vs placebo "
      f"mean {pl_rng['placebo_mean']*100:.3f}% (sd {pl_rng['placebo_sd']*100:.3f}) over "
      f"{pl_rng['n_draws']:,} draws -> p = {pl_rng['p_value']:.5f}")
pl_spy_rng = st.placebo_pvalue(spy, col="range_pct", two_sided=False)
print(f"  [SPY range, one-sided \"louder\"] observed {pl_spy_rng['obs']*100:.3f}% vs "
      f"placebo mean {pl_spy_rng['placebo_mean']*100:.3f}% (sd "
      f"{pl_spy_rng['placebo_sd']*100:.3f}) over {pl_spy_rng['n_draws']:,} draws -> "
      f"p = {pl_spy_rng['p_value']:.5f}")

print("\n# Event window — SPY return (bps) by session offset around the release (Welch t vs far days)")
ev = st.event_study(spy, spy_sessions)
for k, row in ev.iterrows():
    tag = "  <-- CPI day" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_bps']:+.2f} bps  (n={int(row['n'])}, "
          f"Welch t={row['welch_t']:+.2f}){tag}")
ru = st.runup_stats(spy, spy_sessions)
print(f"  pre-release run-up [-3..-1] cumulative: {ru['mean_runup_bps']:+.2f} bps/event "
      f"(one-sample t = {ru['t']:+.2f}, n={ru['n_events']} events)")

print("\n# Realized high-low range on the same days ((H-L)/prev close)")
for name, df in (("SPY", spy), ("TLT", tlt)):
    rg = st.range_stats(df)
    print(f"  [{name}] CPI days {rg['cpi_range_pct']:.3f}%  vs other days "
          f"{rg['rest_range_pct']:.3f}%   Welch t = {rg['welch_t']:+.2f}")

print(f"\n# Regime contrast — pre vs post {data.REGIME_SPLIT} (justified: the Fed's Dec-2021 "
      "hawkish pivot)")
ec_ret = st.era_contrast(spy, data.REGIME_SPLIT, col="ret")
print(f"  SPY return : pre {ec_ret['early']:+.2f} bps (n={ec_ret['n_early']}, "
      f"t={ec_ret['welch_t_early']:+.2f})  |  post {ec_ret['late']:+.2f} bps "
      f"(n={ec_ret['n_late']}, t={ec_ret['welch_t_late']:+.2f})  |  diff t = "
      f"{ec_ret['welch_t_diff']:+.2f}")
ec_rng = st.era_contrast(spy, data.REGIME_SPLIT, col="range_pct")
print(f"  SPY range  : pre {ec_rng['early']:.3f}% (t={ec_rng['welch_t_early']:+.2f})  |  "
      f"post {ec_rng['late']:.3f}% (t={ec_rng['welch_t_late']:+.2f})  |  diff t = "
      f"{ec_rng['welch_t_diff']:+.2f}")

print("\n# THIRD AXIS — is CPI day the SINGLE BIGGEST trading day of its calendar month?")
for name, df_, mask, met in (("SPY |return|", spy, spy_sessions, "ret"),
                             ("SPY range", spy, spy_sessions, "range_pct"),
                             ("TLT |return|", tlt, tlt_sessions, "ret"),
                             ("TLT range", tlt, tlt_sessions, "range_pct")):
    bd = st.biggest_day_of_month(df_, mask, data.REGIME_SPLIT, metric=met)
    print(f"  [{name}] pre: {bd['pre_rate']*100:.1f}% (Wilson [{bd['pre_lo']*100:.1f}%, "
          f"{bd['pre_hi']*100:.1f}%], n={bd['n_pre']})  |  post: {bd['post_rate']*100:.1f}% "
          f"(Wilson [{bd['post_lo']*100:.1f}%, {bd['post_hi']*100:.1f}%], n={bd['n_post']})  |  "
          f"null {bd['null_rate']*100:.1f}%  |  diff t = {bd['welch_t_diff']:+.2f}")

print("\n# TRADABILITY — a naive 'own SPY only on CPI day' timer")
print("  (enter prior close — the calendar is public months ahead, zero look-ahead; exit")
print("   the CPI-day close; one round trip = 2 x one-way cost x NAV; long-only, no borrow)")
for cb in (5.0, 10.0):
    tc = st.timer_capture(spy, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: gross {tc['gross_bps']:+.2f} bps/event -> net "
          f"{tc['net_bps']:+.2f} bps/event  (~{tc['ann_net_pct']:+.2f}%/yr at 12 events)")
tc = st.timer_capture(spy, cost_bps=5.0)
print(f"  CPI-day SPY {tc['gross_bps']:+.2f} bps vs other-day {tc['rest_bps']:+.2f} bps   "
      f"Welch t = {tc['welch_t']:+.2f}  (n={tc['n_cpi']} events)")
print(f"  hit rate {tc['hit_rate']*100:.1f}% | worst single CPI day {tc['worst_day_pct']:+.1f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detectors must NOT fire on a null world and must recover a planted")
print("  return shift AND, independently, a planted volatility multiplier. Null checked")
print("  over 20 seeds (never a single stream).")
null_ret_ts, null_rng_ts = [], []
for s_ in range(20):
    close, dec = data.synthetic_world(mu_shift=0.0, vol_mult=1.0, seed=644 + s_)
    d = st.synthetic_detect(close, dec)
    null_ret_ts.append(d["welch_t"])
    null_rng_ts.append(d["range_welch_t"])
null_ret_ts = np.asarray(null_ret_ts)
null_rng_ts = np.asarray(null_rng_ts)
print(f"  null (mu_shift=0, vol_mult=1), 20 seeds: return mean t = {null_ret_ts.mean():+.2f} "
      f"(sd {null_ret_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ret_ts) >= 2).sum()}/20; "
      f"range mean t = {null_rng_ts.mean():+.2f} (sd {null_rng_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_rng_ts) >= 2).sum()}/20")
close, dec = data.synthetic_world(mu_shift=0.0015, vol_mult=1.0, seed=644)
sy_ret = st.synthetic_detect(close, dec)
print(f"  planted return shift = +15 bps (vol_mult=1, seed 644): CPI-day mean "
      f"{sy_ret['cpi_bps']:+.2f} vs rest {sy_ret['rest_bps']:+.2f} bps   Welch t = "
      f"{sy_ret['welch_t']:+.2f}")
close, dec = data.synthetic_world(mu_shift=0.0, vol_mult=1.6, seed=644)
sy_rng = st.synthetic_detect(close, dec)
print(f"  planted vol multiplier = 1.6x (mu_shift=0, seed 644): range Welch t = "
      f"{sy_rng['range_welch_t']:+.2f}  (return Welch t = {sy_rng['welch_t']:+.2f}, "
      "should stay small)")
