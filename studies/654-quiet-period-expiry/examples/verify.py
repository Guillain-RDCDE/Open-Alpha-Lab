"""Reproducible headline run for Study 654 — Quiet-Period-Expiry.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached per-ticker + SPY tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
from quantlab.repro import data_stamp  # noqa: E402

from quiet_period_expiry import data, strategy as st  # noqa: E402

print("# Quiet-Period-Expiry — does the IPO pop when the 25-day analyst quiet period ends?")

basket = data.basket_frame()
print(f"basket: {len(basket)} hardcoded underwritten US IPOs "
      f"{basket['ipo_date'].min().date()} -> {basket['ipo_date'].max().date()} "
      f"(direct listings & SPAC mergers excluded by construction)")

if not data.have_real():
    print("(cache miss — fetching per-ticker + SPY closes once)")
    data.fetch()

panel = data.fetch_event_panel()
n_on_tape = panel["ticker"].nunique()
print(f"IPOs on the tape: {n_on_tape} of {len(basket)} "
      f"(the rest lack {data.MAX_TD}+ trading days of post-listing history yet, or "
      f"delisted before day {data.MAX_TD} — this can only THIN the basket)")
print(data_stamp("event panel (abnormal returns)", panel.set_index("ipo_date"),
                 cols=["t", "abn_ret"], asof=data.AS_OF))
print(f"fingerprint (abn_ret content) = {data.fingerprint(panel)}")

print(f"\n# THE HEADLINE — CAR vs SPY, trading days [{data.WINDOW_LO}..{data.WINDOW_HI}] "
      f"(the quiet-period-expiry window)")
c = st.car_window_stats(panel)
print(f"  mean CAR = {c['mean_car_pct']:+.3f}%  (median {c['median_car_pct']:+.3f}%, "
      f"sd {c['std_car_pct']:.3f}%, n={c['n']})")
print(f"  one-sample t = {c['t']:+.2f}")
print(f"  hit rate (CAR > 0): {c['hit_up']}/{c['n']} = {c['hit_rate']*100:.1f}%  "
      f"(Wilson 95% [{c['hit_lo']*100:.1f}%, {c['hit_hi']*100:.1f}%])")

print(f"\n# Paired placebo — [{data.WINDOW_LO}..{data.WINDOW_HI}] CAR minus same-width "
      f"[{data.PLACEBO_LO}..{data.PLACEBO_HI}] CAR, per IPO")
pp = st.paired_placebo_stats(panel)
print(f"  mean difference = {pp['mean_diff_pct']:+.3f}%  (n={pp['n']}, t = {pp['t']:+.2f})")

print(f"\n# Random-window placebo (20 seeds x 1,000 draws of same-width random windows)")
rp = st.random_window_placebo(panel)
print(f"  observed window mean {rp['obs']*100:+.3f}% vs placebo mean {rp['placebo_mean']*100:+.3f}% "
      f"(sd {rp['placebo_sd']*100:.3f}%) over {rp['n_draws']:,} draws -> p = {rp['p_value']:.4f}")

print(f"\n# Anatomy — mean abnormal return by trading day (one-sample t vs zero)")
ev = st.per_day_stats(panel, lo=15, hi=35)
for t, row in ev.iterrows():
    tag = "  <-- quiet-period-expiry proxy" if t == data.QUIET_TD else ""
    inwin = " *" if data.WINDOW_LO <= t <= data.WINDOW_HI else "  "
    print(f"  day {t:>2d}{inwin}: mean {row['mean_pct']:+.3f}%  (n={int(row['n'])}, "
          f"t={row['t_stat']:+.2f}){tag}")

print(f"\n# THE LITERAL RETAIL TRADE — buy day {data.TIMER_BUY} close, "
      f"sell day {data.TIMER_SELL} close")
for cb in (5.0, 10.0):
    ts = st.timer_strategy(panel, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: gross {ts['gross_pct']:+.3f}% -> net {ts['net_pct']:+.3f}%  "
          f"(t_gross={ts['t_gross']:+.2f}, t_net={ts['t_net']:+.2f}, n={ts['n']})")
ts5 = st.timer_strategy(panel, cost_bps=5.0)
print(f"  hit rate {ts5['hit_rate']*100:.1f}%  |  best {ts5['best_pct']:+.2f}%  |  "
      f"worst {ts5['worst_pct']:+.2f}%")

print(f"\n# Era contrast — before vs since {data.ERA_SPLIT} "
      f"(sample midpoint, fixed ex ante)")
ec = st.era_contrast(panel, basket)
print(f"  before: mean CAR {ec['early_pct']:+.3f}%  (n={ec['n_early']}, t = {ec['t_early']:+.2f})")
print(f"  since : mean CAR {ec['late_pct']:+.3f}%  (n={ec['n_late']}, t = {ec['t_late']:+.2f})")
print(f"  Welch t of the difference (since - before): {ec['t_diff']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT fire on a null world (pop_bps=0) and must recover a")
print("  planted quiet-period pop. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    syn_panel, _ = data.synthetic_panel(pop_bps=0.0, seed=654 + s_)
    null_ts.append(st.synthetic_detect(syn_panel)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (pop_bps=0), 20 seeds: mean t = {null_ts.mean():+.2f}  (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
syn_panel, _ = data.synthetic_panel(pop_bps=400.0, seed=654)
sy = st.synthetic_detect(syn_panel)
print(f"  planted pop=+400 bps (seed 654): mean CAR {sy['mean_car_pct']:+.3f}%  t = {sy['t']:+.2f}")
