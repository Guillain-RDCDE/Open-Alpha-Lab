"""Reproducible headline run for Study 669 — RSI-Divergence.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached six-ticker daily tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from rsi_divergence import data, strategy as st  # noqa: E402

print("# RSI-Divergence — does a lower price low + higher RSI(14) low mark a reversal?")

if not data.have_real():
    print("(cache miss — fetching SPY + basket once)")
    data.fetch()

tape = data.load_real()
for t, df in tape.items():
    print(data_stamp(t, df, asof=data.AS_OF))

events = st.basket_events(tape)
print(f"\nconfirmed bullish divergences: {len(events)} across {len(tape)} tickers "
      f"({data.START} -> {data.AS_OF}, swing order=5, RSI(14))")
print("  by ticker:", dict(events["ticker"].value_counts().sort_index()))

print("\n# THE HEADLINE — divergence-signal forward return vs the unconditional baseline")
print("  (enter next session's OPEN after confirmation, exit at CLOSE h sessions later; the")
print("   confirmation itself needs `order`=5 future bars — that is how a swing low is DEFINED,")
print("   not a look-ahead violation; the one execution lag is the next-open entry)")
headline = {}
for h in (5, 10, 20):
    s = st.headline_stats(tape, events, h)
    headline[h] = s
    print(f"  h={h:>2d}d: signal {s['sig_mean_bps']:+7.2f} bps (n={s['n_sig']})  vs  "
          f"unconditional {s['base_mean_bps']:+7.2f} bps (n={s['n_base']:,})   "
          f"gap {s['gap_bps']:+7.2f} bps  Welch t={s['welch_t']:+.2f}  NW t={s['nw_t']:+.2f}  "
          f"hit {s['hit_rate']*100:.1f}% (Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])")

print("\n# Random-signal placebo (20 seeds x 200 draws of the same signal COUNT per ticker, "
      "random dates)")
placebo = {}
for h in (5, 10, 20):
    pl = st.random_signal_placebo(tape, events, h, n_draws_per_seed=200, n_seeds=20)
    placebo[h] = pl
    print(f"  h={h:>2d}d: observed {pl['obs']*1e4:+7.2f} bps vs placebo mean "
          f"{pl['placebo_mean']*1e4:+7.2f} bps (sd {pl['placebo_sd']*1e4:.2f}) over "
          f"{pl['n_draws']:,} draws -> p(random >= observed) = {pl['p_value']:.4f}")

print("\n# Era contrast — pre vs post the zero-commission retail era (split 2019-10-01, "
      "justified: Schwab/Robinhood cut equity commissions to zero, retail chart-pattern "
      "trading went frictionless and crowded)")
ec = st.era_contrast(tape, events, 10, "2019-10-01")
print(f"  2010 -> 2019-10: signal {ec['early_bps']:+.2f} bps (n={ec['n_early']})")
print(f"  2019-10 -> 2026: signal {ec['late_bps']:+.2f} bps (n={ec['n_late']})")
print(f"  Welch t of the difference (late - early): {ec['welch_t_diff']:+.2f}")

print("\n# THIRD AXIS — a timer with costs (h=10d hold, the headline horizon)")
for cb in (5.0, 10.0):
    tw = st.timer_with_costs(tape, events, 10, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: gross {tw['gross_bps']:+.2f} bps/trade -> net "
          f"{tw['net_bps']:+.2f} bps/trade  (n={tw['n']})")
tw5 = st.timer_with_costs(tape, events, 10, cost_bps=5.0)
print(f"  hit rate {tw5['hit_rate']*100:.1f}% (Wilson [{tw5['hit_lo']*100:.1f}%, "
      f"{tw5['hit_hi']*100:.1f}%])  |  worst trade {tw5['worst_bps']:+.1f} bps  |  "
      f"best trade {tw5['best_bps']:+.1f} bps")

# unconditional hit rate for the same horizon — the fair "coin" comparison (this basket has a
# strong bull-market drift, so plain positivity is not a fair bar; unconditional/random is)
fwd10 = st.basket_forward_series(tape, 10)
allv = np.concatenate([s.dropna().values for s in fwd10.values()])
uncond_hit = float((allv > 0).mean())
print(f"  for reference, the UNCONDITIONAL hit rate at h=10 on this basket is "
      f"{uncond_hit*100:.1f}% (n={len(allv):,}) — the bull-market drift itself beats a coin; "
      "divergence must beat THIS bar, not 50%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT fire on a null world (bounce=0) and must recover a")
print("  planted post-signal bounce planted exactly on the pipeline's OWN flagged dates.")
print("  Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    world = data.synthetic_world(bounce=0.0, seed=669 + s_)
    null_ts.append(st.synthetic_detect(world, h=10)["welch_t"])
null_ts = np.asarray(null_ts)
print(f"  null (bounce=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f}  "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
world = data.synthetic_world(bounce=0.02, seed=669)
sy = st.synthetic_detect(world, h=10)
print(f"  planted bounce=+2.0% over 5 bars (seed 669): signal mean "
      f"{sy['sig_mean_bps']:+.1f} bps (n={sy['n_sig']})  Welch t = {sy['welch_t']:+.2f}  "
      f"NW t = {sy['nw_t']:+.2f}")
