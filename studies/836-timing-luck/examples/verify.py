"""Reproducible headline run for Study 836 — Rebalance Timing Luck.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic and OFFLINE — a synthetic-only method demo,
no network, no real-data fetch. The single-tape headline uses seed 836; the robustness
block averages over 25 seeds (the house rule for any synthetic-dependent claim).

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from timing_luck import data, strategy as st  # noqa: E402

PERIOD, LOOKBACK, TOP_FRAC = 21, 126, 0.3

print("# Rebalance Timing Luck — does the SAME strategy on a different rebalance day "
      "print a different Sharpe?")

# --------------------------------------------------------------------------- #
# Data stamp — the null tape (no momentum edge); real free data can't certify "zero edge"
# --------------------------------------------------------------------------- #
ret0, truth0 = data.synthetic_panel(mom_edge=0.0, seed=836)
print(f"\n[data] null panel: {ret0.shape[1]} assets x {ret0.shape[0]} daily rows  "
      f"{ret0.index.min().date()} -> {ret0.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(ret0)}  (mom_edge={truth0.mom_edge:g})")
print("  SYNTHETIC-ONLY: real free data can never certify 'zero momentum edge', so there "
      "is no real-tape stamp — capped at NONE on the Signal axis.")

# --------------------------------------------------------------------------- #
# THE HEADLINE — phantom Sharpe dispersion across every rebalance offset (null tape)
# --------------------------------------------------------------------------- #
tl = st.timing_luck(ret0, PERIOD, LOOKBACK, TOP_FRAC)
print(f"\n# THE HEADLINE — one momentum book, {PERIOD} rebalance offsets, {tl['n_days']} common days")
print("  per-offset annualised Sharpe:")
print("   ", np.array2string(np.round(tl["sharpes"], 3), max_line_width=100))
print(f"  luckiest offset #{tl['best_offset']:>2d}: Sharpe {tl['sharpe_best']:+.3f}")
print(f"  unluckiest offset #{tl['worst_offset']:>2d}: Sharpe {tl['sharpe_worst']:+.3f}")
print(f"  PHANTOM GAP (luckiest - unluckiest): {tl['sharpe_spread']:.3f} Sharpe units "
      f"(sd across offsets {tl['sharpe_sd']:.3f}); the offsets average {tl['sharpe_mean']:+.3f}")

# --------------------------------------------------------------------------- #
# Is the lucky offset skill or luck? Out-of-sample persistence of the ranking
# --------------------------------------------------------------------------- #
pr = st.offset_persistence(ret0, PERIOD, LOOKBACK, TOP_FRAC)
print(f"\n# LUCK, NOT SKILL — does the lucky offset stay lucky?")
print(f"  Spearman rank corr of offset Sharpes (first half vs second half): "
      f"{pr['rank_corr']:+.3f}  (~0 => unforecastable => pure luck)")

# --------------------------------------------------------------------------- #
# THE FIX — tranched / overlapping portfolio collapses the dispersion
# --------------------------------------------------------------------------- #
tr = st.tranched_portfolio(ret0, PERIOD, LOOKBACK, TOP_FRAC)
print(f"\n# THE FIX — tranch/overlap all {PERIOD} offsets into ONE book")
print(f"  tranched Sharpe {tr['sharpe']:+.3f}  mean {tr['mean_bps']:+.3f} bps/day  "
      f"NW(10) t = {tr['t_nw']:+.2f}  one-sample t = {tr['t_1s']:+.2f}  n={tr['n_days']}")
print(f"  dispersion after tranching: 0.000 (a single curve — nothing left to be lucky "
      f"about), vs {tl['sharpe_spread']:.3f} across offsets before")

# --------------------------------------------------------------------------- #
# THE TIMER — can you get paid for the tranched book? (null tape)
# --------------------------------------------------------------------------- #
print(f"\n# THE TIMER — the tranched book, costed (null tape)")
for cb in (1.0, 5.0):
    tm = st.timer_stats(ret0, PERIOD, LOOKBACK, TOP_FRAC, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.3f} -> net {tm['net_bps']:+.3f} "
          f"bps/day (cost {tm['cost_bps_per_day']:.3f}/day, Sharpe net {tm['sharpe_net']:+.3f}, "
          f"t net {tm['t_net']:+.2f})")

# --------------------------------------------------------------------------- #
# SYNTHETIC POSITIVE CONTROL — machinery detects a PLANTED momentum premium
# --------------------------------------------------------------------------- #
print(f"\n# SYNTHETIC POSITIVE CONTROL — 25 seeds each (the house rule)")
sr0 = st.seed_robust(data, mom_edge=0.0, n_seeds=25)
sr1 = st.seed_robust(data, mom_edge=1.0, n_seeds=25)
print(f"  NULL   (mom_edge=0): phantom spread {sr0['mean_sharpe_spread']:.3f}, "
      f"rank corr {sr0['mean_rank_corr']:+.3f}, tranched Sharpe {sr0['mean_tranched_sharpe']:+.3f} "
      f"(NW t {sr0['mean_tranched_t_nw']:+.3f}), |t|>=2 in {sr0['tranched_t_fires']}/25")
print(f"  PLANTED(mom_edge=1): phantom spread {sr1['mean_sharpe_spread']:.3f}, "
      f"rank corr {sr1['mean_rank_corr']:+.3f}, tranched Sharpe {sr1['mean_tranched_sharpe']:+.3f} "
      f"(NW t {sr1['mean_tranched_t_nw']:+.3f}), |t|>=2 in {sr1['tranched_t_fires']}/25")
print(f"  best-offset identity across 25 null seeds: {sr0['best_offsets']}")
print(f"    (scattered across 0..{PERIOD-1} => the winner is a coin-flip => timing luck)")

print("\nVERDICT: Signal NONE (synthetic-only; the offset dispersion is luck, not edge) | "
      "Tradability MIRAGE (the lucky offset does not persist; nothing to harvest) | "
      "Does rebalance timing break inference? CONFIRMED (a ~0.44-Sharpe phantom gap on the "
      "identical strategy, collapsed to a single curve by tranching).")
