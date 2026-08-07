"""Reproducible headline run for Study 837 — Look-Ahead Standardization.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic, offline, synthetic-only (no network, no cache):

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from lookahead_standardization import data, strategy as st  # noqa: E402

N_SEEDS = 20

print("# Look-Ahead Standardization — does full-sample z-scoring leak the future?")
print(f"[config] N={data.N_NAMES} names, T={data.N_DAYS} days, horizon={data.HORIZON}, "
      f"min_periods={data.MIN_PERIODS}, seeds={N_SEEDS}, base_seed={data.BASE_SEED}")
print(f"[config] fingerprint {data.config_fingerprint(n_seeds=N_SEEDS)}  as-of {data.AS_OF}")

print("\n# WORLD 1 — STATIONARY null (feature AR(1), returns iid): the contrast, no leak expected")
s1 = st.seed_robust(data.null_stationary, n_seeds=N_SEEDS, base_seed=data.BASE_SEED)
print(f"  full-sample : IC {s1['full_ic']:+.4f} (NW t {s1['full_ic_t']:+.2f})  "
      f"fake Sharpe {s1['full_sharpe']:.2f}  | sig in {s1['full_sig_seeds']}/{N_SEEDS} seeds")
print(f"  expanding   : IC {s1['exp_ic']:+.4f} (NW t {s1['exp_ic_t']:+.2f})  "
      f"Sharpe {s1['exp_sharpe']:.2f}  | sig in {s1['exp_sig_seeds']}/{N_SEEDS} seeds")

print("\n# WORLD 2 — NON-STATIONARY null (random-walk feature, forward change): THE TRAP")
s2 = st.seed_robust(data.null_nonstationary, n_seeds=N_SEEDS, base_seed=data.BASE_SEED)
print(f"  full-sample : IC {s2['full_ic']:+.4f} (NW t {s2['full_ic_t']:+.2f})  "
      f"fake Sharpe {s2['full_sharpe']:.2f}  | sig in {s2['full_sig_seeds']}/{N_SEEDS} seeds")
print(f"  expanding   : IC {s2['exp_ic']:+.4f} (NW t {s2['exp_ic_t']:+.2f})  "
      f"Sharpe {s2['exp_sharpe']:.2f}  | sig in {s2['exp_sig_seeds']}/{N_SEEDS} seeds")
print(f"  GAP (full-exp): |IC| gap {s2['abs_ic_gap']:+.4f}   Sharpe gap {s2['sharpe_gap']:+.2f}")

print("\n# WORLD 3 — PLANTED real edge (stationary feature predicts next return): the control")
s3 = st.seed_robust(data.planted_edge, n_seeds=N_SEEDS, base_seed=data.BASE_SEED)
print(f"  full-sample : IC {s3['full_ic']:+.4f} (NW t {s3['full_ic_t']:+.2f})  "
      f"Sharpe {s3['full_sharpe']:.2f}  | sig in {s3['full_sig_seeds']}/{N_SEEDS} seeds")
print(f"  expanding   : IC {s3['exp_ic']:+.4f} (NW t {s3['exp_ic_t']:+.2f})  "
      f"Sharpe {s3['exp_sharpe']:.2f}  | sig in {s3['exp_sig_seeds']}/{N_SEEDS} seeds")
print("  -> the honest (expanding) method RECOVERS the real edge: it is unbiased, not always-zero.")

print("\n# THE COSTED TIMER — even the fake edge is a Mirage after friction (non-stationary null)")
X, Rr = data.null_nonstationary(seed=data.BASE_SEED)
spread = st.long_short_spread(st.full_standardize(X), Rr, frac=0.2)
for cb in (1.0, 5.0):
    tm = st.timer_stats(spread, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.1f} -> net {tm['net_bps']:+.1f} "
          f"bps/day (cost {tm['cost_bps_per_day']:.1f}/day, net Sharpe {tm['sharpe_net']:.2f})")

print("\n# HORIZON SWEEP (non-stationary null) — the leak GROWS with the forward horizon")
hs = st.horizon_sweep(horizons=(1, 5, 10, 20, 40), n_seeds=10)
for h, row in hs.iterrows():
    print(f"  H={h:>3d}: full IC {row['full_ic']:+.4f}  exp IC {row['exp_ic']:+.4f}  "
          f"full fake Sharpe {row['full_sharpe']:.2f}")

print("\n# LENGTH SWEEP (non-stationary null) — the leak DILUTES as the sample grows (finite-sample)")
ls = st.length_sweep(lengths=(250, 500, 1000, 2000), n_seeds=10)
for T, row in ls.iterrows():
    print(f"  T={T:>4d}: full IC {row['full_ic']:+.4f}  exp IC {row['exp_ic']:+.4f}  "
          f"full fake Sharpe {row['full_sharpe']:.2f}")
