"""Reproducible headline run for Study 401 — Signal-Stacking.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached real SPY tape under ``_cache/``
if present (the real-tape signal stack), and always runs the deterministic synthetic null and
positive control with no network.

The study is a **methods demo** (sibling of 343–350, 399): it builds the viral "stack K weak
signals into one composite z-score weighted by historical Sharpe" machine and shows that
stacking raises the information coefficient only like **sqrt(K)**, and only when the signals carry
genuine, decorrelated edge. On a pure-noise stack the composite times nothing; redundant
signals plateau; the gorgeous equity curve is in-sample selection that evaporates out of sample.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from signal_stacking import data, strategy as st


def have_real() -> bool:
    """Is the cached real SPY tape present (offline-readable)?"""
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))


def _line(tag, ex):
    """One row of the run_experiment bundle, printed compactly."""
    c = ex["composite"]
    ic = ex["ic"]
    sn = ex["snoop"]
    pm = ex["perm"]
    print(f"  {tag:<26} comp_Sh={c['sharpe']:>6.2f} (t={c['hac_t']:>5.2f})  "
          f"best_single={ex['best_single']:>6.2f}  mean_single={ex['mean_single']:>6.2f}")
    print(f"  {'':<26} comp_IC={ic['composite_ic']:>+.4f}  mean_single_IC={ic['mean_single_ic']:>.4f}  "
          f"lift={ic['lift']:>4.2f}  sqrt(K)={ic['sqrt_k_reference']:.2f}")
    print(f"  {'':<26} perm_p={pm['perm_p']:>.3f}   snoop IS_Sh={sn['is_sharpe']:>6.2f} -> "
          f"OOS_Sh={sn['oos_sharpe']:>6.2f}  (gap={sn['is_sharpe']-sn['oos_sharpe']:>6.2f}, "
          f"n_chosen={sn['n_chosen']})")


print("# Signal-Stacking — does combining weak signals manufacture an edge?")
print("# A methods demo: stacking raises IC only ~sqrt(K), and only for REAL, DECORRELATED edges.")

# ---------------------------------------------------------------------------
# The null (pure noise) and the positive controls — deterministic, no network.
# ---------------------------------------------------------------------------
print("\n# (A) THE NULL — 10 pure-noise signals (signal_ic = 0): stacking can only make luck")
s0, r0, t0 = data.synthetic_panel(n_signals=10, signal_ic=0.0, signal_corr=0.0, seed=401)
ex0 = st.run_experiment(s0, r0, n_perm=2000, seed=401)
_line("null (ic=0, corr=0)", ex0)

print("\n# (B) POSITIVE CONTROL — 10 weak signals each ic=0.05, DECORRELATED: the sqrt(K) kernel")
sP, rP, tP = data.synthetic_panel(n_signals=10, signal_ic=0.05, signal_corr=0.0, seed=401)
exP = st.run_experiment(sP, rP, n_perm=2000, seed=401)
_line("real edge, decorrelated", exP)

print("\n# (C) REDUNDANT — same ic=0.05 but signal_corr=0.9: the sqrt(K) boost collapses to a ceiling")
sR, rR, tR = data.synthetic_panel(n_signals=10, signal_ic=0.05, signal_corr=0.9, seed=401)
exR = st.run_experiment(sR, rR, n_perm=2000, seed=401)
_line("real edge, redundant", exR)

print("\n# (D) THE sqrt(K) CURVE — decorrelated real edge: composite IC lift vs the sqrt(K) reference")
print(f"  {'K':>3} {'mean_single_IC':>15} {'composite_IC':>13} {'lift':>6} {'sqrt(K)':>6}")
for k in (1, 2, 5, 10, 20, 40):
    sk, rk, _ = data.synthetic_panel(n_signals=k, signal_ic=0.05, signal_corr=0.0, seed=401)
    ick = st.composite_ic(sk, rk)
    print(f"  {k:>3} {ick['mean_single_ic']:>15.4f} {ick['composite_ic']:>+13.4f} "
          f"{ick['lift']:>6.2f} {ick['sqrt_k_reference']:>6.2f}")

print("\n# (E) THE sqrt(K) CURVE — REDUNDANT (corr=0.9): lift plateaus far below sqrt(K)")
print(f"  {'K':>3} {'mean_single_IC':>15} {'composite_IC':>13} {'lift':>6} {'sqrt(K)':>6}")
for k in (1, 2, 5, 10, 20, 40):
    sk, rk, _ = data.synthetic_panel(n_signals=k, signal_ic=0.05, signal_corr=0.9, seed=401)
    ick = st.composite_ic(sk, rk)
    print(f"  {k:>3} {ick['mean_single_ic']:>15.4f} {ick['composite_ic']:>+13.4f} "
          f"{ick['lift']:>6.2f} {ick['sqrt_k_reference']:>6.2f}")

# ---------------------------------------------------------------------------
# The real SPY tape — the same 12-signal stack on a price series (cache-first).
# ---------------------------------------------------------------------------
print("\n# (F) THE REAL TAPE — 12 textbook weak signals on SPY (cache-first; not a clean null)")
if have_real():
    sig, fwd = data.load_real()
    span = (sig.index.max() - sig.index.min()).days / 365.25
    print(f"  tape: SPY {sig.index.min().date()} -> {sig.index.max().date()}  "
          f"({len(sig)} days, {span:.1f}y, {sig.shape[1]} signals)")
    print(f"  fingerprint: {data.fingerprint(sig, fwd)}")
    exF = st.run_experiment(sig, fwd, n_perm=2000, seed=401)
    _line("SPY stack (equal-weight)", exF)
    print(f"  buy-and-hold Sharpe (excess) : {exF['bh_sharpe']:>6.2f}")
    # The snooped, Sharpe-weighted blend on the SPY tape (the headline equity curve).
    print(f"  snoop chosen signals (IS-best): {exF['snoop']['chosen']}")
else:
    print("  (no _cache/stacking_SPY.parquet — run data.load_real(fetch=True) once to build it)")

print("\n# (G) SNOOPING TAX — selecting signals + Sharpe-weights in-sample, paying out-of-sample")
print("  On the DECORRELATED positive control (a real edge exists) and on the NULL (none does):")
for tag, (s, r) in {"positive control (ic=0.05)": (sP, rP), "null (ic=0)": (s0, r0)}.items():
    sn = st.snoop_split(s, r)
    print(f"  {tag:<28} IS_Sh={sn['is_sharpe']:>6.2f}  OOS_Sh={sn['oos_sharpe']:>6.2f}  "
          f"gap={sn['is_sharpe']-sn['oos_sharpe']:>6.2f}  n_chosen={sn['n_chosen']}")
