"""Reproducible headline run for Study 698 — ABCD-Harmonic.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily basket tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import fingerprint as fp_content  # noqa: E402

from abcd_harmonic import data, strategy as st  # noqa: E402

PCT = 0.03            # zigzag threshold defining a confirmed swing pivot
COST_BPS = 5.0         # headline one-way cost per leg

print("# ABCD-Harmonic — does the AB=CD harmonic predict the reversal at point D?")
print(f"basket: {', '.join(data.BASKET)}  (identical basket to sibling 77-golden-mean)")

if not data.have_real():
    print("(cache miss — fetching daily OHLC once)")
    data.fetch()

bars = {t: data.load_real(t) for t in data.BASKET}
for t in data.BASKET:
    b = bars[t]
    print(f"  {t:5s} {b.index[0].date()} -> {b.index[-1].date()}  n={len(b):,}  "
          f"fp={fp_content(b, cols=['close'])}")


def pooled(placebo: bool, cost: float = 0.0, seed: int = 698):
    frames, pivn, candn = [], 0, 0
    for t in data.BASKET:
        piv, cand, ledger = st.detect_and_scan(bars[t], pct=PCT, placebo=placebo, seed=seed,
                                               cost_bps=cost)
        pivn += len(piv)
        candn += len(cand)
        if not ledger.empty:
            frames.append(ledger)
    ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ledger, pivn, candn


print("\n# THE HEADLINE — Fibonacci ABCD (BC retrace 0.618 +-0.10, CD=AB +-0.15) "
      "vs a placebo arm (same pivots, random off-Fibonacci retrace/extension targets)")
fib_g, n_piv, n_cand_fib = pooled(placebo=False, cost=0.0)
plac_g, _, n_cand_plac = pooled(placebo=True, cost=0.0)
print(f"  pooled confirmed pivots: {n_piv:,}  |  Fibonacci ABCD candidates: {n_cand_fib:,}  "
      f"|  placebo candidates: {n_cand_plac:,}")

for h in (1, 5, 10):
    sf = st.summarize(fib_g, f"ret_gross_{h}")
    sp = st.summarize(plac_g, f"ret_gross_{h}")
    wt = st.welch_t(fib_g[f"ret_gross_{h}"], plac_g[f"ret_gross_{h}"])
    tag = "  <-- headline horizon" if h == 5 else ""
    print(f"  {h:2d}-day: FIB n={sf['n']:4d} hit={sf['hit_rate']*100:5.1f}% "
          f"mean={sf['mean_bps']:+7.2f}bps HAC t={sf['t']:+.2f}   |   "
          f"PLAC n={sp['n']:4d} hit={sp['hit_rate']*100:5.1f}% mean={sp['mean_bps']:+7.2f}bps "
          f"HAC t={sp['t']:+.2f}   |   Fib-Plac Welch t={wt:+.2f}{tag}")

sf5 = st.summarize(fib_g, "ret_gross_5")
sp5 = st.summarize(plac_g, "ret_gross_5")
print(f"\n  hit rate (5-day): FIB {sf5['hit_rate']*100:.1f}% (Wilson 95% "
      f"[{sf5['hit_lo']*100:.1f}%, {sf5['hit_hi']*100:.1f}%])  |  PLAC {sp5['hit_rate']*100:.1f}% "
      f"(Wilson [{sp5['hit_lo']*100:.1f}%, {sp5['hit_hi']*100:.1f}%])")

print("\n# Per-instrument, 5-day gross")
for t in data.BASKET:
    piv, cand, l_fib = st.detect_and_scan(bars[t], pct=PCT, cost_bps=0.0)
    _, _, l_plac = st.detect_and_scan(bars[t], pct=PCT, placebo=True, seed=698, cost_bps=0.0)
    sf = st.summarize(l_fib, "ret_gross_5")
    sp = st.summarize(l_plac, "ret_gross_5")
    print(f"  {t:5s} FIB n={sf['n']:3d} mean={sf['mean_bps']:+7.2f}bps t={sf['t']:+.2f}   |   "
          f"PLAC n={sp['n']:3d} mean={sp['mean_bps']:+7.2f}bps t={sp['t']:+.2f}")

print("\n# THE FADE-AT-D TIMER, net of costs (5-day hold, one round trip = "
      "2 x one-way cost x NAV)")
for c in (0.0, 5.0, 10.0):
    l, _, _ = pooled(placebo=False, cost=c)
    s = st.summarize(l, "ret_net_5")
    print(f"  cost={c:4.1f}bps  FIB net mean={s['mean_bps']:+7.2f}bps/event  HAC t={s['t']:+.2f}")
l_net, _, _ = pooled(placebo=False, cost=COST_BPS)
p_net, _, _ = pooled(placebo=True, cost=COST_BPS)
wtn = st.welch_t(l_net["ret_net_5"], p_net["ret_net_5"])
print(f"  Fib-Plac Welch t at {COST_BPS:.0f}bps net (5-day): {wtn:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  pooled across a synthetic 6-name basket per seed (mirrors the real pooled sample);")
print("  the detector must NOT fire on the null (mean_rev=0) across 20 seeds, and must")
print("  recover a planted mean-reversion tendency.")


def pooled_synth(mean_rev: float, seed: int, n_names: int = 6) -> pd.DataFrame:
    frames = []
    for i in range(n_names):
        sbars = data.synthetic_world(mean_rev=mean_rev, seed=seed * 1000 + i, n_days=6300)
        _, _, ledger = st.detect_and_scan(sbars, pct=PCT, cost_bps=0.0)
        if not ledger.empty:
            frames.append(ledger)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


null_ts = []
for s_ in range(20):
    ledger = pooled_synth(0.0, 698 + s_)
    null_ts.append(st.summarize(ledger, "ret_gross_5")["t"])
null_ts = np.asarray(null_ts, dtype=float)
print(f"  null (mean_rev=0), 20 seeds: mean HAC t = {np.nanmean(null_ts):+.2f}  "
      f"(sd {np.nanstd(null_ts, ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")

planted = pooled_synth(0.08, 698)
sp_planted = st.summarize(planted, "ret_gross_5")
print(f"  planted mean_rev=0.08 (seed 698): n={sp_planted['n']}  mean={sp_planted['mean_bps']:+.2f}bps  "
      f"HAC t={sp_planted['t']:+.2f}")
