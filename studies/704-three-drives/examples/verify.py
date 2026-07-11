"""Reproducible headline run for Study 704 — Three Drives.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily basket tapes under
``_cache/`` if present (fetching once on a miss), and always runs the synthetic positive
control with no network.

    python examples/verify.py

Headline is the pooled basket fade test; SPY is the named single tape.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from three_drives import data, strategy as st  # noqa: E402

PCT = 0.04
COST_BPS = 5.0
N_DRAWS = 1000
SEED = 704

print("# Three Drives — do three Fibonacci-proportioned drives exhaust the trend and reverse?")
print(f"# ZigZag reversal threshold: {PCT*100:.0f}%  |  Fibonacci grid: corr "
      f"[{st.THREE_DRIVES['corr_lo']:.3f}, {st.THREE_DRIVES['corr_hi']:.3f}], "
      f"ext [{st.THREE_DRIVES['ext_lo']:.2f}, {st.THREE_DRIVES['ext_hi']:.2f}]  |  "
      f"as-of {data.AS_OF}")

have = [tk for tk in data.TICKERS if data.have_real(tk)]
print(f"# basket cached: {have}")
if not have:
    print("(cache miss - fetching the basket once)")
    data.fetch_all()
    have = list(data.TICKERS)

basket = data.load_basket(have)
for tk in have:
    print(data_stamp(f"{tk} OHLC", basket[tk], cols=["close"], asof=data.AS_OF))

# --------------------------------------------------------------------------- #
# THE HEADLINE — pooled fade test vs the coin-flip (random time+direction) placebo
# --------------------------------------------------------------------------- #
print("\n# THE HEADLINE — pooled three-drives fade test (basket)")
HEADLINE_H = 20
pooled = st.pooled_fade_test(basket, pct=PCT, horizon=HEADLINE_H, cost_bps=COST_BPS)
for tk, c in pooled["counts"].items():
    print(f"  {tk:>7}: pivots={c['pivots']:>4}  three-drives candidates={c['candidates']:>3}")
n_cand = len(pooled["entries"])
print(f"  pooled candidates: {n_cand}")

per_ticker_n = {tk: c["candidates"] for tk, c in pooled["counts"].items()}

print(f"\n  fade trade @ H={HEADLINE_H}d (enter next close after point 5, opposite the drives):")
sg = st.summarize(pooled["gross"])
sn = st.summarize(pooled["net"])
print(f"    gross: n={sg['n']} mean={sg['mean_bps']:+.1f}bps win={sg['win']*100:.1f}% "
      f"(Wilson [{sg['wilson_lo']*100:.1f}%, {sg['wilson_hi']*100:.1f}%]) "
      f"t={sg['t']:+.2f} hac_t={sg['hac_t']:+.2f}")
print(f"    net (cost={COST_BPS:.0f}bps): mean={sn['mean_bps']:+.1f}bps t={sn['t']:+.2f} "
      f"hac_t={sn['hac_t']:+.2f}")

placebo = st.coin_placebo_pvalue(basket, per_ticker_n, HEADLINE_H, sg["mean_bps"] / 1e4,
                                 n_draws=N_DRAWS, seed=SEED)
print(f"\n  coin-flip (random time + random direction) placebo, {N_DRAWS} draws:")
print(f"    observed mean {placebo['obs']*1e4:+.1f} bps vs placebo mean "
      f"{placebo['placebo_mean']*1e4:+.1f} bps (sd {placebo['placebo_sd']*1e4:.1f} bps)")
print(f"    p = {placebo['p_value']:.3f}")

# --------------------------------------------------------------------------- #
# ALL HORIZONS — the fade timer
# --------------------------------------------------------------------------- #
print(f"\n# THE FADE TIMER — every horizon, gross and net of {COST_BPS:.0f}bps costs")
print(f"  {'H':>3} {'n':>5} {'gross_bps':>9} {'net_bps':>8} {'win%':>5} {'t':>6} {'hac_t':>6}")
by_h = {}
for h in st.HORIZONS:
    p_h = st.pooled_fade_test(basket, pct=PCT, horizon=h, cost_bps=COST_BPS)
    sgh = st.summarize(p_h["gross"])
    snh = st.summarize(p_h["net"])
    by_h[h] = {"gross": sgh, "net": snh}
    print(f"  {h:>3} {sgh['n']:>5} {sgh['mean_bps']:>9.1f} {snh['mean_bps']:>8.1f} "
          f"{sgh['win']*100:>4.1f} {sgh['t']:>6.2f} {sgh['hac_t']:>6.2f}")

# --------------------------------------------------------------------------- #
# Fibonacci-grid placebo — are the ratios load-bearing?
# --------------------------------------------------------------------------- #
print(f"\n# FIBONACCI-GRID PLACEBO (pooled basket, H={HEADLINE_H}, {500} random grids)")
grid_pl = st.ratio_grid_placebo(basket, pct=PCT, horizon=HEADLINE_H, n_draws=500, seed=SEED)
print(f"  observed {grid_pl['obs']*1e4:+.1f} bps   placebo p={grid_pl['p_value']:.3f}  "
      f"(valid draws={grid_pl['n_draws']})  (>0.05 => Fibonacci ratios not load-bearing)")

# --------------------------------------------------------------------------- #
# Time-symmetry myth-check — does "symmetric" carry predictive content?
# --------------------------------------------------------------------------- #
print(f"\n# TIME-SYMMETRY MYTH-CHECK (H={HEADLINE_H}, more- vs less-symmetric half)")
sym = st.symmetry_split_test(pooled["entries"], pooled["gross"])
print(f"  n={sym['n']}  median CV={sym['median_cv']:.3f}")
print(f"  more-symmetric half: {sym['mean_sym_bps']:+.1f} bps   "
      f"less-symmetric half: {sym['mean_asym_bps']:+.1f} bps   Welch t = {sym['welch_t']:+.2f}")

# --------------------------------------------------------------------------- #
# ROBUSTNESS — ZigZag threshold sweep
# --------------------------------------------------------------------------- #
print(f"\n# ROBUSTNESS - ZigZag reversal threshold sweep (H={HEADLINE_H})")
for pct in (0.03, 0.04, 0.05, 0.08):
    p2 = st.pooled_fade_test(basket, pct=pct, horizon=HEADLINE_H, cost_bps=0.0)
    s2 = st.summarize(p2["gross"])
    print(f"  pct={pct:.2f}: candidates={s2['n']:>4}  mean={s2['mean_bps']:>7.1f}bps  "
          f"win={s2['win']*100:>5.1f}%  t={s2['t']:>6.2f}")

# --------------------------------------------------------------------------- #
# SPY-only — the named single tape
# --------------------------------------------------------------------------- #
print(f"\n# SPY-only (named single tape, H={HEADLINE_H})")
spy_pool = st.pooled_fade_test({"SPY": basket["SPY"]}, pct=PCT, horizon=HEADLINE_H, cost_bps=0.0)
s_spy = st.summarize(spy_pool["gross"])
print(f"  candidates={s_spy['n']}  mean={s_spy['mean_bps']:+.1f}bps  win={s_spy['win']*100:.1f}% "
      f"(Wilson [{s_spy['wilson_lo']*100:.1f}%, {s_spy['wilson_hi']*100:.1f}%])  t={s_spy['t']:+.2f}")

# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
print("\n# Synthetic positive control - deterministic, no network (H=20 fade)")
print("  the detector must NOT manufacture significance on a pure random-walk null (edge=0,")
print("  exact Fibonacci geometry with no follow-through) and must light up on a planted")
print("  post-point-5 reversal (edge=0.30). Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    bars, _ = data.synthetic_panel(edge=0.0, seed=SEED + s_, n_days=9000)
    piv = st.zigzag(bars["close"].to_numpy(float), pct=PCT)
    ent = st.three_drives_candidates(piv)
    r = st.forward_returns(bars["close"], ent, 20)
    null_ts.append(st.summarize(r)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")

print("  planted edge=0.30 (seed 704), by horizon:")
bars, truth = data.synthetic_panel(edge=0.30, seed=SEED, n_days=9000)
piv = st.zigzag(bars["close"].to_numpy(float), pct=PCT)
ent = st.three_drives_candidates(piv)
print(f"  planted structures: {truth['n_planted']}  detected candidates: {len(ent)}")
for h in st.HORIZONS:
    r = st.forward_returns(bars["close"], ent, h)
    s = st.summarize(r)
    print(f"    H={h:>2}: n={s['n']:>3}  mean={s['mean_bps']:>7.0f} bps  "
          f"t={s['t']:>6.2f}  hac_t={s['hac_t']:>6.2f}")
