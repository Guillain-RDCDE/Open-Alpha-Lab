"""Reproducible headline run for Study 737 — Sunspot-Cycle.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ^GSPC price-only tape
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

from sunspot_cycle import data, strategy as st  # noqa: E402

print("# Sunspot-Cycle — does the ~11-year solar cycle drive stock returns?")
print("# (Jevons 1878 sunspot theory of the trade cycle, in its surviving")
print("#  '11-year solar clock on the market' form. Almost certainly a curio.)")

cycles = data.solar_cycles()
tps = data.turning_points()
print(f"\nsolar calendar: cycles {int(cycles['cycle'].min())}-{int(cycles['cycle'].max())} "
      f"({len(cycles)} cycles), SILSO/NOAA turning points hardcoded; a LABELLED cosine "
      f"proxy of the smoothed monthly sunspot number (not the raw SILSO file)")

if not data.have_real():
    print("(cache miss — fetching ^GSPC once)")
    data.fetch()

close = data.load_real()
m = data.monthly_close(close)
ret = st.monthly_returns(m)
ar = st.abnormal_returns(ret)
prox = data.sunspot_proxy(m.index)

print(data_stamp("^GSPC price-only daily close", close.to_frame("Close"), asof=data.AS_OF))
print(data_stamp("^GSPC price-only monthly close", m.to_frame("Close"), asof=data.AS_OF))
print("  (PRICE-ONLY index — no dividends; labelled price-only, never 'total return')")

HORIZON = 12

print("\n# H1 — the 11-year wave: phase regression of monthly abnormal returns on (1, cos, sin)")
pr = st.phase_regression(ar, prox)
print(f"  cos t = {pr['t_cos']:+.3f}   sin t = {pr['t_sin']:+.3f}   (HAC / Newey-West, 12 lags)")
print(f"  R2 = {pr['r2']*100:.3f}%   fitted sinusoid amplitude = {pr['amp_ann_bps']:.1f} bps/yr")
print("  -> neither harmonic clears |t|>=2; the cycle explains ~1 part in 280 of variance")

print("\n# H2 — active vs quiet Sun: high/low activity tercile regime split")
rs = st.regime_split(ret, prox)
print(f"  high-activity {rs['hi_mean']*100:+.3f}%/mo   low-activity {rs['lo_mean']*100:+.3f}%/mo "
      f"(n_hi={rs['n_hi']}, n_lo={rs['n_lo']})")
print(f"  spread = {rs['spread_ann_bps']:+.1f} bps/yr   block-boot 95% CI "
      f"[{rs['ci'][0]*100:+.3f}%, {rs['ci'][1]*100:+.3f}%]/mo   two-sided p = {rs['p_boot']:.3f}")
pl = st.placebo_regime_spread(ret, prox, n_draws=5000, seed=737)
p_pl = st.placebo_pvalue(rs["spread"], pl, tail="two")
print(f"  phase-shift placebo: mean {pl.mean()*12*1e4:+.1f} bps/yr (sd {pl.std()*12*1e4:.1f}) "
      f"over 5,000 draws -> p = {p_pl:.3f}")
print("  -> the spread is WRONG-SIGNED (active < quiet) and inside the luck cloud")

print(f"\n# H3 — turning points: forward {HORIZON}-month return after solar maxima vs minima")
tp = st.turning_point_stats(m, tps, horizon=HORIZON)
plmin = st.forward_placebo(m, tp["n_min"], HORIZON, n_draws=20000, seed=737)
plmax = st.forward_placebo(m, tp["n_max"], HORIZON, n_draws=20000, seed=737)
p_min = float((plmin >= tp["min_mean"]).mean())
p_max = float((plmax >= tp["max_mean"]).mean())
uncond = float((m.shift(-HORIZON) / m - 1.0).mean())
k_min = int((tp["min_fwd"] > 0).sum())
k_max = int((tp["max_fwd"] > 0).sum())
w_min = st.wilson_interval(k_min, tp["n_min"])
w_max = st.wilson_interval(k_max, tp["n_max"])
print(f"  after MAX (n={tp['n_max']}): {tp['max_mean']*100:+.2f}%  one-sample t = {tp['max_t']:+.3f}  "
      f"hit {k_max}/{tp['n_max']} (Wilson [{w_max[0]*100:.1f}%, {w_max[1]*100:.1f}%])  "
      f"random-calendar p = {p_max:.3f}")
print(f"  after MIN (n={tp['n_min']}): {tp['min_mean']*100:+.2f}%  one-sample t = {tp['min_t']:+.3f}  "
      f"hit {k_min}/{tp['n_min']} (Wilson [{w_min[0]*100:.1f}%, {w_min[1]*100:.1f}%])  "
      f"random-calendar p = {p_min:.3f}")
print(f"  unconditional next-{HORIZON}m return (any random year): {uncond*100:+.2f}%")
print(f"  Welch t (max - min) = {tp['welch_t']:+.3f}   diff = {tp['diff_mean']*100:+.2f}%")
print("  -> the ONLY |t|>=2 (minima) is wrong-signed, n=9, and only p~0.05 vs a random calendar:")
print("     a fat one-sample t here just re-discovers that the market drifts up over any 9 dates")

print("\n# H4 — the solar-clock timer: long the rising (min->max) half, else cash")
print("  one documented lag: the phase is lagged 6 months (SILSO smoothing lag); costs one-way x NAV")
for c in (0.0, 5.0, 10.0):
    tm = st.solar_timer(m, prox, smooth_lag=6, cost_bps=c)
    print(f"  {int(c):>2d} bps: timer {tm['timer_cagr']*100:.2f}%/yr  buy&hold {tm['bh_cagr']*100:.2f}%/yr  "
          f"excess {tm['excess_cagr']*100:+.2f}%/yr  excess t = {tm['t_diff']:+.2f}  "
          f"Sharpe {tm['timer_sharpe']:.2f} vs {tm['bh_sharpe']:.2f}  exposure {tm['exposure']*100:.0f}%  "
          f"switches {tm['n_switches']}")
print("  -> loses to buy-and-hold at every cost; mostly by sitting out of a rising market")

print("\n# Synthetic positive control — deterministic, no network")
print("  the regime/phase detector must NOT fire on a null world (amp=0) and must recover")
print("  a planted solar cycle. Null checked over 20 seeds (never a single stream).")
null_p = []
for s_ in range(20):
    cc, px = data.synthetic_world(amp=0.0, seed=737 + s_)
    null_p.append(st.synthetic_detect(cc, px)["regime_p"])
null_p = np.asarray(null_p)
print(f"  null (amp=0), 20 seeds: mean regime p = {null_p.mean():.2f}  "
      f"(p<0.05 in {(null_p < 0.05).sum()}/20 seeds)")
cc, px = data.synthetic_world(amp=0.02, seed=737)
sy = st.synthetic_detect(cc, px)
print(f"  planted amp=2% (seed 737): regime spread {sy['regime_spread_ann_bps']:+.0f} bps/yr, "
      f"p = {sy['regime_p']:.3f}, phase R2 = {sy['phase_r2']*100:.1f}%, cos t = {sy['t_cos']:.1f}")
