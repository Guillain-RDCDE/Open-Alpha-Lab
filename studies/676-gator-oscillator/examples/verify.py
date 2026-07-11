"""Reproducible headline run for Study 676 — Gator Oscillator.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp, fingerprint  # noqa: E402

from gator_oscillator import data, strategy as st  # noqa: E402

NDRAWS = 5_000
COST_BPS = 5.0
HOLD_DAYS = st.HOLD_DAYS

print("# Gator Oscillator — does the alligator's 'awakening' (bars flipping red->green) catch a trend?")

if not data.have_real():
    print("(cache miss — fetching the 30-name basket once)")
    data.fetch()

panel = data.load_panel()
wide_close = pd.DataFrame({t: b["close"] for t, b in panel.items()}).sort_index()
print(f"basket: {len(panel)} names (SPY + 29 liquid US large-caps, a SURVIVORS basket — "
      f"named on the Signal axis), {wide_close.index.min().date()} -> "
      f"{wide_close.index.max().date()}")
print(data_stamp("basket closes (wide)", wide_close, asof=data.AS_OF))
print(data_stamp("SPY OHLC", panel["SPY"], cols=["open", "high", "low", "close"], asof=data.AS_OF))

print(f"\n# THE HEADLINE — wake events (both Gator bars flip red->green after "
      f">= {st.MIN_SLEEP} both-red 'sleeping' bars), signed by the concurrent fan "
      "direction, forward return vs the unconditional base rate")
res = st.run_experiment(panel, horizons=st.HORIZONS, n_draws=NDRAWS, cost_bps=COST_BPS)
print(f"  {'H':>3} {'n_wake':>7} {'n_dir':>6} {'bull':>5} {'bear':>5} {'mean%':>7} "
      f"{'base%':>7} {'win%':>5} {'t_hac':>6} {'t_welch':>7} {'p_plac':>7} {'net%':>7}")
for _, r in res.iterrows():
    print(f"  {int(r['horizon']):>3} {int(r['n_wake']):>7} {int(r['n_dir']):>6} "
          f"{int(r['n_bull']):>5} {int(r['n_bear']):>5} {r['mean_signed']*100:>7.3f} "
          f"{r['base_mean']*100:>7.3f} {r['win']*100:>5.1f} {r['t_hac']:>6.2f} "
          f"{r['t_welch']:>7.2f} {r['p_placebo']:>7.4f} {r['net']*100:>7.3f}")

print("\n# Trend-capture (magnitude) — |forward return| after a wake vs unconditional |return|")
print(f"  {'H':>3} {'mean|%|':>8} {'base|%|':>8} {'t_welch':>7} {'p_plac':>7}")
for _, r in res.iterrows():
    print(f"  {int(r['horizon']):>3} {r['mean_abs']*100:>8.3f} {r['base_abs']*100:>8.3f} "
          f"{r['t_welch_abs']:>7.2f} {r['p_placebo_abs']:>7.4f}")

print(f"\n# THIRD AXIS / TRADABILITY — a real timer on SPY: enter the fan direction the day "
      f"after a wake, hold {HOLD_DAYS} sessions, one execution lag, {COST_BPS:.0f} bps "
      "one-way costs, flat cash-leg proxy 4%/yr when out of the market")
out = st.run_timer_experiment(panel["SPY"], placebo=True, n_draws=2000, cost_bps=COST_BPS)


def fmt(s: dict) -> str:
    return (f"CAGR={s['cagr']:+.2%}  Sharpe={s['sharpe']:+.3f}  Vol={s['vol_ann']:.2%}  "
            f"MaxDD={s['max_drawdown']:+.2%}  t(mean)={s['tstat']:+.2f}")


print(f"  n_wakes (SPY, {panel['SPY'].index.min().date()} -> "
      f"{panel['SPY'].index.max().date()}): {out['n_wakes']}  |  "
      f"in-position fraction: {out['in_pos_frac']:.2%}")
print(f"  Buy-and-hold:          {fmt(out['bh'])}")
print(f"  Gator-wake timer:      {fmt(out['wake'])}")
print(f"  421-style 'in fan':    {fmt(out['fan'])}")
print(f"  Sharpe-diff t (wake vs BH):   {out['t_wake_vs_bh']:+.2f}")
print(f"  Sharpe-diff t (wake vs fan):  {out['t_wake_vs_fan']:+.2f}")
print(f"  Block-permutation placebo p (wake advantage over BH): {out['placebo_p']:.4f}  "
      f"(observed Sharpe-diff {out['placebo_obs']:+.3f})")

out10 = st.run_timer_experiment(panel["SPY"], placebo=False, cost_bps=10.0)
print(f"  cost sweep 10 bps:     {fmt(out10['wake'])}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the wake-event Welch/HAC detector must NOT fire on a null world (edge=0) and "
      "must recover a planted multi-week trend. Null checked over 10 independent panel "
      "seeds (never a single stream); each panel = 20 synthetic series (the synthetic "
      "analogue of the real 30-name basket).")
null_ts = []
for s_ in range(10):
    pan = data.synthetic_multi_panel(edge=0.0, seed=676 + s_)
    null_ts.append(st.summarize(pan, horizon=HOLD_DAYS, placebo=False)["t_hac"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 10 seeds: mean t_hac = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/10 seeds")
pan = data.synthetic_multi_panel(edge=8.0, seed=676)
sy = st.summarize(pan, horizon=HOLD_DAYS, placebo=False)
print(f"  planted trend-persistence edge=8.0 (seed 676): n_wake={sy['n_wake']} "
      f"n_dir={sy['n_dir']}  mean {sy['mean_signed']*100:+.2f}% vs base "
      f"{sy['base_mean']*100:+.2f}%  win {sy['win']*100:.1f}%  "
      f"t_hac={sy['t_hac']:+.2f}  t_welch={sy['t_welch']:+.2f}")
