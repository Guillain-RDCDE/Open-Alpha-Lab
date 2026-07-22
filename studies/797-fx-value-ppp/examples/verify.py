"""Reproducible headline run for Study 797 — FX Value (PPP real-rate mean reversion).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached FX + CPI tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from fx_value import data, strategy as st  # noqa: E402

print("# FX Value (PPP) — do undervalued real exchange rates predict appreciation?")

if not data.have_real():
    print("(cache miss — fetching FX + CPI once)")
    data.fetch()

fx = data.load_fx()
cpi = data.load_cpi()
logq = data.real_rate_panel(fx, cpi)
print(f"[data] FX  {fx.shape} {fx.index.min().date()}..{fx.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint={fingerprint(fx)}")
print(f"[data] CPI {cpi.shape} {cpi.index.min().date()}..{cpi.index.max().date()}  "
      f"fingerprint={fingerprint(cpi)}")
print(f"real-rate panel: {logq.shape} log q = log S + log CPI_i - log CPI_US "
      f"(CPI lagged {data.CPI_PUB_LAG} mo)")

sig = st.value_signal(logq, window=data.TRAIL_WINDOW, min_periods=data.MIN_TRAIL)
rets = st.spot_returns(fx)
port = st.portfolio_returns(sig, rets)

print("\n# THE HEADLINE — value long-short (long cheap / short rich), monthly")
h = st.headline_stats(port)
print(f"  months {h['n_months']}  ({h['start']} -> {h['end']})")
print(f"  mean {h['mean_bps']:+.2f} bps/mo  ann {h['ann_pct']:+.2f}%  vol {h['vol_ann_pct']:.2f}%  "
      f"Sharpe {h['sharpe']:.2f}")
print(f"  one-sample t = {h['t_one_sample']:+.2f}   Newey-West(6) t = {h['t_nw']:+.2f}")
print(f"  hit {h['hit']}/{h['n_months']} = {h['hit_rate']*100:.1f}%  "
      f"Wilson 95% [{h['hit_lo']*100:.1f}%, {h['hit_hi']*100:.1f}%]")

print("\n# Random-sign placebo (20 seeds x 500 draws = 10,000 books)")
pl = st.placebo_pvalue(sig, rets)
print(f"  observed Sharpe {pl['obs_sharpe']:.3f} vs placebo mean {pl['placebo_mean']:.3f} "
      f"(sd {pl['placebo_sd']:.3f}) over {pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Trailing-window robustness sweep")
for wnd in (36, 48, 60, 84):
    s2 = st.value_signal(logq, window=wnd, min_periods=data.MIN_TRAIL)
    h2 = st.headline_stats(st.portfolio_returns(s2, rets))
    print(f"  window={wnd:>2} mo: Sharpe {h2['sharpe']:.2f}  NW t = {h2['t_nw']:+.2f}")

print("\n# Sub-period (a snooped split at 2015 — reported, not certified)")
for lbl, lo, hi in [("2005->2014", "2000", "2015-01-01"), ("2015->2025-06", "2015-01-01", "2100")]:
    seg = port[(port.index >= lo) & (port.index < hi)].to_numpy()
    print(f"  {lbl}: n={len(seg)} Sharpe {st.annualized_sharpe(seg):.2f} NW t = {st.newey_west_t(seg):+.2f}")

print("\n# THE TIMER — turnover x cost both legs, shorts pay 100 bps/yr borrow")
for cb in (5.0, 10.0, 20.0):
    t = st.timer_stats(sig, rets, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps: gross {t['gross_ann_pct']:+.2f}%/yr -> net "
          f"{t['net_ann_pct']:+.2f}%/yr (Sharpe {t['net_sharpe']:.2f}, NW t {t['net_t_nw']:+.2f}, "
          f"turnover {t['avg_turnover']:.2f}, worst month {t['worst_month_pct']:+.2f}%)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the value sort must NOT fire on a null panel (value_strength=0) and must recover")
print("  a planted PPP reversion. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    w = data.synthetic_world(value_strength=0.0, seed=797 + s_)
    null_ts.append(st.synthetic_detect(w)["t_nw"])
null_ts = np.asarray(null_ts)
print(f"  null (value_strength=0), 20 seeds: mean NW t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
planted = st.synthetic_detect(data.synthetic_world(value_strength=0.15, seed=797))
print(f"  planted reversion (value_strength=0.15, seed 797): NW t = {planted['t_nw']:+.2f} "
      f"(Sharpe {planted['sharpe']:.2f}, {planted['ann_pct']:+.2f}%/yr)")
