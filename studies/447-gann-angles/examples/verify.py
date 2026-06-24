"""Reproducible headline run for Study 447 — Gann Angles (the 1x1 line).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under ``_cache/``
(the real-tape numbers) and always runs the synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gann_angles import data, strategy as st

NDRAWS = 5000

print("# Gann 1x1 angle — long-when-above-the-line vs buy-and-hold (yfinance daily)")
print("# Decisive number: HAC t on the active-minus-benchmark daily spread (>= 2 = real)\n")

if data.have_real("SPY"):
    print(f"  {'tkr':>5} {'n':>5} {'span':>23} {'fp':>13} {'slope':>9} "
          f"{'on%':>5} {'flips':>5} {'Sh_s':>5} {'Sh_bh':>6} {'spread%':>8} "
          f"{'HAC_t':>6} {'plac_p':>7}")
    for tk in data.DEFAULT_TICKERS:
        b = data.load_real(tk, fetch=False)
        r = st.run_experiment(b, cost_bps=2.0, n_draws=NDRAWS)
        u, pl = r["unlatched"], r["placebo"]
        span = f"{b.index.min().date()}->{b.index.max().date()}"
        print(f"  {tk:>5} {u['n']:>5} {span:>23} {data.fingerprint(b):>13} "
              f"{r['slope']:>9.4f} {u['on_frac']*100:>4.0f} {u['n_flips']:>5} "
              f"{u['sharpe_strat']:>5.2f} {u['sharpe_bh']:>6.2f} "
              f"{u['spread_mean_ann']*100:>+7.2f} {u['t_spread']:>6.2f} "
              f"{pl['p_value']:>7.3f}")

    print("\n# Latched ('trend holds until the angle breaks') vs un-latched, SPY")
    b = data.load_real("SPY", fetch=False)
    for lat in (False, True):
        bt = st.backtest(b, latched=lat, cost_bps=2.0)
        print(f"  latched={str(lat):>5}: Sharpe={bt['sharpe_strat']:.2f}  "
              f"spread_ann={bt['spread_mean_ann']*100:+.2f}%  "
              f"HAC_t={bt['t_spread']:.2f}  flips={bt['n_flips']}")
else:
    print("  (no _cache/bars_SPY_1d.parquet — run data.load_real('SPY', fetch=True) once)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the line-timing detector must NOT fake significance at edge=0, and must light up")
print("  when a real line effect is planted (edge>0).")
print(f"  {'edge':>7} {'on%':>5} {'flips':>5} {'Sh_s':>5} {'Sh_bh':>6} "
      f"{'spread%':>8} {'HAC_t':>6} {'plac_p':>7}")
for edge in (0.0, 0.0008, 0.0015, 0.0030):
    b, _ = data.synthetic_panel(edge=edge)
    r = st.run_experiment(b, cost_bps=2.0, n_draws=2000)
    u, pl = r["unlatched"], r["placebo"]
    print(f"  {edge:>7.4f} {u['on_frac']*100:>4.0f} {u['n_flips']:>5} "
          f"{u['sharpe_strat']:>5.2f} {u['sharpe_bh']:>6.2f} "
          f"{u['spread_mean_ann']*100:>+7.2f} {u['t_spread']:>6.2f} {pl['p_value']:>7.3f}")
