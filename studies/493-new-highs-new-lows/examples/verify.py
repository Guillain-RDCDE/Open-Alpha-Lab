"""Reproducible headline run for Study 493 — New-Highs-New-Lows breadth.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_highs_new_lows import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch thrust-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
IDX = "SPY"


print("# New-Highs-New-Lows breadth — mechanical thrust on a liquid-ETF breadth proxy (yfinance daily)")
if data.have_real():
    panel = data.load_basket(allow_fetch=False, asof=ASOF)
    print(f"breadth basket ({len(panel)} members): {sorted(panel)}")
    spy = panel[IDX]
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}")

    idx_close = spy["close"]
    ent = st.breadth_thrust_entries(panel, IDX)
    print(f"breadth-thrust entries: {len(ent)}  (lookback={st.LOOKBACK}, smooth={st.SMOOTH}, "
          f"thresh={st.THRESH})")

    print("\n# Breadth-thrust vs drift-matched random baseline (SPY)")
    print(f"  {'H':>3} {'n':>5} {'thr_bps':>9} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    for h in st.HORIZONS:
        tt = st.forward_returns(idx_close, ent, h)
        rr = st.forward_returns(idx_close, st.random_entries(idx_close, max(len(ent), 50), seed=7), h)
        ne = st.forward_returns(idx_close, ent, h, cost_bps=1.0)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY and tt.size and rr.size
                  else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>9.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")

    print("\n# Shuffled-membership breadth placebo (SPY, H=20, 500 draws)")
    pl = st.shuffled_membership_placebo(panel, IDX, horizon=20, n_draws=500, seed=493)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => breadth aggregation not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted breadth-lead must light up.")
for edge in (0.0, 0.60):
    panel_s, _ = data.synthetic_panel(edge=edge, seed=493, n_days=4000)
    ent_s = st.breadth_thrust_entries(panel_s, "SPY")
    s = st.summarize(st.forward_returns(panel_s["SPY"]["close"], ent_s, 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  thrust={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
