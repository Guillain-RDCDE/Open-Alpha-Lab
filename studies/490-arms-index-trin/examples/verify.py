"""Reproducible headline run for Study 490 — Arms Index (TRIN).

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

from arms_index_trin import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch panic-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
Q = 0.90
TRADED = "SPY"


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


def _panel():
    panel = {}
    for t in data.DEFAULT_TICKERS:
        panel[t] = _load(t)
    return panel


print("# Arms Index (TRIN) — breadth-proxy panic-day entry on SPY (yfinance daily basket)")
if data.have_real():
    panel = _panel()
    trin = st.compute_trin(panel)
    close = panel[TRADED]["close"]
    close = close[close.index.isin(trin.index)]
    trin = trin[trin.index.isin(close.index)]
    span = (close.index[-1] - close.index[0]).days / 365.25
    print(f"window   : {close.index[0].date()} -> {close.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(panel['SPY'])}   basket: {data.DEFAULT_TICKERS}")
    thr = float(trin.quantile(Q))
    print(f"TRIN     : median={trin.median():.2f}  q90={thr:.2f}  (panic threshold)")

    ent = st.panic_entries(trin, q=Q)
    print(f"\n# Pooled high-TRIN panic entry (q={Q}) vs drift-matched random baseline  (traded={TRADED})")
    print(f"  {'H':>3} {'n':>5} {'panic_bps':>10} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    for h in st.HORIZONS:
        tt = st.forward_returns(close, ent, h)
        ne = st.forward_returns(close, ent, h, cost_bps=1.0)
        re = st.random_entries(close, max(len(ent), 50), seed=7)
        rr = st.forward_returns(close, re, h)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>10.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")
    print(f"  total panic entries: {len(ent)}")

    print("\n# Threshold sweep, H=10: does a higher panic cutoff help?")
    for q in (0.80, 0.90, 0.95):
        e = st.panic_entries(trin, q=q)
        s = st.summarize(st.forward_returns(close, e, 10))
        re = st.random_entries(close, max(len(e), 50), seed=7)
        sr = st.summarize(st.forward_returns(close, re, 10))
        print(f"  q={q:.2f}: ent={len(e):>3} panic={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Shuffled-TRIN TIMING placebo (SPY, H=10, 500 draws)")
    pl = st.shuffled_trin_placebo(trin, close, 10, q=Q, n_draws=500, seed=490)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => timing not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=10, n_days=4000)")
print("  edge=0 must NOT reach significance; planted bounce must light up.")
for edge in (0.0, 0.60):
    panel, truth = data.synthetic_panel(edge=edge, seed=490, n_days=4000)
    trin = st.compute_trin(panel)
    close = panel[truth["traded"]]["close"]
    close = close[close.index.isin(trin.index)]
    trin = trin[trin.index.isin(close.index)]
    e = st.panic_entries(trin, q=Q)
    s = st.summarize(st.forward_returns(close, e, 10))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  panic={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
