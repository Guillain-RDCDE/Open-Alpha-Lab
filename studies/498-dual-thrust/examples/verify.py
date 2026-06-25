"""Reproducible headline run for Study 498 — Dual Thrust.

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

from dual_thrust import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch breakout-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
N = st.DEFAULT_N
K1 = st.DEFAULT_K1
K2 = st.DEFAULT_K2


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Dual Thrust — mechanical upper-trigger breakout on 5 indices/ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")
    print(f"params   : N={N}  k1={K1}  k2={K2}")

    print("\n# Pooled breakout vs drift-matched random baseline")
    print(f"  {'H':>3} {'n':>5} {'brk_bps':>8} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    total = 0
    for h in st.HORIZONS:
        tt, rr, ne = [], [], []
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            c = b["close"]
            e = st.breakout_entries(b, n=N, k1=K1, k2=K2)
            re = st.random_entries(b, max(len(e), 50), n=N, seed=7)
            tt.append(st.forward_returns(c, e, h))
            rr.append(st.forward_returns(c, re, h))
            ne.append(st.forward_returns(c, e, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                total += len(e)
        tt = np.concatenate(tt); rr = np.concatenate(rr); ne = np.concatenate(ne)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>8.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")
    print(f"  total breakouts (H={st.HORIZONS[0]}): {total}")

    print("\n# Per-ticker, H=20: breakout one-sample t and breakout-minus-random delta")
    for t in data.DEFAULT_TICKERS:
        b = _load(t)
        c = b["close"]
        e = st.breakout_entries(b, n=N, k1=K1, k2=K2)
        re = st.random_entries(b, max(len(e), 50), n=N, seed=7)
        s = st.summarize(st.forward_returns(c, e, 20))
        sr = st.summarize(st.forward_returns(c, re, 20))
        print(f"  {t}: ent={len(e):>4} brk={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Scrambled-Range GEOMETRY placebo (SPY, H=20, 500 draws)")
    pl = st.scrambled_range_placebo(_load("SPY"), 20, n=N, k1=K1, k2=K2, n_draws=500, seed=498)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => geometry not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=6000)")
print("  edge=0 must NOT beat random; planted continuation must light up vs random.")
for edge in (0.0, 2.0):
    px, _ = data.synthetic_panel(edge=edge, seed=498, n_days=6000)
    c = px["close"]
    e = st.breakout_entries(px, n=N, k1=K1, k2=K2)
    re = st.random_entries(px, max(len(e), 50), n=N, seed=7)
    tt = st.forward_returns(c, e, 20)
    rr = st.forward_returns(c, re, 20)
    s = st.summarize(tt)
    wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  brk={s['mean_bps']:+7.1f}bps  win={s['win']*100:.0f}%  "
          f"1samp_t={s['t']:+.2f}  rnd={rr.mean()*1e4:+.1f}bps  delta={(tt.mean()-rr.mean())*1e4:+.1f}  "
          f"welch_t={wt:+.2f} p={wp:.3f}")
