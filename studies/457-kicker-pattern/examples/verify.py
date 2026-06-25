"""Reproducible headline run for Study 457 — Kicker-Pattern.

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

from kicker_pattern import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch kicker-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
BF = 0.60


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Kicker-pattern — mechanical gap-and-reverse marubozu pair on 5 indices/ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")

    # directional mix across the basket (for the data stamp)
    nlong = nshort = 0
    for t in data.DEFAULT_TICKERS:
        _, dd = st.kicker_entries(_load(t), body_frac=BF)
        nlong += int((dd > 0).sum()); nshort += int((dd < 0).sum())
    print(f"kickers  : {nlong} bullish (long) + {nshort} bearish (short)")

    print("\n# Pooled kicker-direction vs drift-matched random baseline (same long/short mix)")
    print(f"  {'H':>3} {'n':>5} {'kick_bps':>9} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    total = 0
    for h in st.HORIZONS:
        tt, rr, ne = [], [], []
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            e, dirs = st.kicker_entries(b, body_frac=BF)
            re, rdir = st.random_entries(b, max(len(e), 50), dirs=dirs, seed=7)
            tt.append(st.forward_returns(b, e, dirs, h))
            rr.append(st.forward_returns(b, re, rdir, h))
            ne.append(st.forward_returns(b, e, dirs, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                total += len(e)
        tt = np.concatenate(tt); rr = np.concatenate(rr); ne = np.concatenate(ne)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>9.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")
    print(f"  total kickers (H={st.HORIZONS[0]}): {total}")

    print("\n# Per-ticker, H=20: kicker one-sample t and kicker-minus-random delta")
    for t in data.DEFAULT_TICKERS:
        b = _load(t)
        e, dirs = st.kicker_entries(b, body_frac=BF)
        re, rdir = st.random_entries(b, max(len(e), 50), dirs=dirs, seed=7)
        s = st.summarize(st.forward_returns(b, e, dirs, 20))
        sr = st.summarize(st.forward_returns(b, re, rdir, 20))
        print(f"  {t}: ent={len(e):>3} kick={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Gap-scramble GEOMETRY placebo (SPY, H=20, 500 draws)")
    pl = st.gap_scramble_placebo(_load("SPY"), 20, body_frac=BF, n_draws=500, seed=457)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => gap geometry not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted continuation must light up.")
for edge in (0.0, 0.60):
    px, _ = data.synthetic_panel(edge=edge, seed=458, n_days=6000)
    e, dirs = st.kicker_entries(px, body_frac=BF)
    s = st.summarize(st.forward_returns(px, e, dirs, 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  kick={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
