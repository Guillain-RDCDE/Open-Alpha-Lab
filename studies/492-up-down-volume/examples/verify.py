"""Reproducible headline run for Study 492 — Up-Down-Volume breadth.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under ``_cache/`` if
present (the OHLC tape for SPY forward returns + the OHLCV breadth basket), and always runs the
synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from up_down_volume import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch climax-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
WINDOW = 60
Q = 0.10


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


def _breadth(allow=False):
    bk = data.load_breadth(allow_fetch=allow)
    bk = {t: df[df.index <= ASOF] for t, df in bk.items()}
    return st.up_down_volume(bk)


print("# Up/down-volume selling-climax on SPY, breadth from 10 sector ETFs (yfinance daily)")
if data.have_real() and data.have_breadth():
    spy = _load("SPY")
    br = _breadth(allow=False)
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   breadth basket: {data.BREADTH_TICKERS}")
    print(f"breadth rows: {len(br)}  uvs mean={br['uvs'].mean():+.3f} sd={br['uvs'].std():.3f}")

    spy_close = spy["close"]
    ent = st.climax_entries(br["uvs"], window=WINDOW, q=Q, side="selling")
    ent = ent[ent.isin(spy_close.index)]
    print(f"\n# Selling-climax entries on SPY (q={Q}, window={WINDOW}): n={len(ent)}")
    print("# Climax vs drift-matched random baseline (SPY forward returns)")
    print(f"  {'H':>3} {'n':>5} {'cli_bps':>9} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    for h in st.HORIZONS:
        tt = st.forward_returns(spy_close, ent, h)
        ne = st.forward_returns(spy_close, ent, h, cost_bps=1.0)
        re = st.random_entries(spy_close, max(len(ent), 50), warmup=WINDOW, seed=7)
        rr = st.forward_returns(spy_close, re, h)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>9.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")

    print("\n# Per-instrument, H=20: climax (breadth fixed) applied to each ETF's forward return")
    print("# (same climax dates, different forward instrument) + its own random baseline")
    for t in data.DEFAULT_TICKERS:
        c = _load(t)["close"]
        e = ent[ent.isin(c.index)]
        re = st.random_entries(c, max(len(e), 50), warmup=WINDOW, seed=7)
        s = st.summarize(st.forward_returns(c, e, 20))
        sr = st.summarize(st.forward_returns(c, re, 20))
        print(f"  {t}: ent={len(e):>3} climax={s['mean_bps']:+8.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+8.1f}bps delta={s['mean_bps']-sr['mean_bps']:+8.1f}")

    print("\n# Shuffled-volume GEOMETRY placebo (SPY, H=20, 500 draws)")
    pl = st.shuffled_volume_placebo(spy_close, br, 20, window=WINDOW, q=Q,
                                    side="selling", n_draws=500, seed=492)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => up/down structure not load-bearing)")

    print("\n# Buying-climax check (the symmetric blow-off-top rule), SPY H=20")
    be = st.climax_entries(br["uvs"], window=WINDOW, q=Q, side="buying")
    be = be[be.isin(spy_close.index)]
    sb = st.summarize(st.forward_returns(spy_close, be, 20))
    print(f"  buying-climax n={sb['n']}  fwd20={sb['mean_bps']:+.1f}bps win={sb['win']*100:.0f}% t={sb['t']:+.2f}")
else:
    print("(no _cache parquets — call data.load_real(t)/load_breadth() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted selling-climax bounce must light up.")
for edge in (0.0, 0.40):
    px, _ = data.synthetic_panel(edge=edge, seed=492, n_days=4000)
    c = px["close"]
    br_s = st.breadth_from_panel(px)
    e = st.climax_entries(br_s["uvs"], window=WINDOW, q=Q, side="selling")
    s = st.summarize(st.forward_returns(c, e, 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  climax={s['mean_bps']:+8.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
