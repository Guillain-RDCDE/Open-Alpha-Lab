"""Reproducible headline run for Study 494 — Bullish Percent Index.

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

from bullish_percent_index import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch cross-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
MA = st.DEFAULT_MA
OS = st.OVERSOLD


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Bullish Percent Index — mechanical oversold-cross buy on SPY (breadth proxy, yfinance daily)")
if data.have_real():
    panel = data.breadth_basket(allow_fetch=False)
    panel = panel[panel.index <= ASOF]
    bser = st.bpi(panel, ma_win=MA)
    spy = _load("SPY")
    spy = spy[spy.index.isin(bser.index)]
    bser = bser[bser.index.isin(spy.index)]
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   breadth members: {data.BREADTH_MEMBERS}")
    bb = bser.dropna()
    print(f"BPI range: {bb.min():.0f}..{bb.max():.0f}  mean={bb.mean():.1f}  "
          f"%<{OS:.0f}={float((bb < OS).mean())*100:.1f}%  %>70={float((bb > 70).mean())*100:.1f}%")

    ent = st.oversold_cross_entries(bser, OS)
    lvl = st.oversold_level_entries(bser, OS)
    print(f"oversold crosses: {len(ent)}   oversold-level firsts: {len(lvl)}")

    cl = spy["close"]
    print("\n# Oversold-cross buy on SPY vs drift-matched random baseline")
    print(f"  {'H':>3} {'n':>4} {'cross_bps':>10} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    for h in st.HORIZONS:
        tt = st.forward_returns(cl, ent, h)
        rr = st.forward_returns(cl, st.random_entries(cl, max(len(ent), 50), seed=7), h)
        ne = st.forward_returns(cl, ent, h, cost_bps=1.0)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY and tt.size and rr.size
                  else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>4} {s['mean_bps']:>10.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")

    print("\n# Level rule (BPI < 30, first of run) on SPY, vs random, all horizons")
    print(f"  {'H':>3} {'n':>4} {'lvl_bps':>9} {'1samp_t':>8} {'rnd_bps':>8} {'delta':>7}")
    for h in st.HORIZONS:
        tt = st.forward_returns(cl, lvl, h)
        rr = st.forward_returns(cl, st.random_entries(cl, max(len(lvl), 50), seed=7), h)
        s = st.summarize(tt); sr = st.summarize(rr)
        print(f"  {h:>3} {s['n']:>4} {s['mean_bps']:>9.1f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f}")

    print("\n# Per-instrument, H=20: oversold-cross one-sample t and cross-minus-random delta")
    print("  (same breadth BPI signal, applied to each tradable tape)")
    for t in data.DEFAULT_TICKERS:
        c = _load(t)["close"]
        c = c[c.index.isin(bser.index)]
        e = st.oversold_cross_entries(bser, OS)
        s = st.summarize(st.forward_returns(c, e, 20))
        sr = st.summarize(st.forward_returns(c, st.random_entries(c, max(len(e), 50), seed=7), 20))
        print(f"  {t}: ent={s['n']:>3} cross={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Scrambled-breadth placebo (SPY, H=20, 500 draws)")
    pl = st.scrambled_breadth_placebo(cl, bser, 20, oversold=OS, n_draws=500, seed=494)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => breadth timing not load-bearing)  draws={pl['n_draws']}")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted oversold-bounce must light up.")
for edge in (0.0, 0.60):
    bars, mclose, _ = data.synthetic_panel(edge=edge, seed=7, n_days=4000)
    bser = st.bpi(mclose, ma_win=st.DEFAULT_MA)
    cl = bars["close"]
    e = st.oversold_cross_entries(bser, OS)
    s = st.summarize(st.forward_returns(cl, e, 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  cross={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
