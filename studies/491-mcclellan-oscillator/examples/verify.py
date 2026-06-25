"""Reproducible headline run for Study 491 — McClellan Oscillator.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under
``_cache/`` if present (the real-tape numbers + the breadth proxy), and always runs the
synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcclellan_oscillator import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch trigger-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# McClellan Oscillator — EMA19-EMA39 of net advances; up-cross-from-negative long on SPY")
if data.have_real():
    members = data.breadth_members()
    net = data.load_breadth(asof=ASOF)
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"breadth basket : {members}  (proxy for true exchange breadth)")
    print(f"net_adv range  : [{int(net.min())}, {int(net.max())}]   ({len(net)} days)")
    print(f"window         : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp         : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")

    c = spy["close"]
    netA = net.reindex(c.index).dropna()
    ent = st.osc_up_cross_entries(netA)
    print(f"up-cross long entries on SPY: {len(ent)}")

    print("\n# McClellan up-cross vs drift-matched random baseline (SPY)")
    print(f"  {'H':>3} {'n':>4} {'touch_bps':>10} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    for h in st.HORIZONS:
        tt = st.forward_returns(c, ent, h)
        rr = st.forward_returns(c, st.random_entries(c.index, max(len(ent), 50), seed=7), h)
        nn = st.forward_returns(c, ent, h, cost_bps=1.0)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(nn)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>4} {s['mean_bps']:>10.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")

    print("\n# Per-ticker, H=20: trade each index on the SAME breadth up-cross")
    for t in data.DEFAULT_TICKERS:
        cc = _load(t)["close"]
        na = net.reindex(cc.index).dropna()
        e = st.osc_up_cross_entries(na)
        re = st.random_entries(cc.index, max(len(e), 50), seed=7)
        s = st.summarize(st.forward_returns(cc, e, 20))
        sr = st.summarize(st.forward_returns(cc, re, 20))
        print(f"  {t}: ent={len(e):>3} touch={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Shuffled-breadth GEOMETRY placebo (SPY, H=20, 500 draws)")
    pl = st.shuffled_breadth_placebo(c, netA, 20, n_draws=500, seed=491)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => breadth geometry not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted post-cross bounce must light up.")
for edge in (0.0, 0.30):
    px, _ = data.synthetic_panel(edge=edge, seed=491, n_days=4000)
    c = px["close"]
    s = st.summarize(st.forward_returns(c, st.osc_up_cross_entries(px["net_adv"]), 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  touch={s['mean_bps']:+8.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
