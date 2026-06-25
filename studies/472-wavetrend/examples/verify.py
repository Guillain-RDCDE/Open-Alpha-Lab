"""Reproducible headline run for Study 472 — WaveTrend (LazyBear).

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

from wavetrend import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch cross-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# WaveTrend (LazyBear) — mechanical oversold cross-up on 5 indices/ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")

    print("\n# Pooled WaveTrend cross-up vs drift-matched random baseline (single seed=7)")
    print(f"  {'H':>3} {'n':>5} {'cross_bps':>10} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    total = 0
    for h in st.HORIZONS:
        tt, rr, ne = [], [], []
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            c = b["close"]
            e = st.cross_up_entries(b)
            re = st.random_entries(c, max(len(e), 50), seed=7)
            tt.append(st.forward_returns(c, e, h))
            rr.append(st.forward_returns(c, re, h))
            ne.append(st.forward_returns(c, e, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                total += len(e)
        tt = np.concatenate(tt); rr = np.concatenate(rr); ne = np.concatenate(ne)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>10.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")
    print(f"  total WaveTrend cross-ups (H={st.HORIZONS[0]}): {total}")

    # ------------------------------------------------------------------ #
    # THE DECISIVE TEST — seed-averaged Welch t (a single lucky seed can
    # throw a t>2 that is NOT a real edge; cf. Study 452). We sweep 30
    # baseline seeds and report mean + spread (min/max) + fraction >= 2
    # per horizon. A Real stamp is only legitimate if the seed-AVERAGED
    # Welch t robustly clears 2.
    # ------------------------------------------------------------------ #
    print("\n# Seed-robustness of the cross-vs-random Welch t (30 seeds) — the honest bar")
    print(f"  {'H':>3} {'mean_t':>8} {'min_t':>8} {'max_t':>8} {'frac>=2':>8}")
    SEEDS = list(range(1, 31))
    for h in st.HORIZONS:
        tt = []
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            tt.append(st.forward_returns(b["close"], st.cross_up_entries(b), h))
        tt = np.concatenate(tt)
        welch = []
        for sd in SEEDS:
            rr = []
            for t in data.DEFAULT_TICKERS:
                b = _load(t)
                e = st.cross_up_entries(b)
                re = st.random_entries(b["close"], max(len(e), 50), seed=sd)
                rr.append(st.forward_returns(b["close"], re, h))
            rr = np.concatenate(rr)
            w = (stats.ttest_ind(tt, rr, equal_var=False)[0] if HAVE_SCIPY else float("nan"))
            welch.append(w)
        welch = np.asarray(welch)
        print(f"  {h:>3} {welch.mean():>8.2f} {welch.min():>8.2f} {welch.max():>8.2f} "
              f"{(welch >= 2).mean():>8.2f}")
    print("  (Real bar = seed-AVERAGED Welch t robustly >= 2; a single-seed t>2 is not enough.)")

    print("\n# Cross vs the zero-sampling-noise UNCONDITIONAL drift (every bar's fwd return)")
    print(f"  {'H':>3} {'welch_t':>8} {'p':>6}")
    for h in st.HORIZONS:
        tt, allb = [], []
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            c = b["close"]
            tt.append(st.forward_returns(c, st.cross_up_entries(b), h))
            allb.append(st.forward_returns(c, c.index[40:], h))
        tt = np.concatenate(tt); allb = np.concatenate(allb)
        w, p = (stats.ttest_ind(tt, allb, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {w:>8.2f} {p:>6.3f}")

    print("\n# Per-ticker, H=20: cross-up one-sample t and cross-minus-random delta")
    for t in data.DEFAULT_TICKERS:
        b = _load(t)
        c = b["close"]
        e = st.cross_up_entries(b)
        re = st.random_entries(c, max(len(e), 50), seed=7)
        s = st.summarize(st.forward_returns(c, e, 20))
        sr = st.summarize(st.forward_returns(c, re, 20))
        print(f"  {t}: ent={len(e):>3} cross={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Scrambled-signal GEOMETRY placebo (SPY, H=20, 500 draws)")
    pl = st.scrambled_signal_placebo(_load("SPY"), 20, n_draws=500, seed=472)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => geometry not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted bounce must light up.")
for edge in (0.0, 0.60):
    px, _ = data.synthetic_panel(edge=edge, seed=472, n_days=4000)
    c = px["close"]
    s = st.summarize(st.forward_returns(c, st.cross_up_entries(px), 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  cross={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
