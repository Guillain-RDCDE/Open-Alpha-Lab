"""Reproducible headline run for Study 483 — Zero-Lag EMA (ZLEMA).

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

from zlema import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch ZLEMA-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
L = st.DEFAULT_LENGTH


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Zero-Lag EMA — mechanical price>ZLEMA upcross on 5 indices/ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}   length={L}")

    print("\n# Pooled ZLEMA upcross vs drift-matched random baseline (and plain-EMA head-to-head)")
    print(f"  {'H':>3} {'n':>5} {'zlema_bps':>10} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'d_rnd':>7} {'ema_bps':>8} {'d_ema':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    total = 0
    for h in st.HORIZONS:
        zz, rr, ee, nn = [], [], [], []
        for t in data.DEFAULT_TICKERS:
            c = _load(t)["close"]
            e = st.zlema_entries(c, length=L)
            em = st.ema_entries(c, length=L)
            re = st.random_entries(c, max(len(e), 50), length=L, seed=7)
            zz.append(st.forward_returns(c, e, h))
            rr.append(st.forward_returns(c, re, h))
            ee.append(st.forward_returns(c, em, h))
            nn.append(st.forward_returns(c, e, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                total += len(e)
        zz = np.concatenate(zz); rr = np.concatenate(rr); ee = np.concatenate(ee); nn = np.concatenate(nn)
        s = st.summarize(zz); sr = st.summarize(rr); se = st.summarize(ee); sn = st.summarize(nn)
        wt, wp = (stats.ttest_ind(zz, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>10.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {se['mean_bps']:>8.1f} "
              f"{s['mean_bps']-se['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} {wt:>8.2f} {wp:>6.3f}")
    print(f"  total ZLEMA upcross entries (H={st.HORIZONS[0]}): {total}")

    print("\n# Per-ticker, H=20: ZLEMA one-sample t, ZLEMA-minus-random and ZLEMA-minus-EMA deltas")
    for t in data.DEFAULT_TICKERS:
        c = _load(t)["close"]
        e = st.zlema_entries(c, length=L)
        em = st.ema_entries(c, length=L)
        re = st.random_entries(c, max(len(e), 50), length=L, seed=7)
        s = st.summarize(st.forward_returns(c, e, 20))
        sr = st.summarize(st.forward_returns(c, re, 20))
        se = st.summarize(st.forward_returns(c, em, 20))
        print(f"  {t}: ent={len(e):>3} zlema={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps d_rnd={s['mean_bps']-sr['mean_bps']:+7.1f} "
              f"d_ema={s['mean_bps']-se['mean_bps']:+7.1f}")

    print("\n# De-lag GEOMETRY placebo (SPY, H=20, 500 draws)")
    pl = st.delag_placebo(_load("SPY")["close"], 20, length=L, n_draws=500, seed=483)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => de-lag correction not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# ZLEMA upcross foil (SPY, H=20) — the bare cross whipsaws vs the steady filter")
if data.have_real():
    c = _load("SPY")["close"]
    su = st.summarize(st.forward_returns(c, st.zlema_upcross_entries(c, length=L), 20))
    print(f"  SPY upcross: n={su['n']:>3}  mean={su['mean_bps']:+7.1f}bps  win={su['win']*100:.0f}%  t={su['t']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted trend persistence must light up.")
for edge in (0.0, 2.0):
    px, _ = data.synthetic_panel(edge=edge, seed=483, n_days=4000)
    c = px["close"]
    s = st.summarize(st.forward_returns(c, st.zlema_entries(c, length=L), 20))
    se = st.summarize(st.forward_returns(c, st.ema_entries(c, length=L), 20))
    print(f"  edge={edge:.2f}: zlema n={s['n']:>4} mean={s['mean_bps']:+8.1f}bps win={s['win']*100:.0f}% "
          f"t={s['t']:+.2f}  | ema mean={se['mean_bps']:+8.1f}bps t={se['t']:+.2f}")
