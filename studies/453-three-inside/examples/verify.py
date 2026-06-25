"""Reproducible headline run for Study 453 — Three-Inside-Up / Down.

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

from three_inside import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch confirmed-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
TL = 5  # trend lookback


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Three-inside-up — mechanical harami+confirmation long on 5 indices/ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")

    print("\n# Pooled confirmed three-inside-up vs drift-matched random baseline")
    print(f"  {'H':>3} {'n':>5} {'conf_bps':>9} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    total = 0
    for h in st.HORIZONS:
        tt, rr, ne = [], [], []
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            c = b["close"]
            e = st.three_inside_entries(b, trend_lookback=TL, require_confirm=True)
            re = st.random_entries(c, max(len(e), 50), seed=7)
            tt.append(st.forward_returns(c, e, h))
            rr.append(st.forward_returns(c, re, h))
            ne.append(st.forward_returns(c, e, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                total += len(e)
        tt = np.concatenate(tt); rr = np.concatenate(rr); ne = np.concatenate(ne)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>9.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")
    print(f"  total confirmed three-inside-up entries (H={st.HORIZONS[0]}): {total}")

    print("\n# Per-ticker, H=20: confirmed one-sample t and confirmed-minus-random delta")
    for t in data.DEFAULT_TICKERS:
        b = _load(t)
        c = b["close"]
        e = st.three_inside_entries(b, trend_lookback=TL, require_confirm=True)
        re = st.random_entries(c, max(len(e), 50), seed=7)
        s = st.summarize(st.forward_returns(c, e, 20))
        sr = st.summarize(st.forward_returns(c, re, 20))
        print(f"  {t}: ent={len(e):>3} conf={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Harami-only PLACEBO (thesis: does the confirmation candle add edge?), pooled, H=20")
    cf, hr = [], []
    for t in data.DEFAULT_TICKERS:
        b = _load(t)
        c = b["close"]
        ec = st.three_inside_entries(b, trend_lookback=TL, require_confirm=True)
        eh = st.three_inside_entries(b, trend_lookback=TL, require_confirm=False)
        cf.append(st.forward_returns(c, ec, 20)); hr.append(st.forward_returns(c, eh, 20))
    cf = np.concatenate(cf); hr = np.concatenate(hr)
    sc, sh = st.summarize(cf), st.summarize(hr)
    wt2, wp2 = (stats.ttest_ind(cf, hr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
    print(f"  confirmed: n={sc['n']} mean={sc['mean_bps']:+.1f}bps  "
          f"harami-only: n={sh['n']} mean={sh['mean_bps']:+.1f}bps")
    print(f"  confirmation adds delta={sc['mean_bps']-sh['mean_bps']:+.1f}bps  "
          f"Welch t={wt2:+.2f} p={wp2:.3f}  (t<2 => confirmation candle adds nothing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=8000)")
print("  edge=0 must NOT reach significance; planted bounce must light up.")
for edge in (0.0, 0.60):
    px, _ = data.synthetic_panel(edge=edge, seed=453, n_days=8000)
    c = px["close"]
    e = st.three_inside_entries(px, trend_lookback=TL, require_confirm=True)
    s = st.summarize(st.forward_returns(c, e, 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  conf={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
