"""Reproducible headline run for Study 467 — Bump-and-Run Reversal.

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

from bump_and_run import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch break-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
LEAD, BUMP, MULT = 60, 30, 2.0


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Bump-and-run reversal — mechanical trendline-break SHORT on 5 indices/ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")

    print("\n# Pooled BARR-break SHORT vs drift-matched random baseline")
    print(f"  {'H':>3} {'n':>5} {'brk_bps':>9} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    total = 0
    for h in st.HORIZONS:
        tt, rr, ne = [], [], []
        for t in data.DEFAULT_TICKERS:
            c = _load(t)["close"]
            e = st.barr_break_entries(c, lead=LEAD, bump=BUMP, bump_mult=MULT)
            re = st.random_entries(c, max(len(e), 50), lead=LEAD, bump=BUMP, seed=7)
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
    print(f"  total BARR-break shorts (H={st.HORIZONS[0]}): {total}")

    print("\n# Per-ticker, H=20: break one-sample t and break-minus-random delta")
    for t in data.DEFAULT_TICKERS:
        c = _load(t)["close"]
        e = st.barr_break_entries(c, lead=LEAD, bump=BUMP, bump_mult=MULT)
        re = st.random_entries(c, max(len(e), 50), lead=LEAD, bump=BUMP, seed=7)
        s = st.summarize(st.forward_returns(c, e, 20))
        sr = st.summarize(st.forward_returns(c, re, 20))
        print(f"  {t}: ent={len(e):>3} brk={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Shuffled-window GEOMETRY placebo (SPY, H=20, 500 draws)")
    pl = st.shuffled_window_placebo(_load("SPY")["close"], 20, lead=LEAD, bump=BUMP,
                                    bump_mult=MULT, n_draws=500, seed=467)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => geometry not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=5, n_days=8000)")
print("  edge=0 must NOT reach significance; planted reversal must light up (short profits).")
for edge in (0.0, 0.70):
    px, _ = data.synthetic_panel(edge=edge, seed=467, n_days=8000)
    c = px["close"]
    e = st.barr_break_entries(c, lead=LEAD, bump=BUMP, bump_mult=MULT)
    s = st.summarize(st.forward_returns(c, e, 5))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  short={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
