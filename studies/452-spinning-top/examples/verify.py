"""Reproducible headline run for Study 452 — Spinning-Top.

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

from spinning_top import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch top-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
BODY, WICK, BAL = 0.25, 0.25, 0.5


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Spinning-Top — mechanical small-body/balanced-wick on 5 indices/ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")

    print("\n# Pooled spinning-top vs drift-matched random baseline")
    print("  (Welch t is AVERAGED over 20 random-baseline seeds — a single lucky draw of random")
    print("   dates is not evidence; we report the seed-robust mean and its spread.)")
    print(f"  {'H':>3} {'n':>5} {'top_bps':>10} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'[min,max]':>14}")
    total = 0
    N_SEEDS = 20
    for h in st.HORIZONS:
        tt, ne = [], []
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            e = st.spinning_top_entries(b, BODY, WICK, BAL)
            tt.append(st.forward_returns(b, e, h))
            ne.append(st.forward_returns(b, e, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                total += len(e)
        tt = np.concatenate(tt); ne = np.concatenate(ne)
        # seed-robust random baseline + Welch
        rnd_means, welch_ts = [], []
        for seed in range(N_SEEDS):
            rr = []
            for t in data.DEFAULT_TICKERS:
                b = _load(t)
                e = st.spinning_top_entries(b, BODY, WICK, BAL)
                re = st.random_entries(b, max(len(e), 50), seed=seed)
                rr.append(st.forward_returns(b, re, h))
            rr = np.concatenate(rr)
            rnd_means.append(rr.mean() * 1e4)
            if HAVE_SCIPY:
                welch_ts.append(stats.ttest_ind(tt, rr, equal_var=False)[0])
        s = st.summarize(tt); sn = st.summarize(ne)
        rnd_bps = float(np.mean(rnd_means))
        wt = float(np.mean(welch_ts)) if welch_ts else float("nan")
        wmin = float(np.min(welch_ts)) if welch_ts else float("nan")
        wmax = float(np.max(welch_ts)) if welch_ts else float("nan")
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>10.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{rnd_bps:>8.1f} {s['mean_bps']-rnd_bps:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {('['+format(wmin,'+.2f')+','+format(wmax,'+.2f')+']'):>14}")
    print(f"  total spinning tops (H={st.HORIZONS[0]}): {total}")

    print("\n# Per-ticker, H=20: spinning-top one-sample t and top-minus-random delta")
    for t in data.DEFAULT_TICKERS:
        b = _load(t)
        e = st.spinning_top_entries(b, BODY, WICK, BAL)
        re = st.random_entries(b, max(len(e), 50), seed=7)
        s = st.summarize(st.forward_returns(b, e, 20))
        sr = st.summarize(st.forward_returns(b, re, 20))
        print(f"  {t}: ent={len(e):>3} top={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Wick-scramble GEOMETRY placebo (SPY, H=20, 500 draws)")
    pl = st.wick_scramble_placebo(_load("SPY"), 20, BODY, WICK, BAL, n_draws=500, seed=452)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => geometry not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted resolution must light up.")
for edge in (0.0, 1.2):
    px, _ = data.synthetic_panel(edge=edge, seed=452, n_days=4000)
    e = st.spinning_top_entries(px, BODY, WICK, BAL)
    s = st.summarize(st.forward_returns(px, e, 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  top={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
