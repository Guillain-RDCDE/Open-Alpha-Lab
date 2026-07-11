"""Reproducible headline run for Study 675 — Qstick.

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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from qstick import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch cross-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-06-30"     # last complete calendar month at publication (2026-07-10)
SMOOTH = st.SMOOTH


def _load(t):
    b = data.load_real(t, allow_fetch=True)
    return b[b.index <= ASOF]


print("# Qstick — mechanical smoothed-body up-cross on 5 indices/ETFs (yfinance daily)")
if not data.have_real():
    print("(cache miss — fetching SPY/QQQ/IWM/DIA/GLD once)")
    for t in data.DEFAULT_TICKERS:
        data.load_real(t, allow_fetch=True)

spy = _load("SPY")
span = (spy.index[-1] - spy.index[0]).days / 365.25
print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
print(f"tickers  : {data.DEFAULT_TICKERS}   smooth={SMOOTH} (Chande & Kroll's original window)")
for t in data.DEFAULT_TICKERS:
    print(data_stamp(t, _load(t), asof=ASOF))

print("\n# Pooled smoothed-Qstick up-cross vs drift-matched random baseline")
print(f"  {'H':>3} {'n':>5} {'cross_bps':>10} {'win%':>5} {'1samp_t':>8} "
      f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
total = 0
for h in st.HORIZONS:
    tt, rr, ne = [], [], []
    for t in data.DEFAULT_TICKERS:
        b = _load(t)
        c = b["close"]
        e = st.qstick_cross_entries(b, smooth=SMOOTH)
        re = st.random_entries(b, max(len(e), 50), smooth=SMOOTH, seed=7)
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
print(f"  total smoothed-Qstick up-crosses (H={st.HORIZONS[0]}): {total}")

print("\n# Per-ticker, H=20: cross one-sample t and cross-minus-random delta")
for t in data.DEFAULT_TICKERS:
    b = _load(t)
    c = b["close"]
    e = st.qstick_cross_entries(b, smooth=SMOOTH)
    re = st.random_entries(b, max(len(e), 50), smooth=SMOOTH, seed=7)
    s = st.summarize(st.forward_returns(c, e, 20))
    sr = st.summarize(st.forward_returns(c, re, 20))
    print(f"  {t}: ent={len(e):>3} cross={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
          f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

print("\n# Sign-scramble ordering placebo (SPY, H=20, 500 draws)")
pl = st.scramble_placebo(_load("SPY"), 20, smooth=SMOOTH, n_draws=500, seed=675)
print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
      f"(>0.05 => body ordering not load-bearing)")

print("\n# 'Just a slow trend proxy?' — Qstick vs plain trailing-momentum")
print(f"  {'ticker':>6} {'corr':>7} {'qs_x':>6} {'mom_x':>6} {'jaccard':>8}")
corrs = []
for t in data.DEFAULT_TICKERS:
    b = _load(t)
    tp = st.trend_proxy_stats(b, smooth=SMOOTH)
    corrs.append(tp["corr"])
    print(f"  {t:>6} {tp['corr']:>7.3f} {tp['n_qs_entries']:>6} {tp['n_mom_entries']:>6} "
          f"{tp['jaccard']:>8.3f}")
print(f"  mean correlation across basket: {np.mean(corrs):+.3f}")

tt, rr = [], []
for t in data.DEFAULT_TICKERS:
    b = _load(t)
    c = b["close"]
    e = st.momentum_cross_entries(b, n=SMOOTH)
    re = st.random_entries(b, max(len(e), 50), smooth=SMOOTH, seed=7)
    tt.append(st.forward_returns(c, e, 20))
    rr.append(st.forward_returns(c, re, 20))
tt = np.concatenate(tt); rr = np.concatenate(rr)
sm_, srm = st.summarize(tt), st.summarize(rr)
wtm, wpm = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
print(f"  the naive momentum-only cross (no open/close split at all), H=20 pooled: "
      f"n={sm_['n']}  cross={sm_['mean_bps']:+.1f}bps  rnd={srm['mean_bps']:+.1f}bps  "
      f"delta={sm_['mean_bps']-srm['mean_bps']:+.1f}  Welch t={wtm:+.2f}  p={wpm:.3f}")
print("  (i.e. neither the open/close split nor the plain price-only version convincingly")
print("   beats a random entry — Qstick's split adds no information over the trend it echoes)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=8000)")
print("  the null (edge=0) is checked over 20 seeds (never a single stream); a planted")
print("  Qstick-respecting factor (edge=2.0) must light up.")
null_ts = []
for s_ in range(20):
    px, _ = data.synthetic_panel(edge=0.0, seed=675 + s_, n_days=8000)
    s = st.summarize(st.forward_returns(
        px["close"], st.qstick_cross_entries(px, smooth=SMOOTH), 20))
    null_ts.append(s["t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
px, _ = data.synthetic_panel(edge=2.0, seed=675, n_days=8000)
s = st.summarize(st.forward_returns(px["close"], st.qstick_cross_entries(px, smooth=SMOOTH), 20))
print(f"  planted edge=2.0 (seed 675): n={s['n']}  cross={s['mean_bps']:+.1f}bps  "
      f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
