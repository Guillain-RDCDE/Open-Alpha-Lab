"""Reproducible headline run for Study 482 — VWMA-Crossover.

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

from vwma_crossover import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch two-sample tests
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
FAST, SLOW = st.FAST, st.SLOW


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# VWMA-Crossover — VWMA golden-cross vs same-length SMA cross (yfinance daily, 5 ETFs)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}   fast/slow: {FAST}/{SLOW}")

    print("\n# Pooled VWMA cross vs SMA cross vs drift-matched random baseline")
    print(f"  {'H':>3} {'nV':>4} {'vwma_bps':>9} {'win%':>5} {'1samp_t':>8} "
          f"{'sma_bps':>8} {'dVsma':>7} {'wT_VS':>6} {'rnd_bps':>8} {'dVrnd':>7} {'wT_VR':>6} {'net':>7}")
    nv_total = ns_total = 0
    for h in st.HORIZONS:
        vv, ss, rr, nn = [], [], [], []
        for t in data.DEFAULT_TICKERS:
            b = _load(t); c = b["close"]; v = b["volume"]
            ev = st.vwma_cross_entries(c, v, FAST, SLOW)
            es = st.sma_cross_entries(c, FAST, SLOW)
            re = st.random_entries(c, max(len(ev), 50), warmup=SLOW, seed=7)
            vv.append(st.forward_returns(c, ev, h))
            ss.append(st.forward_returns(c, es, h))
            rr.append(st.forward_returns(c, re, h))
            nn.append(st.forward_returns(c, ev, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                nv_total += len(ev); ns_total += len(es)
        vv = np.concatenate(vv); ss = np.concatenate(ss); rr = np.concatenate(rr); nn = np.concatenate(nn)
        sv = st.summarize(vv); ssm = st.summarize(ss); srn = st.summarize(rr); snn = st.summarize(nn)
        wt_vs, _ = (stats.ttest_ind(vv, ss, equal_var=False) if HAVE_SCIPY else (float("nan"), 0))
        wt_vr, _ = (stats.ttest_ind(vv, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), 0))
        print(f"  {h:>3} {sv['n']:>4} {sv['mean_bps']:>9.1f} {sv['win']*100:>5.0f} {sv['t']:>8.2f} "
              f"{ssm['mean_bps']:>8.1f} {sv['mean_bps']-ssm['mean_bps']:>7.1f} {wt_vs:>6.2f} "
              f"{srn['mean_bps']:>8.1f} {sv['mean_bps']-srn['mean_bps']:>7.1f} {wt_vr:>6.2f} {snn['mean_bps']:>7.1f}")
    print(f"  total VWMA crosses (H={st.HORIZONS[0]}): {nv_total}   total SMA crosses: {ns_total}")

    print("\n# Per-ticker, H=20: VWMA cross one-sample t and VWMA-minus-SMA delta")
    for t in data.DEFAULT_TICKERS:
        b = _load(t); c = b["close"]; v = b["volume"]
        ev = st.vwma_cross_entries(c, v, FAST, SLOW)
        es = st.sma_cross_entries(c, FAST, SLOW)
        sv = st.summarize(st.forward_returns(c, ev, 20))
        ssm = st.summarize(st.forward_returns(c, es, 20))
        print(f"  {t}: nV={len(ev):>3} vwma={sv['mean_bps']:+7.1f}bps t={sv['t']:+5.2f} "
              f"sma={ssm['mean_bps']:+7.1f}bps delta={sv['mean_bps']-ssm['mean_bps']:+7.1f}")

    print("\n# Shuffled-VOLUME placebo (SPY, H=20, 500 draws)")
    b = _load("SPY")
    pl = st.shuffled_volume_placebo(b["close"], b["volume"], 20, FAST, SLOW, n_draws=500, seed=482)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => volume weighting not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT separate VWMA from SMA; volume-led pulse must light up VWMA-minus-SMA.")
for edge in (0.0, 0.60):
    px, _ = data.synthetic_panel(edge=edge, seed=482, n_days=4000)
    c = px["close"]; v = px["volume"]
    sv = st.summarize(st.forward_returns(c, st.vwma_cross_entries(c, v, FAST, SLOW), 20))
    ssm = st.summarize(st.forward_returns(c, st.sma_cross_entries(c, FAST, SLOW), 20))
    print(f"  edge={edge:.2f}: nV={sv['n']:>4} vwma={sv['mean_bps']:+7.1f}bps win={sv['win']*100:.0f}% "
          f"t={sv['t']:+.2f}  sma={ssm['mean_bps']:+7.1f}bps  vwma-sma={sv['mean_bps']-ssm['mean_bps']:+7.1f}")
