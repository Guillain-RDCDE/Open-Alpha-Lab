"""Reproducible headline run for Study 484 — Vertical-Horizontal-Filter.

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

from vertical_horizontal_filter import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch gate-vs-ungated / gate-vs-random tests
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"
MOM_N = 50
VHF_N = 28
Q = 0.667
LOOKBACK = 252


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Vertical-Horizontal-Filter — VHF-gated momentum on 5 indices/ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")

    print("\n# Pooled: VHF-gated momentum vs ungated momentum vs drift-matched random baseline")
    print(f"  {'H':>3} {'nG':>5} {'gate_bps':>9} {'win%':>5} {'1samp_t':>8} "
          f"{'ung_bps':>8} {'Δgate':>7} {'rnd_bps':>8} {'Δrnd':>7} {'net':>7} "
          f"{'wt_ung':>7} {'p_ung':>6} {'wt_rnd':>7} {'p_rnd':>6}")
    n_gated_total = 0
    n_ungated_total = 0
    for h in st.HORIZONS:
        gg, uu, rr, ne = [], [], [], []
        for t in data.DEFAULT_TICKERS:
            c = _load(t)["close"]
            ge = st.gated_entries(c, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK)
            ue = st.momentum_entries(c, mom_n=MOM_N)
            re = st.random_entries(c, max(len(ge), 50), warmup=LOOKBACK, seed=7)
            gg.append(st.forward_returns(c, ge, h))
            uu.append(st.forward_returns(c, ue, h))
            rr.append(st.forward_returns(c, re, h))
            ne.append(st.forward_returns(c, ge, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                n_gated_total += len(ge)
                n_ungated_total += len(ue)
        gg = np.concatenate(gg); uu = np.concatenate(uu)
        rr = np.concatenate(rr); ne = np.concatenate(ne)
        s = st.summarize(gg); su = st.summarize(uu); sr = st.summarize(rr); sn = st.summarize(ne)
        if HAVE_SCIPY:
            wtu, wpu = stats.ttest_ind(gg, uu, equal_var=False)
            wtr, wpr = stats.ttest_ind(gg, rr, equal_var=False)
        else:
            wtu = wpu = wtr = wpr = float("nan")
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>9.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{su['mean_bps']:>8.1f} {s['mean_bps']-su['mean_bps']:>7.1f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>7.1f} "
              f"{wtu:>7.2f} {wpu:>6.3f} {wtr:>7.2f} {wpr:>6.3f}")
    print(f"  total gated entries (H={st.HORIZONS[0]}): {n_gated_total}   "
          f"ungated: {n_ungated_total}")

    print("\n# Per-ticker, H=20: gated one-sample t, gate-minus-ungated, gate-minus-random")
    for t in data.DEFAULT_TICKERS:
        c = _load(t)["close"]
        ge = st.gated_entries(c, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK)
        ue = st.momentum_entries(c, mom_n=MOM_N)
        re = st.random_entries(c, max(len(ge), 50), warmup=LOOKBACK, seed=7)
        s = st.summarize(st.forward_returns(c, ge, 20))
        su = st.summarize(st.forward_returns(c, ue, 20))
        sr = st.summarize(st.forward_returns(c, re, 20))
        print(f"  {t}: nG={len(ge):>3} gate={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"Δung={s['mean_bps']-su['mean_bps']:+7.1f} Δrnd={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Shuffled-gate placebo (SPY, H=20, 500 draws)")
    pl = st.shuffled_gate_placebo(_load("SPY")["close"], 20, mom_n=MOM_N, vhf_n=VHF_N,
                                  q=Q, lookback=LOOKBACK, n_draws=500, seed=484)
    print(f"  observed gated {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => gate timing not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0: gate must NOT beat ungated; planted regime momentum: gate must beat ungated.")
for edge in (0.0, 0.45):
    px, _ = data.synthetic_panel(edge=edge, seed=484, n_days=8000)
    c = px["close"]
    ge = st.gated_entries(c, mom_n=MOM_N, vhf_n=VHF_N, q=Q, lookback=LOOKBACK)
    ue = st.momentum_entries(c, mom_n=MOM_N)
    sg = st.summarize(st.forward_returns(c, ge, 20))
    su = st.summarize(st.forward_returns(c, ue, 20))
    print(f"  edge={edge:.2f}: nG={sg['n']:>4} gate={sg['mean_bps']:+7.1f}bps win={sg['win']*100:.0f}% "
          f"t={sg['t']:+.2f} | ungated={su['mean_bps']:+7.1f}bps  Δgate={sg['mean_bps']-su['mean_bps']:+7.1f}")
