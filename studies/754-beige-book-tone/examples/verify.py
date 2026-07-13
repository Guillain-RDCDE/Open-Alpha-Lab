"""Reproducible headline run for Study 754 — Beige-Book-Tone.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the labelled LM-tone proxy + real
release calendar (always available) and the cached daily SPY under ``_cache/`` if present;
always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from beige_book_tone import data, strategy as st

try:
    from quantlab import repro
except Exception:                                    # pragma: no cover
    repro = None

print("# Beige-Book-Tone — LM net-tone PROXY (labelled) on the real release calendar + SPY (yfinance)")

if data.have_real():
    rel, spy = data.load_real()
    yrs = (rel.index.max() - rel.index.min()).days / 365.25
    if repro is not None:
        print(repro.data_stamp("SPY_daily", spy.to_frame("SPY"), cols=["SPY"], asof="2024-12-31"))
        print(repro.data_stamp("BeigeBook_releases", rel, cols=["tone"], asof="2024-12-31"))
    print(f"events         : {len(rel)}  ({rel.index.min().date()} -> {rel.index.max().date()}, "
          f"{yrs:.1f} years)  |  positive-tone: {int((rel['tone'] > 0).sum())}")
    print("signal         : POSITIVE = LM net-tone proxy > 0; drift from the release-day close (no look-ahead)")

    print("\n# Forward SPY returns after a POSITIVE-tone Beige Book vs the base rate")
    print(f"  {'h':>3} {'n_pos':>5} {'pos_mean':>9} {'neg_mean':>9} {'base_mean':>9} "
          f"{'pos_up':>6} {'base_up':>7} {'Welch_t':>8} {'p_plac':>7}")
    for h in st.HORIZONS:
        s = st.summarize(rel, spy, h)
        print(f"  {str(h)+'d':>3} {s['n_pos']:>5} {s['pos_mean']*100:>8.3f}% "
              f"{s['neg_mean']*100:>8.3f}% {s['base_mean']*100:>8.3f}% "
              f"{s['pos_uprate']*100:>5.0f}% {s['base_uprate']*100:>6.0f}% "
              f"{s['t']:>8.2f} {s['p_placebo']:>7.3f}")

    print("\n# Continuous tone->drift regression  r = a + b*tone  (Newey-West HAC t, 4 lags)")
    for h in st.HORIZONS:
        r = st.tone_drift_regression(rel, spy, h)
        print(f"  {str(h)+'d':>3}: beta={r['beta']*100:>7.4f}%/unit  t_ols={r['t_ols']:>6.2f}  "
              f"t_hac={r['t_hac']:>6.2f}  corr={r['corr']:>6.3f}")

    print("\n# Tradability — long the 5-day post-release window on a POSITIVE book (1 bp one-way)")
    o = st.event_overlay(rel, spy, h=5, cost_bps=1.0)
    print(f"  per-event : gross={o['per_event_gross']*100:>6.3f}%  net={o['per_event_net']*100:>6.3f}%  "
          f"base={o['base_event']*100:>6.3f}%  event-Sharpe={o['event_sharpe']:.2f}")
    print(f"  annualised: overlay net={o['ann_net']*100:>5.1f}%/yr  vs buy&hold={o['bh_ann']*100:>5.1f}%/yr  "
          f"({o['n_trades']} trades, {o['trades_per_yr']:.1f}/yr)")

    print("\n# Robustness — threshold, and ex-COVID (5-day)")
    med = float(rel["tone"].median())
    for lbl, thr in (("thr>0", 0.0), (f"thr>med({med:.2f})", med)):
        s = st.summarize(rel, spy, 5, thresh=thr)
        print(f"  {lbl:>12}: n_pos={s['n_pos']:>3}  pos={s['pos_mean']*100:>6.3f}%  "
              f"base={s['base_mean']*100:>6.3f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    rel2 = rel[(rel.index < "2020-01-01") | (rel.index >= "2021-01-01")]
    s = st.summarize(rel2, spy, 5)
    print(f"  {'ex-2020':>12}: n_pos={s['n_pos']:>3}  pos={s['pos_mean']*100:>6.3f}%  "
          f"base={s['base_mean']*100:>6.3f}%  t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")

    # Is the drift just beta? tone vs the PRIOR 5-day tape (regime clustering, not a lead)
    pos = spy.index.searchsorted(rel.index, side="left")
    pre = np.full(len(rel), np.nan)
    for j, i in enumerate(pos):
        if 0 <= i - 5 and i < len(spy):
            pre[j] = spy.values[i] / spy.values[i - 5] - 1.0
    ok = ~np.isnan(pre)
    print(f"\n# Regime check: corr(tone, PRIOR 5-day SPY return) = "
          f"{np.corrcoef(rel['tone'].values[ok], pre[ok])[0,1]:+.3f}  "
          f"(positive-tone books cluster in expansions, not a lead)")
else:
    print("(no _cache/spy.csv — run data.fetch_spy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must recover a PLANTED tone->drift link and must NOT manufacture")
print("  significance from a no-link series (edge=0).")
for edge in (0.0, 0.004):
    srel, sspy = data.synthetic(n_years=14, edge=edge, seed=754)
    s = st.summarize(srel, sspy, 5)
    r = st.tone_drift_regression(srel, sspy, 5)
    print(f"  planted edge={edge:.3f}: n_pos={s['n_pos']:>3}  pos5d={s['pos_mean']*100:>6.3f}%  "
          f"base5d={s['base_mean']*100:>6.3f}%  Welch_t={s['t']:>6.2f}  "
          f"beta={r['beta']*100:>6.3f}  t_hac={r['t_hac']:>6.2f}")
