"""Reproducible headline run for Study 427 — Rate of Change (ROC).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the cached SPY daily tape under ``_cache/``
(the real-tape numbers, pinned to the last complete month) and always runs the synthetic
positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rate_of_change import data, strategy as st

W = 126           # headline ROC lookback (days)
MODE = "long_flat"
COST = 1.0        # one-way bps
PIN = "2026-05-31"  # last complete month

print("# Rate of Change — ROC long/flat timing on SPY (yfinance, total-return closes)")
if data.have_real("SPY"):
    bars = data.load_real("SPY").loc[:PIN]
    close = bars["close"]
    span = (bars.index.max() - bars.index.min()).days / 365.25
    print(f"tape           : SPY {bars.index.min().date()} -> {bars.index.max().date()}  "
          f"({len(bars):,} bars, {span:.1f} calendar years)")
    print(f"fingerprint    : {data.fingerprint(bars)}")

    roc = st.summarize(st.book_returns(close, st.roc_signal(close, W, MODE), COST))
    hold = st.summarize(st.buy_and_hold(close))
    diff = st.diff_vs_hold(close, W, MODE, COST)
    plac = st.permutation_pvalue(close, W, MODE, COST, n_draws=5000)
    be = st.breakeven_cost(close, W, MODE)

    print(f"\n# ROC({W}) long/flat vs buy & hold (1 bp one-way, excess-vs-excess Sharpe, 1-day lag)")
    print(f"  {'rule':>10} {'sharpe':>7} {'cagr':>7} {'maxdd':>7} {'t_exc':>6} {'turnover':>9} {'expo':>5}")
    print(f"  {'ROC':>10} {roc['sharpe']:>7.3f} {roc['cagr']*100:>6.1f}% {roc['maxdd']*100:>6.0f}% "
          f"{roc['t_exc']:>6.2f} {roc['ann_turnover']:>9.1f} {roc['exposure']:>5.2f}")
    print(f"  {'Buy&Hold':>10} {hold['sharpe']:>7.3f} {hold['cagr']*100:>6.1f}% {hold['maxdd']*100:>6.0f}% "
          f"{hold['t_exc']:>6.2f} {hold['ann_turnover']:>9.1f} {hold['exposure']:>5.2f}")

    print(f"\n# The decisive test — ROC minus buy & hold (HAC t on the net difference)")
    print(f"  delta vs hold  : {diff['delta_ann']*100:+.2f}%/yr   HAC t = {diff['t_diff']:+.2f}")
    print(f"  break-even cost: {be:.1f} bps one-way (ROC's zero-cost CAGR already below hold => 0)")

    print(f"\n# Permutation placebo — circular shifts of the position vector (5,000 draws)")
    print(f"  observed Sharpe {plac['obs']:.3f}   p = {plac['p_value']:.3f}")

    print(f"\n# Long/short variant (short when ROC < 0)")
    ls = st.summarize(st.book_returns(close, st.roc_signal(close, W, 'long_short'), COST))
    lsd = st.diff_vs_hold(close, W, 'long_short', COST)
    print(f"  sharpe={ls['sharpe']:.3f}  cagr={ls['cagr']*100:.1f}%  maxdd={ls['maxdd']*100:.0f}%  "
          f"delta={lsd['delta_ann']*100:+.2f}%/yr  t={lsd['t_diff']:+.2f}")

    print(f"\n# Window robustness — difference vs hold by ROC lookback")
    for w in (21, 63, 126, 252):
        s = st.summarize(st.book_returns(close, st.roc_signal(close, w, MODE), COST))
        d = st.diff_vs_hold(close, w, MODE, COST)
        print(f"  ROC({w:>3}): sharpe={s['sharpe']:.3f}  cagr={s['cagr']*100:.1f}%  "
              f"delta={d['delta_ann']*100:+.2f}%/yr  t={d['t_diff']:+.2f}")

    print(f"\n# Cost sweep — net excess Sharpe / CAGR by one-way bps")
    for _, row in st.cost_sweep(close, W, MODE).iterrows():
        print(f"  {row['cost_bps']:>4.0f} bps: sharpe={row['sharpe']:.3f}  cagr={row['cagr']*100:.2f}%")

    print(f"\n# The rivals race — same lag/costs/rf (excess Sharpe)")
    race = st.benchmark_race(close, W, MODE, COST)
    for name, r in race.iterrows():
        print(f"  {name:>14}: sharpe={r['sharpe']:.3f}  cagr={r['cagr']*100:>5.1f}%  "
              f"maxdd={r['maxdd']*100:>4.0f}%  t={r['t_exc']:>5.2f}  turn={r['ann_turnover']:>5.1f}")
else:
    print("(no _cache/bars_SPY_1d.parquet — run data.load_real('SPY') once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (regime-switching tape)")
print("  edge off -> ROC must NOT beat hold (difference t ~ 0); edge on -> ROC must light up.")
for edge in (0.0, 0.5):
    px, _ = data.synthetic_panel(edge=edge, seed=427)
    sr = st.summarize(st.book_returns(px["close"], st.roc_signal(px["close"], 63, MODE), COST))
    hh = st.summarize(st.buy_and_hold(px["close"]))
    d = st.diff_vs_hold(px["close"], 63, MODE, COST)
    print(f"  edge={edge:+.1f}: roc_sharpe={sr['sharpe']:+.2f}  hold_sharpe={hh['sharpe']:+.2f}  "
          f"delta={d['delta_ann']*100:+.1f}%/yr  t_diff={d['t_diff']:+.2f}")
