"""Reproducible headline run for Study 623 — IPO Long-Run Underperformance.

Prints EVERY number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; Track A (Ritter's published cohort table) is
hardcoded and always available offline; Track B (the live ETF tape) reads the cache under
``_cache/`` (fetched once with ``data.fetch_tape()``); the synthetic control runs anywhere
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from ipo_long_run_underperformance import data, strategy as st

LAGS_TAPE = 6     # Newey-West lags, monthly ETF series
LAGS_COHORT = 3   # Newey-West lags, annual cohorts holding overlapping 3-year windows

print("# IPO Long-Run Underperformance — Ritter's drift: published cohorts + the live ETF tape")
print(f"as-of {data.AS_OF} (last complete month; the partial current month is dropped)")

# --------------------------------------------------------------------------- #
# Track A — Ritter Table 19 (published cohort record, hardcoded from the PDF)
# --------------------------------------------------------------------------- #
print("\n# Track A — Ritter Table 19 (published; 45 IPO cohort years 1980-2024, 9,253 IPOs)")
print("  avg 3-year buy-and-hold returns from the first close; NW t across cohort years "
      f"(lag={LAGS_COHORT}, overlapping windows)")
co = data.ritter_cohorts()
for col, label in (("bhr3_raw", "raw (IPOs, absolute)"),
                   ("bhr3_mkt_adj", "market-adjusted (vs CRSP VW)"),
                   ("bhr3_style_adj", "style-adjusted (vs size+B/M match)")):
    s = st.cohort_stats(co, col, lags=LAGS_COHORT)
    print(f"  {label:36s}: mean {s['mean_pct']:+7.2f}%/3yr  NW t = {s['t_nw']:+.2f}  "
          f"(N-weighted {s['weighted_mean_pct']:+.2f}%, {s['share_negative']*100:.1f}% of "
          f"cohorts negative, n={s['n_cohorts']})")
co2 = data.ritter_cohorts(drop_truncated=True)
for col, label in (("bhr3_mkt_adj", "market-adjusted"), ("bhr3_style_adj", "style-adjusted")):
    s = st.cohort_stats(co2, col, lags=LAGS_COHORT)
    print(f"  ex-truncated 2023-24 {label:15s}: mean {s['mean_pct']:+7.2f}%/3yr  "
          f"NW t = {s['t_nw']:+.2f}  (n={s['n_cohorts']})")
agg = data.RITTER_AGGREGATE
print(f"  Ritter's own aggregate row: {agg['n_ipos']:,} IPOs, first-day {agg['first_day']:+.1f}%, "
      f"3-yr raw {agg['bhr3_raw']:+.1f}%, mkt-adj {agg['bhr3_mkt_adj']:+.1f}%, "
      f"style-adj {agg['bhr3_style_adj']:+.1f}%")
print("  (published data re-presented — carries the literature story, never the REAL stamp)")

# --------------------------------------------------------------------------- #
# Track B — the live tape (the judge)
# --------------------------------------------------------------------------- #
if data.have_real():
    tape = data.load_tape()
    panel = data.monthly_panel(tape)
    try:
        from quantlab import repro
        fp = repro.fingerprint(panel.round(8))
    except Exception:
        fp = "n/a"
    ny = len(panel) / 12.0
    print(f"\n# Track B — the live tape: Renaissance IPO ETF vs SPY / IWM / IWO (yfinance, cached)")
    print(f"  monthly total returns (adjusted closes, net of ETF fees), "
          f"{panel.index.min().date()} -> {panel.index.max().date()} "
          f"({len(panel)} months, {ny:.1f} years)  fingerprint {fp}")

    print("\n  growth of $1 (total return):")
    for c in ("IPO", "SPY", "IWM", "IWO"):
        g = st.cumulative_growth(panel, c)
        print(f"    {c:4s}: {g:5.2f}x  (CAGR {((g ** (1/ny)) - 1) * 100:+.2f}%/yr)")
    rel = (np.exp(np.log1p(panel["IPO"]).cumsum())
           / np.exp(np.log1p(panel["SPY"]).cumsum()))
    dd = rel / rel.cummax() - 1.0
    print(f"    IPO/SPY relative line: peak {rel.max():.2f}x ({rel.idxmax().date()}), "
          f"ends {rel.iloc[-1]:.2f}x, max relative drawdown {dd.min()*100:.1f}%")
    print(f"    share of months IPO beats SPY: {(panel['IPO'] > panel['SPY']).mean()*100:.1f}%")

    print(f"\n  long-short spread + CAPM-style alpha (HAC/Newey-West, lag={LAGS_TAPE}):")
    for b in ("SPY", "IWM", "IWO"):
        sp = st.spread_stats(panel, b, lags=LAGS_TAPE)
        al = st.alpha_stats(panel, b, lags=LAGS_TAPE)
        print(f"    vs {b:3s}: spread {sp['mean_bps']:+7.2f} bps/mo ({sp['ann_pct']:+.2f}%/yr) "
              f"t = {sp['t_nw']:+.2f} | alpha {al['alpha_bps']:+7.2f} bps/mo "
              f"({al['alpha_ann_pct']:+.2f}%/yr) t = {al['alpha_t']:+.2f}  beta = {al['beta']:.3f}")
    tf = st.two_factor_alpha(panel, lags=LAGS_TAPE)
    print(f"    two-factor (SPY + small-growth spread IWO-SPY): alpha {tf['alpha_bps']:+.2f} bps/mo "
          f"({tf['alpha_ann_pct']:+.2f}%/yr) t = {tf['alpha_t']:+.2f}  "
          f"beta_mkt = {tf['beta_mkt']:.3f}  beta_style = {tf['beta_style']:.3f}")

    print("\n# Tradability — SHORT the IPO ETF / LONG SPY (net of borrow + one-way costs x NAV)")
    print("  always-on $1/$1 spread, monthly re-hedge; calendar-known rule -> zero signal lag "
          "(documented); short leg pays borrow monthly")
    for cb, br in ((5.0, 1.0), (10.0, 1.0), (5.0, 2.0)):
        ov = st.short_overlay_stats(panel, "SPY", cost_bps=cb, borrow_ann_pct=br, lags=LAGS_TAPE)
        print(f"    cost={cb:>4.1f} bps, borrow={br:.0f}%/yr: gross {ov['gross_bps']:+.2f} bps/mo "
              f"(t={ov['gross_t']:+.2f}) -> net {ov['net_bps']:+.2f} bps/mo "
              f"({ov['net_ann_pct']:+.2f}%/yr, t={ov['net_t']:+.2f})  "
              f"max drawdown {ov['worst_drawdown_pct']:.1f}%")
else:
    print("\n(no _cache/ipo_tape.csv — run data.fetch_tape() once to build the cache)")

# --------------------------------------------------------------------------- #
# Synthetic control — machinery proof (never cited in support of a stamp)
# --------------------------------------------------------------------------- #
print("\n# Synthetic control — deterministic, no network, averaged over 20 seeds (desk law)")
print("  the HAC-alpha detector must NOT fire on a zero-edge world and must recover a planted drag")
for edge in (0.0, -0.005, -0.010):
    r = st.synthetic_control(edge, n_seeds=20, lags=LAGS_TAPE)
    print(f"  planted alpha {r['edge_bps']:+7.1f} bps/mo: measured {r['mean_alpha_bps']:+7.2f} bps/mo  "
          f"mean HAC t = {r['mean_t']:+.2f}  |t|>=2 rejection rate = {r['reject_rate']*100:.0f}% "
          f"({r['n_seeds']} seeds)")
print("  (power note: a Ritter-sized drag of -50 bps/mo is detected only ~45% of the time in a "
      "152-month window — the live tape is honest but modestly powered)")
