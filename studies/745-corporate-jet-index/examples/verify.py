"""Reproducible headline run for Study 745 — Corporate-Jet-Index.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached monthly closes under
``_cache/`` if present (the real-tape long/short), and always runs the synthetic
positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from corporate_jet_index import data, strategy as st

MEGATECH = ("TSLA", "META", "GOOGL", "ORCL")


def _cagr(series):
    s = series.dropna()
    return float((1.0 + s).prod() ** (12.0 / len(s)) - 1.0)


print("# Corporate-Jet-Index — governance long/short (long frugal, short jet-loving CEOs)")
print(f"# perk table: {len(data.HEAVY)} heavy / {len(data.LOW)} low  "
      f"(fingerprint {data.fingerprint(data.JET_FIRMS)})")

if data.have_real():
    prices, firms = data.load_real()
    panel = st.long_short_panel(prices, firms, min_names=4)
    s = st.summarize(panel)
    print(f"\n# Panel: {s['n_months']} months  {s['start']} -> {s['end']}")

    print("\n# Signal — long/short (low - heavy), excess of market")
    print(f"  LS mean        : {s['ls_mean_ann']*100:+.2f}%/yr  "
          f"(monthly {s['ls_mean_month']*100:+.3f}%)")
    print(f"  HAC (NW) t     : {s['ls_hac_t']:+.2f}  (lags {s['ls_hac_lags']})")
    print(f"  Sharpe (ann)   : {s['ls_sharpe']:.2f}")
    print(f"  heavy excess   : {s['heavy_x_ann']*100:+.2f}%/yr   "
          f"low excess: {s['low_x_ann']*100:+.2f}%/yr")

    print("\n# Alpha vs beta — the only significant number is a beta artifact")
    print(f"  market-model alpha : {s['alpha_ann']*100:+.2f}%/yr  "
          f"HAC t = {s['alpha_t']:+.2f}")
    print(f"  long/short beta    : {s['beta']:+.2f}")
    rets = st.monthly_returns(prices)
    mkt = rets["SPY"].reindex(panel.index)
    for nm, b in [("heavy", panel["heavy"]), ("low", panel["low"])]:
        X = np.column_stack([np.ones(len(mkt)), mkt.to_numpy()])
        bb = np.linalg.lstsq(X, b.to_numpy(), rcond=None)[0]
        print(f"  {nm:>5} basket beta : {bb[1]:+.2f}")

    print("\n# Raw compounding (total return, panel window)")
    hb = st.basket_returns(prices, firms, heavy=True).reindex(panel.index)
    lb = st.basket_returns(prices, firms, heavy=False).reindex(panel.index)
    print(f"  low-perk  CAGR : {_cagr(lb)*100:+.2f}%/yr")
    print(f"  SPY       CAGR : {_cagr(mkt)*100:+.2f}%/yr")
    print(f"  heavy-perk CAGR: {_cagr(hb)*100:+.2f}%/yr")

    print("\n# Founder-growth confound — drop the mega-cap flyers (TSLA/META/GOOGL/ORCL)")
    firms2 = [f for f in firms if f["ticker"] not in MEGATECH]
    s2 = st.summarize(st.long_short_panel(prices, firms2, min_names=3))
    print(f"  heavy excess : {s2['heavy_x_ann']*100:+.2f}%/yr  (echoes Yermack's -4%)")
    print(f"  LS mean      : {s2['ls_mean_ann']*100:+.2f}%/yr  HAC t = {s2['ls_hac_t']:+.2f}")
    print(f"  alpha        : {s2['alpha_ann']*100:+.2f}%/yr   HAC t = {s2['alpha_t']:+.2f}  "
          f"(still a beta artifact)")

    print("\n# Tradable — net of costs + short borrow")
    nc = st.net_of_costs(panel["ls"])
    print(f"  gross {nc['gross_ann']*100:+.2f}%/yr  - costs {nc['rebal_cost_ann']*100:.2f} "
          f"- borrow {nc['borrow_ann']*100:.2f}  = net {nc['net_ann']*100:+.2f}%/yr")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED heavy-basket discount and must NOT manufacture")
print("  significance from 186 months when the true edge is 0.")
for edge in (0.0, -80.0):
    syn = data.synthetic_panel(alpha_bps_month=edge, seed=745)
    h = st.hac_tstat(syn["ls"])
    print(f"  planted {edge:+6.1f} bps/mo: LS {st.annualize_mean(h['mean'])*100:+.2f}%/yr  "
          f"HAC t = {h['t']:+.2f}")
