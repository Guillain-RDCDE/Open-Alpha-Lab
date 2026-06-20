"""Reproduce the pinned headline run for Study 332 (Downside-Beta).

Reads the cached real daily-return panel under ../_cache/ (cache-first, no network),
drops the in-progress month, runs the β⁻ / β / β⁻−β quintile sorts plus the cost sweep,
and prints the numbers + fingerprints that docs/results.md is pinned to.

    python examples/verify.py            # uses the cached real tape
    python examples/verify.py --fetch    # refresh the cache from yfinance first

If no cache exists and --fetch is not passed, the synthetic positive/null control is run
instead (a machinery proof — never market evidence).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from downside_beta import data, strategy as st  # noqa: E402


def run_real(fetch: bool) -> None:
    rdf, mkt = data.load_real(fetch=fetch)
    last = rdf.index[-1]
    cut = pd.Timestamp(last.year, last.month, 1) - pd.Timedelta(days=1)
    rdf, mkt = rdf.loc[:cut], mkt.loc[:cut]
    print(f"REAL tape: {rdf.shape[0]} days x {rdf.shape[1]} tickers, "
          f"{rdf.index[0].date()} -> {rdf.index[-1].date()}")
    print(f"  fingerprint(panel)  = {data.fingerprint(rdf)}")
    print(f"  fingerprint(market) = {data.fingerprint(mkt)}\n")

    for col in ("beta_dn", "beta", "rel_dbeta"):
        res = st.monthly_quantile_returns(rdf, mkt, signal_col=col, min_stocks=8)
        s = st.summarize(res["spread"])
        lo, hi = st.block_bootstrap_ci(res["spread"], n_boot=2000)
        print(f"{col:10s} n={s['n']:3d}  spread={s['mean_ann']*100:6.2f}%/yr  "
              f"HAC t={s['tstat']:+.2f}  SR={s['sharpe_ann']:+.2f}  hit={s['hit_rate']:.2f}  "
              f"CI[{lo*1e4:+.0f},{hi*1e4:+.0f}]bps/mo")

    print("\nCost sweep on the β⁻ long-short (one-way bps -> net):")
    res = st.monthly_quantile_returns(rdf, mkt, signal_col="beta_dn", min_stocks=8)
    for c in (0, 5, 10, 20):
        ns = st.summarize(st.net_spread(res, one_way_bps=c))
        print(f"  {c:2d} bps  net={ns['mean_ann']*100:6.2f}%/yr  HAC t={ns['tstat']:+.2f}")


def run_synthetic() -> None:
    print("No real cache (run with --fetch to populate). Synthetic positive/null control:\n")
    for prem, label in ((0.0, "null"), (0.0025, "priced")):
        rdf, mkt, _ = data.synthetic_panel(downside_premium=prem, seed=332)
        res = st.monthly_quantile_returns(rdf, mkt, signal_col="beta_dn")
        s = st.summarize(res["spread"])
        print(f"  {label:7s} spread={s['mean_ann']*100:6.2f}%/yr  HAC t={s['tstat']:+.2f}")


if __name__ == "__main__":
    fetch = "--fetch" in sys.argv
    if fetch or (os.path.exists(data.PANEL_CACHE) and os.path.exists(data.MARKET_CACHE)):
        run_real(fetch=fetch)
    else:
        run_synthetic()
