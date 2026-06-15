"""Real-tape verification — Study 177 (Megacap-Concentration). Regenerates docs/results.md numbers.

Loads (or fetches) the EDGAR shares × yfinance cross-section, builds annual market-cap
rankings (using *prior year-end* market cap — no look-ahead), and computes the honest
verdict: does concentrating in the top-10 S&P 500 names by market cap beat the
equal-weight baseline and SPY?

    python studies/177-megacap-concentration/examples/verify.py            # cache-only
    python studies/177-megacap-concentration/examples/verify.py --fetch    # refresh data
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from megacap_concentration import data, strategy as st  # noqa: E402

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "_cache"))


def main(fetch: bool) -> None:
    # ------------------------------------------------------------------
    # 1. Load the cross-section
    # ------------------------------------------------------------------
    print("=== Study 177 — Megacap-Concentration — real tape ===\n")
    panel, meta = data.load_real(top_n=10, fetch=fetch, cache_dir=CACHE_DIR)
    print(f"Universe: {meta['n_tickers_total']} tickers, {meta['n_years']} years "
          f"({meta['years'][0]}–{meta['years'][-1]})")
    print(f"Fingerprint: {data.fingerprint(panel)}")
    print(f"NOTE: Survivorship bias: current S&P 500 name list only.")
    print()

    # Exclude the current partial year (2026)
    panel = panel[panel["year"] < 2026].copy()
    n_full = int(panel["year"].nunique())
    print(f"Full sample used: {n_full} complete calendar years (excl. partial 2026)\n")

    # ------------------------------------------------------------------
    # 2. Portfolio returns
    # ------------------------------------------------------------------
    spy = st.spy_annual_returns(cache_dir=REPO_CACHE)

    for top_n in (10, 7):
        # Re-rank panel for this top_n
        rows = []
        import numpy as np
        import pandas as _pd
        mcap_path = os.path.join(CACHE_DIR, "megacap_annual_mcap.parquet")
        ret_path  = os.path.join(CACHE_DIR, "megacap_annual_returns.parquet")
        mcap = _pd.read_parquet(mcap_path)
        rets = _pd.read_parquet(ret_path)
        mcap_lagged = mcap.shift(1)
        for yr in sorted(rets["year"].unique()):
            if yr >= 2026:
                continue
            if yr not in mcap_lagged.index:
                continue
            mcap_row = mcap_lagged.loc[yr].dropna()
            if len(mcap_row) < top_n:
                continue
            ranks = mcap_row.rank(ascending=False, method="first").astype(int)
            year_rets = rets[rets["year"] == yr].set_index("ticker")["ret_annual"]
            common = ranks.index.intersection(year_rets.index)
            if len(common) < top_n:
                continue
            for tk in common:
                rows.append({
                    "year": yr, "ticker": tk,
                    "mcap_rank": int(ranks[tk]),
                    "ret_annual": float(year_rets[tk]),
                    "is_top_n": bool(ranks[tk] <= top_n),
                })
        p = _pd.DataFrame(rows)
        annual = st.annual_portfolio_returns(p, top_n=top_n, col="ret_annual")
        enriched = st.with_spy_benchmark(annual, spy)
        result = st.summarize(enriched)

        print(f"=== TOP-{top_n} portfolio ===")
        f = result["full"]
        print(f"  Full ({f['n']} yrs): top-{top_n} {f['topn_mean']:+.1f}%  EW {f['ew_mean']:+.1f}%  "
              f"excess {f['excess_mean']:+.1f}%  (t={f['excess_tstat']:+.2f})")
        for name, sub in result["by_decade"].items():
            if sub["n"] > 0:
                print(f"  {name:12s} ({sub['n']} yrs): top-{top_n} {sub['topn_mean']:+.1f}%  "
                      f"EW {sub['ew_mean']:+.1f}%  excess {sub['excess_mean']:+.1f}%  "
                      f"(t={sub['excess_tstat']:+.2f})")
        vs = result.get("vs_spy", {})
        if vs:
            print(f"  vs SPY: top-{top_n} excess {vs['mean']:+.1f}% (t={vs['tstat']:+.2f})")
        print()

    # ------------------------------------------------------------------
    # 3. Show top-10 each year
    # ------------------------------------------------------------------
    print("=== Who were the top-10? (lagged ranking) ===")
    p10, _ = data.load_real(top_n=10, fetch=False, cache_dir=CACHE_DIR)
    p10 = p10[p10["year"] < 2026]
    for yr in sorted(p10["year"].unique()):
        top10 = p10[(p10["year"] == yr) & (p10["mcap_rank"] <= 10)].sort_values("mcap_rank")
        tickers = list(top10["ticker"])
        ret = p10[(p10["year"] == yr) & (p10["mcap_rank"] <= 10)]["ret_annual"].mean()
        print(f"  {yr}: {tickers[:7]}...  top-10 avg ret={ret:+.1%}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
