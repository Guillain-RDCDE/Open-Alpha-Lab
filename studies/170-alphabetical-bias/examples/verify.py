"""Real-tape verification — Study 170 (Alphabetical-Bias).

Loads (or fetches) daily price data for the S&P 500 universe, builds a monthly
returns + volume panel, and runs the alphabetical-bias study: early-alphabet
(A–C) vs late-alphabet (D–Z) equal-weight return spread and volume comparison.
Also runs the per-letter Bonferroni scan.  Regenerates the numbers in
docs/results.md.

    python studies/170-alphabetical-bias/examples/verify.py             # cache-only
    python studies/170-alphabetical-bias/examples/verify.py --fetch     # refresh prices

Network is touched only with --fetch.  Cache-only mode reads the per-ticker
daily parquets from studies/170-alphabetical-bias/_cache/.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from alphabetical_bias import data, strategy as st  # noqa: E402

STUDY_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
START = "2000-01-01"


def main(fetch: bool) -> None:
    print("=== Study 170 — Alphabetical-Bias — real tape ===\n")

    # Load universe.
    print("Loading S&P 500 universe…")
    try:
        universe = data.load_sp500_names()
    except Exception as exc:
        print(f"  Warning: could not load universe ({exc}). Using hard-coded fallback sample.")
        universe = _fallback_universe()
    print(f"  {len(universe)} tickers loaded: "
          f"{universe['is_early'].sum()} early (A–C), "
          f"{(~universe['is_early']).sum()} late (D–Z)\n")

    if fetch:
        print("Fetching daily prices (this may take several minutes)…")
        n_fetched = 0
        for _, row in universe.iterrows():
            ticker = row["ticker"]
            try:
                data.fetch_daily(ticker, start=START, fetch=True, cache_dir=STUDY_CACHE)
                n_fetched += 1
                if n_fetched % 50 == 0:
                    print(f"  fetched {n_fetched}/{len(universe)}")
            except Exception as exc:
                pass  # ticker may be delisted or renamed
        print(f"  done ({n_fetched} tickers fetched)\n")

    print("Building monthly panel from cached prices…")
    panel = data.build_monthly_panel(universe, cache_dir=STUDY_CACHE, start=START)
    if panel.empty:
        print("  ERROR: no cached daily prices found. Run with --fetch first.")
        sys.exit(1)

    n_tickers = panel["ticker"].nunique()
    n_months = panel["date"].nunique()
    fp = data.fingerprint(panel)
    print(f"  {n_tickers} tickers × {n_months} months  |  fingerprint: {fp}")
    print(f"  date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"  early-alphabet obs: {panel['is_early'].sum():,}  "
          f"late-alphabet obs: {(~panel['is_early']).sum():,}\n")

    # --- Return channel ---
    print("=== Return channel (A–C minus D–Z monthly equal-weight spread) ===")
    monthly = st.monthly_ew_returns(panel)
    result = st.summarize_return_diff(monthly)
    print(f"  n_months        : {result.get('n', 'n/a')}")
    print(f"  mean spread     : {result.get('mean_bps', float('nan')):+.2f} bps/month")
    print(f"  HAC t-stat      : {result.get('tstat', float('nan')):+.2f}")
    print(f"  ann. Sharpe     : {result.get('sharpe_ann', float('nan')):+.3f}")

    # --- Volume channel ---
    print("\n=== Volume channel (log-volume ratio, A–C vs D–Z) ===")
    vcmp = st.volume_comparison(panel)
    print(f"  mean vol early  : {vcmp.get('mean_vol_early', float('nan')):.0f}")
    print(f"  mean vol late   : {vcmp.get('mean_vol_late', float('nan')):.0f}")
    print(f"  volume ratio    : {vcmp.get('vol_ratio', float('nan')):.3f}")
    print(f"  HAC t-stat      : {vcmp.get('tstat', float('nan')):+.2f}")

    # --- Per-letter Bonferroni scan ---
    print(f"\n=== Per-letter Bonferroni scan (threshold |t| >= {st.BONFERRONI_T:.2f}) ===")
    per_letter = st.per_letter_returns(panel)
    top5 = per_letter.dropna().reindex(
        per_letter["tstat"].abs().sort_values(ascending=False).index
    ).head(5)
    print("  Top-5 letters by |t-stat|:")
    for letter, row in top5.iterrows():
        sig = " **BONFERRONI SIG**" if row["bonferroni_significant"] else ""
        print(f"    {letter}: mean={row['mean_bps']:+.1f} bps/mo  t={row['tstat']:+.2f}{sig}")
    n_sig = int(per_letter["bonferroni_significant"].sum())
    print(f"  Letters significant post-Bonferroni: {n_sig} (expected under null: ~0)")

    print("\n=== VERDICT ===")
    t_ret = result.get("tstat", float("nan"))
    t_vol = vcmp.get("tstat", float("nan"))
    if abs(t_ret) >= 2.0:
        print(f"  Return channel: t={t_ret:+.2f} — POSSIBLE SIGNAL (but check survivorship!)")
    else:
        print(f"  Return channel: t={t_ret:+.2f} — no statistically significant return edge")
    if abs(t_vol) >= 2.0:
        print(f"  Volume channel: t={t_vol:+.2f} — attention effect REAL (higher volume for A–C)")
    else:
        print(f"  Volume channel: t={t_vol:+.2f} — no robust volume difference")
    print("  Survivorship bias: S&P 500 universe is not static — named and noted.")


def _fallback_universe() -> pd.DataFrame:
    """A small hardcoded sample if the EDGAR universe is unavailable."""
    import string
    tickers_by_letter = {
        "A": ["AAPL", "AMZN", "ABBV", "AMD", "AMAT"],
        "B": ["BAC", "BLK", "BA", "BMY"],
        "C": ["CAT", "CVX", "CSCO", "CL"],
        "D": ["DHR", "DD", "DE"],
        "E": ["EMR", "ETN"],
        "G": ["GE", "GOOGL", "GS"],
        "H": ["HD", "HON"],
        "I": ["IBM", "INTC"],
        "J": ["JNJ", "JPM"],
        "K": ["KO"],
        "L": ["LMT", "LOW"],
        "M": ["MMM", "MRK", "MSFT", "META"],
        "N": ["NFLX", "NKE"],
        "P": ["PG", "PFE"],
        "S": ["SBUX"],
        "T": ["TSLA", "TXN"],
        "U": ["UNH", "UPS"],
        "V": ["V", "VZ"],
        "W": ["WMT"],
        "X": ["XOM"],
    }
    records = []
    for letter, tickers in tickers_by_letter.items():
        for t in tickers:
            records.append({
                "ticker": t,
                "letter": letter,
                "is_early": letter in data.EARLY_LETTERS,
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
