"""Real-data verification — Study 274 (Fine-Wine). Regenerates docs/results.md numbers.

Loads the Liv-ex 100 year-end levels (hardcoded in data.py), joins with S&P 500
calendar-year price returns from the repo-level ^GSPC cache, and runs the full
diversification analysis: raw and de-smoothed correlation, risk/return, and the
mean-variance benefit test (gross/net, raw/de-smoothed) with HAC t-stats.

Cache-only (no network): the ^GSPC parquet ships with the repo. Run from anywhere:

    python studies/274-fine-wine/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fine_wine import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 274 -- Fine-Wine: does the Liv-ex 100 diversify or just lag?\n")
    print("Data: Liv-ex 100 year-end levels (hardcoded, 2003-2025)")
    print("      S&P 500: repo-level _cache/^GSPC_split_only.parquet (Dec/Dec)\n")

    df = data.fetch_sp500_annual()
    fp = data.fingerprint(df)
    print(f"Data stamp: years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)}, fingerprint={fp}\n")

    s = st.summarize(df)

    print("=== Risk and return (price-only) ===")
    print(f"  Wine CAGR:  {s['wine_cagr_pct']:+.1f}%   vol {s['wine_vol_pct']:.1f}% "
          f"(de-smoothed {s['wine_vol_desmoothed_pct']:.1f}%)   Sharpe {s['wine_sharpe']:.2f}")
    print(f"  Equity CAGR:{s['eq_cagr_pct']:+.1f}%   vol {s['eq_vol_pct']:.1f}%"
          f"                       Sharpe {s['eq_sharpe']:.2f}")
    print(f"  Wine lags equities by {s['eq_cagr_pct']-s['wine_cagr_pct']:+.1f}pp/yr (before frictions)\n")

    print("=== Correlation ===")
    print(f"  Raw correlation:        {s['raw_corr']:+.3f}")
    print(f"  Wine AR(1) (smoothing): {s['wine_ar1']:+.3f}")
    print(f"  De-smoothed correlation:{s['desmoothed_corr']:+.3f}  <- the honest number\n")

    print("=== Diversification benefit (20% wine sleeve, rf 2%) ===")
    db = st.diversification_benefit(df, wine_weight=0.20)
    print(f"  Equity-only Sharpe:           {db['eq_only_sharpe']:.4f}")
    for lens in ["gross_raw", "net_raw", "gross_desmoothed", "net_desmoothed"]:
        print(f"  Blend {lens:18s} Sharpe {db['blend_sharpe_'+lens]:.4f}  "
              f"delta {db['sharpe_delta_'+lens]:+.4f}  "
              f"HAC t {db['benefit_hac_t_'+lens]:+.3f}")
    print()

    print("=== Frontier (Sharpe-maximising wine weight) ===")
    f = st.two_asset_frontier(df["wine_return"].values, df["sp500_return"].values)
    best = f.loc[f["sharpe"].idxmax()]
    print(f"  Max-Sharpe wine weight (raw, gross): {best['wine_weight']:.0%} "
          f"-> Sharpe {best['sharpe']:.3f} (vs equity-only {f['sharpe'].iloc[0]:.3f})\n")

    print("Verdict: Signal=WEAK (low raw corr +0.10 but de-smooths to +0.28; "
          "no benefit HAC t reaches +2),")
    print("         Tradability=MIRAGE (wine lags ~3.9pp/yr; ~10-15% spreads + "
          "storage; net-of-cost sleeve lowers Sharpe).")


if __name__ == "__main__":
    main()
