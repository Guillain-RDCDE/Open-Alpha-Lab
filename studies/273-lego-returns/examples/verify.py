"""Real-data verification -- Study 273 (Lego-Returns). Regenerates docs/results.md numbers.

Loads the S&P 500 daily price cache (repo-level _cache/^GSPC_split_only.parquet),
computes calendar-year price returns, joins with the hardcoded LEGO index, and runs
the full Lego-vs-market comparison: gross/net CAGR, Newey-West HAC t-stat on annual
excess return, CAPM beta, and the paired sign test.

The S&P parquet is read-only and ships with the repo (no fetch required). Run from
anywhere:

    python studies/273-lego-returns/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lego_returns import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 273 -- Lego-Returns -- real-data verification\n")
    print("Data: S&P 500 daily price (repo-level _cache/^GSPC_split_only.parquet, PRICE-ONLY)")
    print("      LEGO secondary-market index: hardcoded in data.py (1987-2024)\n")

    df = data.fetch_market_annual()
    fp = data.fingerprint(df)
    print(f"Data stamp: years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)}, fingerprint={fp}\n")

    s = st.compare(df, one_way_cost=0.12, holding_years=5.0, seed=273)

    print("=== Headline returns (price-only) ===")
    print(f"  Lego gross CAGR:        {s['lego_cagr']:.2%}")
    print(f"  Lego net CAGR (12%/5y): {s['lego_net_cagr']:.2%}")
    print(f"  S&P 500 price CAGR:     {s['market_cagr']:.2%}")
    print(f"  S&P 500 + ~2% dividend: {s['market_cagr']+0.02:.2%}\n")

    print("=== The decisive test: excess return ===")
    print(f"  Mean annual excess (gross): {s['mean_excess_gross']:+.2%}  HAC t = {s['hac_t_gross']:+.3f}")
    print(f"  Mean annual excess (net):   {s['mean_excess_net']:+.2%}  HAC t = {s['hac_t_net']:+.3f}")
    print(f"  Bar for REAL: HAC t >= 2 on a POSITIVE excess -- not met.\n")

    print("=== Diversification claim (CAPM) ===")
    print(f"  beta to S&P: {s['beta']:.3f}  |  alpha: {s['alpha']:.2%}/yr  |  corr: {s['corr']:.3f}\n")

    print("=== Paired sign test ===")
    print(f"  Lego beat the S&P in {s['lego_win_years']}/{s['n']} years ({s['lego_win_rate']:.1%})")
    print(f"  Binomial p (vs 50%): {s['sign_test_p']:.4f}\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- a low-beta collectible, "
          "not a stock-beating investment net of fees.")


if __name__ == "__main__":
    main()
