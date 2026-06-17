"""Real-data verification -- Study 277 (Trading-Cards). Regenerates docs/results.md numbers.

Loads the cached S&P 500 price series (repo-level _cache/^GSPC_split_only.parquet),
computes calendar-year price returns, joins with the hardcoded curated graded-card
index, and runs the full analysis: return/risk, Newey-West alpha, frictions, and
block bootstrap.

The ^GSPC parquet ships with the repo and is read-only (cache-only, no fetch). Run:

    python studies/277-trading-cards/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_cards import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 277 -- Trading-Cards -- real-data verification\n")
    print("Data: S&P 500 price (cached ^GSPC_split_only.parquet)")
    print("      Graded-card index: curated, hardcoded in data.py (base 100 = 1990)\n")

    df = data.fetch_gspc_annual()  # cache-only
    fp_c = data.fingerprint(df, "card_return")
    fp_g = data.fingerprint(df, "gspc_return")

    print(f"Data stamp: years {df['year'].min()}-{df['year'].max()}, n={len(df)}")
    print(f"  card_return fingerprint = {fp_c}")
    print(f"  gspc_return fingerprint = {fp_g}\n")

    s = st.summarize(df, one_way_cost=0.15, holding_years=5.0, n_boot=5000, seed=277)

    print("=== Return & risk (price-only, 1991-2025) ===")
    print(f"  Cards : CAGR {s['card_cagr_pct']:+.1f}%  vol {s['card_vol_pct']:.1f}%  "
          f"Sharpe {s['card_sharpe']:.2f}  maxDD {s['card_maxdd_pct']:.1f}%")
    print(f"  S&P   : CAGR {s['gspc_cagr_pct']:+.1f}%  vol {s['gspc_vol_pct']:.1f}%  "
          f"Sharpe {s['gspc_sharpe']:.2f}  maxDD {s['gspc_maxdd_pct']:.1f}%\n")

    print("=== Structural alpha (Newey-West HAC, 2 lags) ===")
    print(f"  alpha = {s['alpha_ann_pct']:+.2f}%/yr  (HAC t = {s['t_alpha']:+.2f}, p = {s['p_alpha']:.3f})")
    print(f"  beta  = {s['beta']:+.2f}   corr = {s['corr']:+.2f}   R^2 = {s['r2']:.3f}\n")

    print("=== After collectible frictions (~15% one-way, 5yr hold) ===")
    print(f"  Net card CAGR : {s['card_net_cagr_pct']:+.1f}%/yr")
    print(f"  Net alpha     : {s['alpha_net_ann_pct']:+.2f}%/yr  (HAC t = {s['t_alpha_net']:+.2f})\n")

    print("=== Block bootstrap (alpha, gross) ===")
    print(f"  95% CI: [{s['boot_alpha_p2.5']:+.1f}%, {s['boot_alpha_p97.5']:+.1f}%]  "
          f"median {s['boot_alpha_p50']:+.1f}%  frac>0 {s['boot_frac_alpha_pos']:.0%}\n")

    print("=== One-boom fragility ===")
    print(f"  Card CAGR ex-2020-2021: {s['card_cagr_ex_boom_pct']:+.1f}%/yr "
          f"(vs {s['card_cagr_pct']:+.1f}% full sample)\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- one boom, brutal frictions, no net edge.")


if __name__ == "__main__":
    main()
