"""Real-data verification -- Study 300 (Sports-Sentiment). Regenerates docs/results.md.

Loads the per-ticker daily ETF parquet cache, computes the next-trading-day return
after each hardcoded elimination loss, and runs the full analysis: mean, Newey-West
HAC t-stat, bootstrap, sub-group cuts, and the tradable short-on-loss PnL.

Cache-only by default (no network). To populate the cache the first time, run with
the environment variable SPORTS_FETCH=1 (touches yfinance):

    python studies/300-sports-sentiment/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sports_sentiment import data, strategy as st  # noqa: E402


def main() -> None:
    fetch = os.environ.get("SPORTS_FETCH", "0") == "1"
    print("Study 300 -- Sports-Sentiment (Edmans loss effect) -- real-data verification\n")
    print("Data: iShares single-country ETFs + ^GSPC daily closes (price-only, USD)")
    print("      Elimination losses: hardcoded in data.py (1998-2024)\n")

    df = data.fetch_event_returns(fetch=fetch)
    fp = data.fingerprint(df)
    print(f"Data stamp: events n={len(df)}, response dates "
          f"{df['ret_date'].min().date()}..{df['ret_date'].max().date()}, "
          f"fingerprint={fp}\n")

    s = st.loss_effect_stats(df, hac_lags=1, n_boot=10_000, seed=300)
    print("=== Headline next-day loss-effect test ===")
    print(f"  Mean next-day return: {s['mean_bps']:+.2f} bps  (Edmans hypothesis ~ -49 bps)")
    print(f"  Newey-West HAC t:     {s['hac_t']:+.3f}  (REAL bar: <= -2 with negative mean)")
    print(f"  Plain i.i.d. t:       {s['plain_t']:+.3f}")
    print(f"  Bootstrap P(mean>=0): {s['boot_p_ge0']:.3f}")
    print(f"  Down days:            {s['n_down']}/{s['n']} ({s['frac_neg']:.1%})\n")

    print("=== Sub-group cuts ===")
    print(st.subgroup_summary(df, hac_lags=1, seed=300).to_string(index=False))
    print()

    tr = st.tradable_pnl(df, cost_bps_oneway=5.0)
    print("=== Tradability: short the ETF the next session ===")
    print(f"  Gross mean PnL: {tr['gross_mean_bps']:+.2f} bps/event")
    print(f"  Net mean PnL:   {tr['net_mean_bps']:+.2f} bps/event (5 bps one-way x2)")
    print(f"  Net hit-rate:   {tr['net_hit_rate']:.1%}\n")

    print("Verdict: Signal=WEAK (famous in the literature, but real-tape HAC t=+1.6, "
          "wrong sign), Tradability=MIRAGE (short-on-loss nets -43 bps/event).")


if __name__ == "__main__":
    main()
