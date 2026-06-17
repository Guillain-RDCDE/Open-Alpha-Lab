"""Real-data verification -- Study 235 (World-Cup-Effect).

Loads S&P 500 daily data (yfinance ^GSPC, cached locally), tags each trading
day as inside/outside a FIFA World Cup window (19 editions, 1950-2022 hardcoded
in data.py), and runs the full analysis: per-edition means, Welch t-test,
per-edition one-sample t, permutation test, matched-control comparison.

Cache file: studies/235-world-cup-effect/_cache/sp500_daily.parquet
(study-local, NOT the shared repo _cache/).

Run from anywhere:

    python studies/235-world-cup-effect/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_cup_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 235 -- World-Cup-Effect -- real-data verification\n")
    print("Data: S&P 500 daily via yfinance (^GSPC, 1950-2023)")
    print("      World Cup windows: hardcoded in data.py (19 editions, 1950-2022)\n")

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "_cache")
    price_df = data.fetch_sp500_daily(cache_dir=cache_dir)
    tagged = data.tag_wc_windows(price_df)
    fp = data.fingerprint(price_df)

    print(f"Data stamp: {price_df.index.min().date()} to {price_df.index.max().date()}, "
          f"n={len(price_df)}, fingerprint={fp}\n")

    wc_count = tagged["in_wc"].sum()
    non_wc_count = (~tagged["in_wc"]).sum()
    ctrl_count = tagged["ctrl_window"].sum()
    uncond_mean_bps = price_df["sp500_return"].mean() * 1e4
    print(f"WC-window trading days: {wc_count}")
    print(f"Non-WC trading days:    {non_wc_count}")
    print(f"Control window days:    {ctrl_count}")
    print(f"Unconditional daily mean: {uncond_mean_bps:.2f} bps\n")

    # Per-edition breakdown
    wr = st.window_returns(tagged)
    print("Per-edition WC window returns (bps/day):")
    print(wr.to_string(index=False))
    n_neg = (wr["mean_wc_return_bps"] < 0).sum()
    print(f"\nNegative WC windows: {n_neg}/{len(wr)}")

    # Full summary
    s = st.summarize(tagged, n_permutations=10_000, seed=235)

    print("\n=== Primary test: per-edition t-test vs unconditional ===")
    print(f"  Mean WC-window return:     {s['mean_wc_bps']:+.2f} bps/day")
    print(f"  Mean non-WC return:        {s['mean_non_wc_bps']:+.2f} bps/day")
    print(f"  Mean control window:       {s['mean_ctrl_bps']:+.2f} bps/day")
    print(f"  Diff (WC - non-WC):        {s['diff_vs_non_wc_bps']:+.2f} bps/day")
    print(f"  Diff (WC - control):       {s['diff_vs_ctrl_bps']:+.2f} bps/day")

    print("\n=== Statistical tests ===")
    print(f"  Welch t (WC vs non-WC days, pooled):  t={s['t_vs_all']:+.3f}, p={s['p_vs_all']:.4f}")
    print(f"  Welch t (WC vs control days, pooled): t={s['t_vs_ctrl']:+.3f}, p={s['p_vs_ctrl']:.4f}")
    print(f"  Per-edition t vs zero:                t={s['t_edition_zero']:+.3f}, "
          f"p={s['p_edition_zero']:.4f}")
    print(f"  Permutation p (two-sided, 10k):       p={s['perm_p']:.4f}")

    print("\n=== Confound warning ===")
    print("  1950 WC (-87.9 bps/day): Korean War started June 25, 1950")
    print("  1974 WC (-59.3 bps/day): Oil crisis / bear market")
    print("  2002 WC (-33.8 bps/day): Dot-com bust / accounting scandals")
    print("  These macro crises drive the negative signal; football sentiment")
    print("  cannot be separated from contemporaneous macro shocks.")

    print(f"\nFingerprint: {fp}")
    print("\nVerdict: Signal=WEAK, Tradability=MIRAGE -- confounded by macro crises, n=19 fragile.")


if __name__ == "__main__":
    main()
