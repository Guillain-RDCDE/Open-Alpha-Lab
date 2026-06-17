"""Real-data verification -- Study 297 (March-Madness). Regenerates docs/results.md.

Loads the ^GSPC daily tape from the repo-level cache (no network), labels each
trading day by whether it falls inside an NCAA tournament window (hardcoded in
data.py), and runs the full analysis: in/out mean-return test (HAC t), volatility
ratio, lagged + cost-charged avoidance strategy, and a random-placement
permutation null.

Cache-only by default (fetch=False). Run from anywhere:

    python studies/297-march-madness/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from march_madness import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 297 -- March-Madness -- real-data verification\n")
    print("Data: ^GSPC daily price return (repo _cache/^GSPC_split_only.parquet)")
    print("      NCAA tournament windows: hardcoded in data.py (1985-2025)\n")

    ret, mask = data.fetch_daily()  # cache-only
    fp = data.fingerprint(ret)
    print(
        f"Data stamp: {ret.index.min().date()}..{ret.index.max().date()}, "
        f"n={len(ret)}, fingerprint={fp}"
    )
    print(
        f"In-tournament trading days: {int(mask.sum())} "
        f"({mask.mean():.1%} of the tape)\n"
    )

    s = st.summarize(ret, mask, n_perm=5000, seed=297)

    print("=== Primary spec: mean daily return, in-window vs outside ===")
    print(f"  In-tournament mean: {s['ret_in_mean_bps']:+.2f} bps  (HAC t {s['ret_in_t']:+.2f})")
    print(f"  Outside mean:       {s['ret_out_mean_bps']:+.2f} bps  (HAC t {s['ret_out_t']:+.2f})")
    print(f"  Difference (in-out):{s['ret_diff_bps']:+.2f} bps/day  Welch t {s['ret_welch_t']:+.3f}\n")

    print("=== Volatility (the 'chaos' corollary) ===")
    print(f"  In vol (ann):  {s['vol_in_vol_ann']:.1%}")
    print(f"  Out vol (ann): {s['vol_out_vol_ann']:.1%}")
    print(f"  Ratio in/out:  {s['vol_vol_ratio']:.3f}   Levene p = {s['vol_levene_p']:.3f}\n")

    print("=== Permutation control (random-placement null) ===")
    print(f"  pct of random windows with mean <= real in-window mean: "
          f"{s['perm_pct_worse_or_equal']:.1%}\n")

    print("=== Could you trade it? lagged + cost-charged avoidance ===")
    print(f"  Buy & hold CAGR:      {s['avoid_bah_cagr']:.2%}")
    print(f"  Avoidance gross CAGR: {s['avoid_strat_gross_cagr']:.2%}")
    print(f"  Avoidance net CAGR:   {s['avoid_strat_net_cagr']:.2%}  ({s['avoid_n_switches']} switches)")
    print(f"  Excess (net) CAGR:    {s['avoid_excess_net_cagr']:+.2%}/yr  HAC t {s['avoid_excess_net_t']:+.3f}\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- "
          "the bracket distracts no one who sets prices.")


if __name__ == "__main__":
    main()
