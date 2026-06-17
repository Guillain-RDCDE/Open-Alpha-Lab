"""Real-data verification -- Study 283 (Hurricane-Season). Regenerates docs/results.md numbers.

Loads daily ^GSPC from the repo-level cache and (if staged) the equal-weight
P&C-insurer basket from the study cache, flags the Atlantic hurricane season
(Jun 1 - Nov 30), and runs the full analysis: in-season vs off-season mean test
(Welch + HAC t + block permutation), the event-window study around major U.S.
landfalls, and the avoid-the-season timing benchmark.

Cache-only -- no network. Run from anywhere:

    python studies/283-hurricane-season/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hurricane_season import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 283 -- Hurricane-Season -- real-data verification\n")
    print("Data: ^GSPC daily (repo-level _cache/^GSPC_split_only.parquet), price-only")
    print("      Insurer basket: _cache/hurricane_insurers.parquet (if staged)")
    print("      Major landfalls: hardcoded in data.py (1992-2024)\n")

    mkt = data.fetch_market()
    fp = data.fingerprint(mkt)
    print(f"Market stamp: {mkt.index.min().date()}..{mkt.index.max().date()}, "
          f"n={len(mkt)}, fingerprint={fp}\n")

    try:
        ins = data.fetch_insurers()
        have_ins = True
    except FileNotFoundError:
        ins = None
        have_ins = False
        print("(insurer cache not staged -- insurer arm skipped)\n")

    lf = data.landfall_table()
    s = st.summarize(mkt, insurers=ins, landfalls=lf if have_ins else None,
                     n_permutations=5_000, seed=283)

    print("=== Broad market (^GSPC): in-season vs off-season daily returns ===")
    print(f"  n in-season:  {s['n_in']}   n off-season: {s['n_off']}")
    print(f"  mean in:      {s['mkt_mean_in']:+.3f} bps/day")
    print(f"  mean off:     {s['mkt_mean_off']:+.3f} bps/day")
    print(f"  spread (in-off): {s['mkt_spread']:+.3f} bps/day")
    print(f"  Welch t:      {s['mkt_welch_t']:+.3f}")
    print(f"  HAC t:        {s['mkt_hac_t']:+.3f}")
    print(f"  perm p:       {s['mkt_perm_p']:.4f}")
    print(f"  annualised:   in {s['mkt_ann_in']:+.2f}%  off {s['mkt_ann_off']:+.2f}%\n")

    if have_ins:
        print("=== Insurer basket (equal-weight P&C proxy): in-season vs off-season ===")
        print(f"  mean in:      {s['ins_mean_in']:+.3f} bps/day")
        print(f"  mean off:     {s['ins_mean_off']:+.3f} bps/day")
        print(f"  spread:       {s['ins_spread']:+.3f} bps/day")
        print(f"  Welch t:      {s['ins_welch_t']:+.3f}")
        print(f"  HAC t:        {s['ins_hac_t']:+.3f}")
        print(f"  perm p:       {s['ins_perm_p']:.4f}")
        print(f"  annualised:   in {s['ins_ann_in']:+.2f}%  off {s['ins_ann_off']:+.2f}%\n")
        print("=== Event windows around major landfalls ([-5,+20], 1d lag) ===")
        print(f"  n events:     {s['ew_n']}")
        print(f"  insurer CAR:  {s['ew_car_end']:+.3f}%  (t = {s['ew_car_t']:+.3f})\n")

    tw = st.avoid_season_strategy(mkt, cost_bps=1.0)
    print("=== 'Sit out the season' timing rule vs buy-and-hold (^GSPC) ===")
    print(f"  buy & hold:     {tw['bah_ann']:+.2f}%/yr")
    print(f"  strategy gross: {tw['strat_gross_ann']:+.2f}%/yr")
    print(f"  strategy net:   {tw['strat_net_ann']:+.2f}%/yr")
    print(f"  switches/yr:    {tw['flips_per_year']:.2f}\n")

    print("Verdict: see docs/results.md")


if __name__ == "__main__":
    main()
