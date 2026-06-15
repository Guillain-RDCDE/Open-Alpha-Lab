"""Real-tape verification — Study 164 (Mercury-Retrograde). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the ^GSPC daily tape since 2000, labels each trading day
with a Mercury-retrograde flag from the hardcoded ephemeris table, and runs all four tests:
(A) mean return difference, (B) vol comparison, (C) retrograde-avoidance vs buy-and-hold,
(D) permutation null. Network is touched only with --fetch.

    python studies/164-mercury-retrograde/examples/verify.py            # cache-only
    python studies/164-mercury-retrograde/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mercury_retrograde import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        print("Fetching ^GSPC daily data from Yahoo Finance...")
        ret, mask = data.fetch_daily(fetch=True)
        print(
            f"^GSPC  {ret.index[0].date()}..{ret.index[-1].date()}  "
            f"n={len(ret)}  fp={data.fingerprint(ret)}"
        )
        print()
    else:
        ret, mask = data.fetch_daily(fetch=False)

    print(f"=== Data summary ===")
    print(f"  Period   : {ret.index[0].date()} to {ret.index[-1].date()}")
    print(f"  n days   : {len(ret)}")
    print(f"  Retrograde: {int(mask.sum())} days ({float(mask.mean()):.1%})")
    print(f"  Fingerprint: {data.fingerprint(ret)}")
    print()

    # A: Mean return test
    print("=== (A) Return test: retrograde vs direct ===")
    rtest = st.retrograde_return_test(ret, mask)
    print(f"  Retrograde  n={rtest['n_retro']}  mean={rtest['retro_mean_bps']:+.2f} bps  "
          f"HAC t={rtest['retro_t']:+.2f}")
    print(f"  Direct      n={rtest['n_direct']}  mean={rtest['direct_mean_bps']:+.2f} bps  "
          f"HAC t={rtest['direct_t']:+.2f}")
    print(f"  Diff = {rtest['diff_bps']:+.2f} bps  Welch t = {rtest['welch_t']:+.2f}")
    print()

    # B: Volatility test
    print("=== (B) Volatility test ===")
    vtest = st.retrograde_vol_test(ret, mask)
    print(f"  Retrograde vol (ann) : {vtest['retro_vol_ann']:.3f}")
    print(f"  Direct vol (ann)     : {vtest['direct_vol_ann']:.3f}")
    print(f"  Vol ratio (retro/direct) : {vtest['vol_ratio']:.3f}")
    print(f"  Levene stat = {vtest['levene_stat']:.3f}  p = {vtest['levene_p']:.4f}")
    print()

    # C: Avoidance strategy
    print("=== (C) Retrograde-avoidance strategy ===")
    avoid = st.retrograde_avoidance(ret, mask)
    print(f"  Buy-and-hold CAGR  : {avoid['bah_cagr']:+.3%}")
    print(f"  Avoidance CAGR     : {avoid['strat_cagr']:+.3%}")
    print(f"  Excess CAGR        : {avoid['excess_cagr']:+.3%}")
    print(f"  Excess HAC t       : {avoid['excess_hac_t']:+.2f}")
    print()

    # D: Permutation null
    print("=== (D) Permutation null (5000 shuffles) ===")
    perm = st.permutation_null(ret, mask, n_perm=5000, seed=164)
    print(f"  Retrograde observed mean : {perm['retro_mean_bps']:+.2f} bps")
    print(f"  Fraction of random 19%-blocks as bad or worse: {perm['pct_worse_than_retro']:.1%}")
    print()

    print("=== Verdict ===")
    print(f"  Signal   : NONE  (Welch t = {rtest['welch_t']:+.2f}, diff = {rtest['diff_bps']:+.2f} bps)")
    print(f"  Vol      : vol ratio = {vtest['vol_ratio']:.3f}, Levene p = {vtest['levene_p']:.3f}")
    print(f"  Tradability: MIRAGE  (avoidance excess CAGR = {avoid['excess_cagr']:+.3%})")
    print(f"  Permutation: {perm['pct_worse_than_retro']:.1%} of random periods are as bad or worse")
    print("  => The planets do not trade.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
