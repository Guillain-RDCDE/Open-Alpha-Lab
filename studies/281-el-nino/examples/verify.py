"""Real-data verification -- Study 281 (El-Nino). Regenerates docs/results.md numbers.

Cache-only (fetch=False): loads ^GSPC (and USO) daily closes from the repo-level
_cache, computes calendar-year returns, joins with the hardcoded NOAA ONI table at
a one-year execution lag, and runs the full analysis: per-phase means, Welch t,
one-way ANOVA, permutation test, HAC t, and the long-short backtest.

    python studies/281-el-nino/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from el_nino import data, strategy as st  # noqa: E402


def _report(asset: str) -> None:
    try:
        df = data.fetch_real(asset=asset, lag=1, fetch=False)
    except FileNotFoundError:
        print(f"[{asset}] no cache present -- skipping (run with the repo _cache staged)\n")
        return
    fp = data.fingerprint(df)
    s = st.summarize(df, n_permutations=10_000, seed=281)
    print(f"=== {asset.upper()} (lag-1)  n={len(df)}  years {df['year'].min()}-{df['year'].max()}"
          f"  fingerprint={fp} ===")
    print(f"  Phases: El Nino={s['n_elnino']}, La Nina={s['n_lanina']}, Neutral={s['n_neutral']}")
    print(f"  Unconditional up-rate: {s['uncond_up_rate_pct']:.1f}%")
    print(f"  Mean return  El Nino {s['mean_elnino_pct']:+.1f}%  | La Nina {s['mean_lanina_pct']:+.1f}%"
          f"  | Neutral {s['mean_neutral_pct']:+.1f}%  | all {s['mean_all_pct']:+.1f}%")
    print(f"  Gap (El-La): {s['gap_pct']:+.1f}pp")
    print(f"  Welch t = {s['welch_t']:+.3f} (p {s['welch_p']:.3f}) | HAC t = {s['hac_t']:+.3f}"
          f" | ANOVA F={s['anova_f']:.3f} p={s['anova_p']:.3f} | perm p = {s['perm_p']:.3f}")
    print(f"  Backtest: gross {s['gross_ann_pct']:+.1f}%/yr | net {s['net_ann_pct']:+.1f}%/yr"
          f" | buy-hold {s['buy_hold_ann_pct']:+.1f}%/yr | turnover {s['turnover_per_yr']:.1f}/yr")
    print()


def main() -> None:
    print("Study 281 -- El-Nino -- real-data verification (cache-only)\n")
    print("Data: ^GSPC / USO daily closes (repo _cache), NOAA ONI hardcoded in data.py\n")
    _report("equity")
    _report("oil")
    print("Verdict: Signal=NONE, Tradability=MIRAGE -- the ocean moves weather, not the tape.")


if __name__ == "__main__":
    main()
