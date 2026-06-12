"""Real-tape verification — Study 80 (Cold-Open).  Regenerates docs/results.md numbers.

Fetches (or reads from cache) ^GSPC daily history since 1950, builds the annual
January and Feb-Dec return series, runs the January Barometer analysis, and prints
all headline numbers.  Network is touched only with --fetch.

    python studies/80-cold-open/examples/verify.py            # cache-only
    python studies/80-cold-open/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cold_open import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_gspc(fetch=fetch)
    fp = data.fingerprint(df)
    print(f"^GSPC annual tape: {df.index[0]}-{df.index[-1]}  n={len(df)}  fp={fp}")
    print()

    # ---- full-sample analysis ----
    result = st.summarize(df)
    print("=== full-sample hit-rate analysis ===")
    print(f"  n={result['n']}  n_up_jan={result['n_up_jan']}  n_dn_jan={result['n_dn_jan']}")
    print(f"  hit_rate={result['hit_rate']:.3f}  (pval vs coin: {result['pval_vs_coin']:.4f})")
    print(f"  base_rate_fbd_pos={result['base_rate']:.3f}  (pval vs base: {result['pval_vs_base_rate']:.4f})")
    print(f"  perm_pval={result['perm_pval']:.4f}")
    print()

    print("=== mean contrast ===")
    print(f"  mean_fbd | up-Jan: {result['mean_fbd_up_jan_pct']:.2f}%")
    print(f"  mean_fbd | dn-Jan: {result['mean_fbd_dn_jan_pct']:.2f}%")
    print(f"  contrast: {result['contrast_pct']:.2f}%  HAC t={result['tstat_hac']:.4f}")
    print(f"  Welsh p: {result['pval_welsh']:.4f}")
    print()

    # ---- strategy vs buy-and-hold ----
    bstrat = st.barometer_strategy(df, cost_bps=10.0)
    print("=== barometer strategy vs buy-and-hold (cost=10 bps) ===")
    print(f"  strat mean/yr: {bstrat['strat_mean_ann_pct']:.2f}%  Sharpe: {bstrat['strat_sharpe']:.3f}")
    print(f"  bah  mean/yr: {bstrat['bah_mean_ann_pct']:.2f}%  Sharpe: {bstrat['bah_sharpe']:.3f}")
    print()

    # ---- sub-period stability ----
    print("=== sub-period stability ===")
    for start_y, end_y in [(1951, 1985), (1986, 2025)]:
        sub = df.loc[start_y:end_y]
        c = st.mean_contrast(sub)
        h = st.hit_rate_analysis(sub)
        print(f"  {start_y}-{end_y}: n={len(sub)}  hit_rate={h['hit_rate']:.3f}"
              f"  contrast={c['contrast_pct']:.1f}%  HAC-t={c['tstat_hac']:.2f}"
              f"  pval_coin={h['pval_vs_coin']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
