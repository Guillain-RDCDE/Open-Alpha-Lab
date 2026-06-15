"""Real-tape verification — Study 159 (Presidential-Party). Regenerates docs/results.md numbers.

Loads the pre-staged Shiller monthly S&P 500 dataset and computes the party-stratified
real return statistics. Optionally also fetches the Yahoo ^GSPC daily tape for a
complementary check.

    python studies/159-presidential-party/examples/verify.py          # Shiller only
    python studies/159-presidential-party/examples/verify.py --daily  # also fetch ^GSPC
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from presidential_party import data, strategy as st  # noqa: E402


def main(daily: bool = False) -> None:
    print("Study 159 — Presidential-Party — real-tape verification\n")

    # ---- Shiller monthly (primary tape) ------------------------------------
    print("=== Shiller monthly S&P 500 (real returns, 1927-2023) ===")
    df = data.load_shiller()
    fp = data.fingerprint(df)
    print(f"  Rows: {len(df)}, Range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Fingerprint: {fp}")
    print(f"  D months: {(df['party']=='D').sum()}, R months: {(df['party']=='R').sum()}")
    print()

    s = st.summarize(df)

    print("--- Per-party statistics ---")
    print(f"  Democrat:   mean = {s['dem_mean_bps']:+.2f} bps/month  "
          f"({s['dem_mean_ann']:+.2f}%/yr)  HAC t = {s['dem_hac_t']:+.3f}  "
          f"n = {s['n_dem']}")
    print(f"  Republican: mean = {s['rep_mean_bps']:+.2f} bps/month  "
          f"({s['rep_mean_ann']:+.2f}%/yr)  HAC t = {s['rep_hac_t']:+.3f}  "
          f"n = {s['n_rep']}")
    print(f"  Unconditional mean: {s['overall_mean_bps']:+.2f} bps/month")
    print()

    print("--- D-minus-R gap ---")
    print(f"  Gap: {s['gap_monthly_bps']:+.2f} bps/month  ({s['gap_ann_pct']:+.2f}%/yr)")
    print(f"  Welch t (monthly obs): {s['welch_t']:+.3f}")
    print(f"  HAC t (D-dummy regression): {s['hac_t']:+.3f}")
    print()

    print("--- Permutation test (block-level, n_perm=10,000) ---")
    print(f"  Observed gap: {s['gap_monthly_bps']:+.2f} bps/month")
    print(f"  Two-sided p-value: {s['perm_pvalue']:.4f}")
    print(f"  n_terms in permutation: {s['n_terms']}")
    print()

    # ---- Per-president breakdown -------------------------------------------
    print("--- Per-president breakdown ---")
    from presidential_party import strategy as _st
    pr = _st.party_returns(df)
    print(f"  Democrat avg:   {pr.loc['D','mean_ann_pct']:+.2f}%/yr  "
          f"n={pr.loc['D','n']}  HAC t={pr.loc['D','hac_tstat']:+.2f}")
    print(f"  Republican avg: {pr.loc['R','mean_ann_pct']:+.2f}%/yr  "
          f"n={pr.loc['R','n']}  HAC t={pr.loc['R','hac_tstat']:+.2f}")
    print()

    print("--- Verdict ---")
    print("  Signal: WEAK — there is a visible gap (~+4.6%/yr annualised) but")
    print("  Welch t = 1.45 (monthly) and permutation p = 0.32 (block-term level).")
    print("  With only ~17 terms the gap is statistically unresolvable from noise")
    print("  and business-cycle confounds. The Santa-Clara & Valkanov (2003) finding")
    print("  was real but driven by the Depression era and cannot be certified as causal.")
    print("  Tradability: MIRAGE — party composition is known only after election, not")
    print("  tradable on a signal basis, and the gap is not distinguishable from noise.")

    # ---- Optional: Yahoo daily ^GSPC ----------------------------------------
    if daily:
        print()
        print("=== Yahoo daily ^GSPC (1928-present) ===")
        try:
            df_d = data.fetch_gspc(fetch=True)
            # Monthly resample for fair comparison
            df_m = (df_d.groupby(df_d.index.to_period("M"))
                    .apply(lambda g: (1 + g["ret"].dropna()).prod() - 1))
            print(f"  Rows (daily): {len(df_d)}")
            fp_d = data.fingerprint(df_d)
            print(f"  Fingerprint: {fp_d}")
        except Exception as exc:
            print(f"  (skipped: {exc})")


if __name__ == "__main__":
    main(daily="--daily" in sys.argv)
