"""Real-tape verification — Study 192 (Tax-Day).  Regenerates docs/results.md numbers.

Reads (or refreshes) the shared ^GSPC cache, runs the tax-day window comparisons
(pre-window 10 days before April 15, post-window 5 days on/after), the April
broad-seasonal baseline, an October-15 placebo, and a Bonferroni correction.

    python studies/192-tax-day/examples/verify.py            # cache-only
    python studies/192-tax-day/examples/verify.py --fetch    # refresh from network
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tax_day import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_gspc(start="1928-01-01", fetch=fetch)
    fp = data.fingerprint(df)
    n_years = df.index.year.max() - df.index.year.min() + 1
    print(f"^GSPC tape: {df.index[0].date()} to {df.index[-1].date()}  fp={fp}")
    print(
        f"Total trading days: {len(df):,}  |  "
        f"Pre-tax days: {int(df['pre_tax'].sum())}  |  "
        f"Post-tax days: {int(df['post_tax'].sum())}  |  "
        f"Years: {n_years}"
    )
    print()

    s = st.summarize(df)

    print("=== H1: Pre-tax window (10 days before April 15) vs rest ===")
    print(
        f"n_pre={s['n_pre']}  mean_pre={s['mean_pre_bps']:+.2f} bps  "
        f"n_rest={s['n_post'] + len(df) - s['n_pre'] - s['n_post']}  "
        f"mean_rest={st.window_comparison(df, 'pre_tax')['mean_rest_bps']:+.2f} bps"
    )
    print(
        f"contrast={s['contrast_pre_bps']:+.2f} bps  "
        f"Welch-t={s['tstat_pre_welch']:+.3f}  p={s['pval_pre']:.4f}  "
        f"HAC-t={s['tstat_pre_hac']:+.3f}"
    )
    print(f"Bonferroni-corrected p (k=2): {s['pval_pre_bonferroni']:.4f}")
    print()

    print("=== H2: Post-tax window (5 days on/after April 15) vs rest ===")
    print(
        f"n_post={s['n_post']}  mean_post={s['mean_post_bps']:+.2f} bps"
    )
    print(
        f"contrast={s['contrast_post_bps']:+.2f} bps  "
        f"Welch-t={s['tstat_post_welch']:+.3f}  p={s['pval_post']:.4f}  "
        f"HAC-t={s['tstat_post_hac']:+.3f}"
    )
    print(f"Bonferroni-corrected p (k=2): {s['pval_post_bonferroni']:.4f}")
    print()

    print("=== Broad April seasonal baseline ===")
    print(
        f"n_apr={s['n_apr']}  mean_apr={s['mean_apr_bps']:+.2f} bps  "
        f"contrast_vs_rest={s['contrast_apr_bps']:+.2f} bps  p={s['pval_apr']:.4f}"
    )
    print(f"Unconditional all-day mean: {s['mean_all_bps']:+.2f} bps")
    print()

    print("=== Oct-15 placebo (same window size, no IRS event) ===")
    print(
        f"n_oct={s['n_oct_window']}  mean_oct={s['mean_oct_bps']:+.2f} bps  "
        f"contrast={s['contrast_oct_bps']:+.2f} bps  p={s['pval_oct']:.4f}"
    )
    print()

    print("=== Verdict ===")
    bonf_pre_ok = s["pval_pre_bonferroni"] < 0.05
    bonf_post_ok = s["pval_post_bonferroni"] < 0.05
    print(f"Any hypothesis survives Bonferroni: {s['any_survives_bonferroni']}")
    print(f"  pre Bonf-p={s['pval_pre_bonferroni']:.4f} -> {'SURVIVES' if bonf_pre_ok else 'fails'}")
    print(f"  post Bonf-p={s['pval_post_bonferroni']:.4f} -> {'SURVIVES' if bonf_post_ok else 'fails'}")
    signal = "REAL" if s["any_survives_bonferroni"] else "NONE"
    print(f"\nSignal: {signal}")
    print("Tradability: MIRAGE (calendar effect, ~10 events/year, no cost-surviving edge)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
