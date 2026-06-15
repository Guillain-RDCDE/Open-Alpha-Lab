"""Real-tape verification — Study 163 (Friday-13th). Regenerates docs/results.md numbers.

Reads (or refreshes) the shared ^GSPC cache, runs the Friday-13th comparison
against other Fridays, the Friday-6th placebo, and the full multiple-comparisons
sweep across all four "special Friday" day-of-month slots.

    python studies/163-friday-13th/examples/verify.py            # cache-only
    python studies/163-friday-13th/examples/verify.py --fetch    # refresh from network
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from friday_13th import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_gspc(start="1928-01-01", fetch=fetch)
    fp = data.fingerprint(df)
    print(f"^GSPC tape: {df.index[0].date()} to {df.index[-1].date()}  fp={fp}")
    print(f"Total trading days: {len(df):,}  |  Friday-13ths: {int(df['is_f13'].sum())}")
    print()

    s = st.summarize(df)

    print("=== PRIMARY: Friday-13th vs other Fridays ===")
    print(
        f"n_f13={s['n_f13']}  mean_f13={s['mean_f13_bps']:+.2f} bps  "
        f"n_other={s['n_other_fri']}  mean_other={s['mean_other_fri_bps']:+.2f} bps"
    )
    print(
        f"contrast={s['contrast_bps']:+.2f} bps  "
        f"Welch-t={s['tstat_welch']:+.3f}  p={s['pval_welch']:.4f}  "
        f"HAC-t(F13)={s['tstat_f13_hac']:+.3f}"
    )
    print()

    print("=== PLACEBO: Friday-6th vs other Fridays ===")
    print(
        f"n_f6={s['n_f6']}  mean_f6={s['mean_f6_bps']:+.2f} bps  "
        f"contrast={s['contrast_f6_bps']:+.2f} bps  p={s['pval_f6']:.4f}"
    )
    print()

    print("=== MULTIPLE COMPARISONS (Bonferroni k=4) ===")
    print(s["mc_table"].to_string(index=False))
    print(f"\nBonferroni-corrected p for day=13: {s['pval_bonferroni_13']:.4f}")
    print()

    # Sub-period: Kolb & Rodriguez 1987 era
    df_kr = df[(df.index.year >= 1940) & (df.index.year <= 1987)]
    r_kr = st.friday_comparison(df_kr)
    print(f"=== K&R era (1940-1987) n_f13={r_kr['n_f13']} ===")
    print(
        f"contrast={r_kr['contrast_bps']:+.2f} bps  "
        f"Welch-t={r_kr['tstat_welch']:+.3f}  p={r_kr['pval_welch']:.4f}"
    )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
