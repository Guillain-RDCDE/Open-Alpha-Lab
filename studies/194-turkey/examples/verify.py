"""Real-tape verification — Study 194 (Turkey). Regenerates docs/results.md numbers.

Reads (or refreshes) the shared ^GSPC cache, runs the Thanksgiving Wednesday-before
and Black Friday (Friday-after) comparison against matched-weekday baselines, the
Tuesday-before placebo, and the Bonferroni-corrected summary.

    python studies/194-turkey/examples/verify.py            # cache-only
    python studies/194-turkey/examples/verify.py --fetch    # refresh from network
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turkey import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_gspc(start="1928-01-01", fetch=fetch)
    fp = data.fingerprint(df)
    print(f"^GSPC tape: {df.index[0].date()} to {df.index[-1].date()}  fp={fp}")
    print(
        f"Total trading days: {len(df):,}  |  "
        f"Wed-before-Thanksgiving: {int(df['is_wed_before_tsg'].sum())}  |  "
        f"Black Fridays: {int(df['is_fri_after_tsg'].sum())}"
    )
    print()

    s = st.summarize(df)

    print("=== PRIMARY A: Wednesday before Thanksgiving vs other Wednesdays ===")
    print(
        f"n_wed={s['n_wed']}  mean_wed={s['mean_wed_bps']:+.2f} bps  "
        f"n_other={s['n_other_wed']}  mean_other={s['mean_other_wed_bps']:+.2f} bps"
    )
    print(
        f"contrast={s['contrast_wed_bps']:+.2f} bps  "
        f"Welch-t={s['tstat_welch_wed']:+.3f}  p={s['pval_welch_wed']:.4f}  "
        f"HAC-t(wed)={s['tstat_hac_wed']:+.3f}  "
        f"Bonferroni-p={s['pval_bonferroni_wed']:.4f}"
    )
    print()

    print("=== PRIMARY B: Black Friday (day after Thanksgiving) vs other Fridays ===")
    print(
        f"n_fri={s['n_fri']}  mean_fri={s['mean_fri_bps']:+.2f} bps  "
        f"n_other={s['n_other_fri']}  mean_other={s['mean_other_fri_bps']:+.2f} bps"
    )
    print(
        f"contrast={s['contrast_fri_bps']:+.2f} bps  "
        f"Welch-t={s['tstat_welch_fri']:+.3f}  p={s['pval_welch_fri']:.4f}  "
        f"HAC-t(fri)={s['tstat_hac_fri']:+.3f}  "
        f"Bonferroni-p={s['pval_bonferroni_fri']:.4f}"
    )
    print()

    print("=== PLACEBO: Tuesday before Thanksgiving vs other Tuesdays ===")
    print(
        f"n_tue={s['n_tue']}  mean_tue={s['mean_tue_bps']:+.2f} bps  "
        f"contrast={s['contrast_tue_bps']:+.2f} bps  p={s['pval_welch_tue']:.4f}"
    )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
