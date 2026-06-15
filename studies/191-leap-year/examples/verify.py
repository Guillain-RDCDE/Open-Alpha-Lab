"""Real-tape verification — Study 191 (Leap-Year).  Regenerates docs/results.md numbers.

Reads the shared Shiller cache, runs all three tests (leap-year premium, presidential-
cycle breakdown, and the multiple-comparisons Bonferroni reckoning).

    python studies/191-leap-year/examples/verify.py          # cache-only
    python studies/191-leap-year/examples/verify.py --fetch  # (no-op: Shiller is staged)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from leap_year import data, strategy as st  # noqa: E402


def main() -> None:
    # ------------------------------------------------------------------
    # Modern era (GSPC since 1928)
    # ------------------------------------------------------------------
    df = data.load_shiller(start_year=1928)
    fp = data.fingerprint(df)
    print(f"Shiller S&P 500: {int(df.index.min())} to {int(df.index.max())}  "
          f"fp={fp}  n={len(df)} years")
    print()

    s = st.summarize(df)

    print("=== PRIMARY: Leap year vs non-leap year (1928-2025) ===")
    print(
        f"n_leap={s['n_leap']}  mean_leap={s['mean_leap_pct']:+.2f}%  "
        f"n_non_leap={s['n_non_leap']}  mean_non_leap={s['mean_non_leap_pct']:+.2f}%"
    )
    print(
        f"contrast={s['contrast_pct']:+.2f}pp  "
        f"Welch-t={s['tstat_welch']:+.3f}  p={s['pval_welch']:.4f}  "
        f"HAC-t(leap)={s['tstat_leap_hac']:+.3f}"
    )
    print(f"Unconditional mean (all years): {s['mean_all_pct']:+.2f}%")
    print()

    print("=== PRESIDENTIAL-CYCLE BREAKDOWN ===")
    tc = st.term_cycle_comparison(df)
    print(tc.to_string())
    print()

    print("=== MULTIPLE COMPARISONS (Bonferroni k=3) ===")
    mc = st.multiple_comparisons(df)
    print(mc.to_string(index=False))
    print(f"\nBonferroni-corrected p for leap-year hypothesis: {s['pval_leap_bonf']:.4f}")
    print()

    # ------------------------------------------------------------------
    # Sub-period: pre/post 1980
    # ------------------------------------------------------------------
    for label, mask_fn in [("Pre-1980", lambda d: d[d.index < 1980]),
                            ("Post-1980", lambda d: d[d.index >= 1980])]:
        sub = mask_fn(df)
        r = st.leap_comparison(sub)
        print(
            f"{label}: n_leap={r['n_leap']}  contrast={r['contrast_pct']:+.2f}pp  "
            f"Welch-t={r['tstat_welch']:+.3f}  p={r['pval_welch']:.4f}"
        )
    print()

    # ------------------------------------------------------------------
    # Full Shiller era (1872-2025) — longer but noisier
    # ------------------------------------------------------------------
    df_full = data.load_shiller(start_year=1872)
    s_full = st.summarize(df_full)
    print(f"=== FULL SHILLER ERA (1872-2025, n_leap={s_full['n_leap']}) ===")
    print(
        f"contrast={s_full['contrast_pct']:+.2f}pp  "
        f"Welch-t={s_full['tstat_welch']:+.3f}  p={s_full['pval_welch']:.4f}"
    )


if __name__ == "__main__":
    main()
