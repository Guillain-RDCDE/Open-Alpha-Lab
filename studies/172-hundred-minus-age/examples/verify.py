"""Real-tape runner — Study 172 (Hundred-Minus-Age).

Uses the Shiller S&P 500 dataset (already cached at
``_cache/shiller_sp500.parquet``).  Cache-first: never touches the network.
Prints the headline numbers that mirror ``docs/results.md`` and can be used to
regenerate that file's data stamp.

Run:
    python studies/172-hundred-minus-age/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hundred_minus_age import data, strategy as st  # noqa: E402


def main() -> None:
    # Load the Shiller real return series
    ret = data.load_shiller()
    fp = data.fingerprint(ret)

    print("Study 172 — Hundred-Minus-Age — real-tape verification")
    print("=" * 60)
    print(f"Source: Shiller S&P 500 (shiller_sp500.parquet)")
    print(f"Window: {ret.index[0].date()} to {ret.index[-1].date()}")
    print(f"N months: {len(ret):,}")
    print(f"Fingerprint: {fp}")
    print()

    # Run 40-year rolling cohort analysis
    print("Running 40-year rolling cohort analysis...")
    cohorts = st.rolling_cohorts(ret, n_years=40, cost_bps=5)
    stats = st.cohort_stats(cohorts)

    print(f"Number of cohorts: {len(cohorts):,}")
    print()

    # Headline stats table
    print("Terminal wealth statistics (normalised: lifetime contributions = 1 unit)")
    print(f"{'Strategy':14s} {'Mean':>7s} {'Median':>7s} {'P05':>7s} {'P95':>7s} {'Vol':>7s} "
          f"{'Mean/Vol':>9s} {'N':>5s}")
    print("-" * 68)
    for strat in ("hma", "sixty_forty", "all_equity", "hma_rising"):
        r = stats.loc[strat]
        print(
            f"{strat:14s} {r['mean']:>7.3f} {r['median']:>7.3f} {r['p05']:>7.3f} "
            f"{r['p95']:>7.3f} {r['vol']:>7.3f} {r['mean']/r['vol']:>9.3f} "
            f"{int(r['n_cohorts']):>5d}"
        )
    print()

    # Inference
    hma = cohorts["hma"].values
    s6040 = cohorts["sixty_forty"].values
    eq = cohorts["all_equity"].values
    rising = cohorts["hma_rising"].values

    t_hma_6040 = st.hac_tstat(hma, s6040)
    t_hma_eq = st.hac_tstat(hma, eq)
    t_rising_hma = st.hac_tstat(rising, hma)
    t_rising_6040 = st.hac_tstat(rising, s6040)

    print("HAC t-stats (Newey-West, overlapping cohorts)")
    print(f"  HMA vs 60/40:      t = {t_hma_6040:+.3f}")
    print(f"  HMA vs AllEquity:  t = {t_hma_eq:+.3f}")
    print(f"  Rising vs HMA:     t = {t_rising_hma:+.3f}")
    print(f"  Rising vs 60/40:   t = {t_rising_6040:+.3f}")
    print()

    # Frequency HMA wins
    print("Cohort win rates")
    print(f"  HMA beats 60/40:   {(hma > s6040).mean():.1%}")
    print(f"  HMA beats AllEq:   {(hma > eq).mean():.1%}")
    print(f"  Rising beats HMA:  {(rising > hma).mean():.1%}")
    print()

    # Annualised real returns
    eq_ann = (1.0 + ret["EQ"].mean()) ** 12 - 1.0
    bd_ann = (1.0 + ret["BD"].mean()) ** 12 - 1.0
    print("Asset-level annualised real returns (Shiller tape)")
    print(f"  EQ (real equities):  {eq_ann:.2%}/yr")
    print(f"  BD (real bonds):     {bd_ann:.2%}/yr")
    print()

    print("Done. Reproduce docs/results.md from these numbers.")


if __name__ == "__main__":
    main()
