"""Real-tape verification — Study 151 (Stocks-For-Long-Run). Regenerates docs/results.md numbers.

Reads the Shiller S&P 500 monthly panel from the staged parquet (cache only; no network),
computes rolling real equity and real bond returns across five horizons (1/5/10/20/30 years),
checks the worst-case windows, and prints the headline numbers that populate docs/results.md.

    python studies/151-stocks-for-long-run/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stocks_for_long_run import data, strategy as st  # noqa: E402


def main() -> None:
    df = data.load_shiller()
    fp = data.fingerprint(df)
    print(f"Shiller panel: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"Rows: {len(df)}  |  Fingerprint (real_tr): {fp}")
    print()

    print("=== Rolling-horizon summary (Shiller 1871–2023) ===")
    print(
        f"{'Horizon':>8} | {'N windows':>10} | {'Eq mean%':>9} {'Eq worst%':>10} {'Eq neg%':>8} "
        f"| {'Ex mean%':>9} {'Ex worst%':>10} {'Ex neg%':>8} | {'HAC t':>7}"
    )
    print("-" * 105)
    for h in data.HORIZONS:
        r = st.worst_case_windows(df, h)
        print(
            f"  {h:>6}y | {r['n_windows']:>10,} | {r['eq_mean']*100:>+9.2f} {r['eq_min']*100:>+10.2f} "
            f"{r['eq_pct_neg']:>8.2f} "
            f"| {r['excess_mean']*100:>+9.2f} {r['excess_min']*100:>+10.2f} {r['excess_pct_neg']:>8.2f} "
            f"| {r['tstat_excess']:>+7.2f}"
        )

    print()
    print("=== Decade-by-decade breakdown (non-overlapping 10-year windows) ===")
    dec = st.decade_returns(df)
    print(f"{'Decade':>8} | {'Eq ann real%':>12} | {'Bd ann real%':>12} | {'Excess%':>8}")
    print("-" * 50)
    for decade, row in dec.iterrows():
        print(
            f"  {decade}s | {row['eq_ann_real']*100:>+12.2f} | "
            f"{row['bd_ann_real']*100:>+12.2f} | {row['excess']*100:>+8.2f}"
        )
    n_dec = len(dec)
    n_neg = (dec["excess"] < 0).sum()
    print(f"\n  {n_dec} decades: {n_neg} with bonds beating stocks ({n_neg/n_dec*100:.0f}%)")
    print()

    # Key worst-case claims
    r20 = st.worst_case_windows(df, 20)
    r30 = st.worst_case_windows(df, 30)
    print("=== Siegel's headline claims vs the data ===")
    print(
        f"Claim 1 (no negative 20y real equity): "
        f"WORST CASE = {r20['eq_min']*100:+.2f}%/yr "
        f"({r20['eq_pct_neg']:.2f}% of windows negative)"
    )
    print(
        f"Claim 2 (no negative 30y real equity): "
        f"WORST CASE = {r30['eq_min']*100:+.2f}%/yr "
        f"({r30['eq_pct_neg']:.2f}% of windows negative)"
    )
    print(
        f"Claim 3 (equities always beat bonds 20y): "
        f"WORST EXCESS = {r20['excess_min']*100:+.2f}%/yr "
        f"({r20['excess_pct_neg']:.2f}% of windows bonds winning)"
    )
    print(
        f"Claim 4 (equities always beat bonds 30y): "
        f"WORST EXCESS = {r30['excess_min']*100:+.2f}%/yr "
        f"({r30['excess_pct_neg']:.2f}% of windows bonds winning)"
    )


if __name__ == "__main__":
    main()
