"""Real-data verification -- Study 214 (Magazine-Cover-Curse). Regenerates docs/results.md numbers.

Fetches monthly ^GSPC prices from yfinance (cache-first, study-local cache),
computes 6-month and 12-month forward returns for each hardcoded cover event,
and runs the full event-study analysis.

Run from anywhere:

    python studies/214-magazine-cover-curse/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from magazine_cover_curse import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 214 -- Magazine-Cover-Curse -- real-data verification\n")
    print("Data: yfinance ^GSPC monthly (study-local _cache/gspc_monthly.parquet)")
    print("      Cover table: hardcoded in data.py\n")

    covers = data.cover_table()
    print(f"Cover table: {len(covers)} covers, {covers['tone'].value_counts().to_dict()}")
    print(f"Date range: {covers['date'].min().date()} to {covers['date'].max().date()}\n")

    prices = data.fetch_gspc_monthly()
    print(f"GSPC monthly: {len(prices)} months, {prices.index[0].date()} to {prices.index[-1].date()}")

    df = data.compute_forward_returns(covers, prices, horizons_months=[6, 12])
    fp = data.fingerprint(df, col="fwd_12m")

    print(f"Forward return table: {len(df)} rows, {df['fwd_12m'].notna().sum()} valid 12-month windows")
    print(f"Fingerprint: {fp}\n")

    s = st.summarize(df, n_permutations=10_000, seed=214)

    print("=== 12-month forward returns ===")
    print(f"  Doom covers:     n={s['n_doom']}, mean fwd={s['mean_doom_12m']:+.1f}%,"
          f" up-rate={s['doom_up_rate_12m']:.1f}%")
    print(f"  Euphoric covers: n={s['n_euphoric']}, mean fwd={s['mean_euph_12m']:+.1f}%,"
          f" up-rate={s['euph_up_rate_12m']:.1f}%")
    print(f"  Unconditional:   mean fwd={s['mean_all_12m']:+.1f}%,"
          f" up-rate={s['uncond_up_rate_12m']:.1f}%")
    print(f"  Doom - Euph diff: {s['obs_diff_12m']:+.1f}pp")
    print(f"  Contrarian correct rate: {s['contrarian_correct_12m']:.1f}%")
    print(f"  Welch t (doom vs euph): {s['welch_t_12m']:+.3f}, p={s['welch_p_12m']:.4f}")
    print(f"  Binomial p (doom up-rate vs base): {s['binom_p_12m']:.4f}")
    print(f"  Permutation p: {s['perm_p_12m']:.4f}\n")

    print("=== 6-month forward returns ===")
    print(f"  Doom covers:     n={s['n_doom']}, mean fwd={s['mean_doom_6m']:+.1f}%")
    print(f"  Euphoric covers: n={s['n_euphoric']}, mean fwd={s['mean_euph_6m']:+.1f}%")
    print(f"  Welch t: {s['welch_t_6m']:+.3f}, p={s['welch_p_6m']:.4f}")
    print(f"  Permutation p: {s['perm_p_6m']:.4f}\n")

    print("=== Individual cover breakdown (12-month) ===")
    for _, row in df.iterrows():
        fwd = row.get("fwd_12m", float("nan"))
        fwd_str = f"{fwd*100:+.1f}%" if not (fwd != fwd) else "N/A"
        match = ""
        if not (fwd != fwd):
            if row["tone"] == "doom" and fwd > 0:
                match = "CORRECT"
            elif row["tone"] == "euphoric" and fwd < 0:
                match = "CORRECT"
            else:
                match = "WRONG"
        print(f"  [{row['tone'][:4]}] {str(row['date'])[:7]} {row['magazine'][:20]:20s}"
              f" {fwd_str:>8s}  {match}")

    print(f"\nFingerprint: {fp}")
    print(f"\nVerdict: Signal=NONE, Tradability=MIRAGE -- cherry-picked anecdotes dress up as a rule.")


if __name__ == "__main__":
    main()
