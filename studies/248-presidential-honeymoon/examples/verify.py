"""Real-tape verification — Study 248 (Presidential-Honeymoon). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the ^GSPC daily series back to 1928, assigns 100-day honeymoon
windows around each inauguration, and prints the paired honeymoon vs control stats.
Network is touched only with --fetch.

    python studies/248-presidential-honeymoon/examples/verify.py            # cache-only
    python studies/248-presidential-honeymoon/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from presidential_honeymoon import data, strategy as st  # noqa: E402

TICKER = "^GSPC"


def main(fetch: bool) -> None:
    df = data.fetch_prices(TICKER, fetch=fetch)
    fp = data.fingerprint(df)
    start = df.index[0].date()
    end = df.index[-1].date()
    n_days = len(df)
    print(f"{TICKER}: {start} to {end}  n={n_days} days  fingerprint={fp}")
    print()

    periods = st.period_returns(df)
    stats = st.honeymoon_stats(periods)
    fy = st.first_year_stats(df)

    print("=== Per-inauguration honeymoon vs control ===")
    print(f"{'President':<20} {'Year':>4} | {'HM ret':>8} | {'Ctrl ret':>8} | {'HM wins':>7}")
    print("-" * 60)
    for inau_date, row in periods.iterrows():
        wins = "yes" if row["hm_ret"] > row["rest_ret"] else "no"
        print(
            f"  {row['president']:<18} {inau_date.year:>4} | "
            f"{row['hm_ret']*100:>+8.1f} | {row['rest_ret']*100:>+8.1f} | {wins:>7}"
        )
    print()

    print("=== Headline test: honeymoon vs control (paired) ===")
    print(f"  n inaugurations:        {stats['n']}")
    print(f"  Honeymoon mean (100d):  {stats['hm_mean']:>+7.2f}%  HAC t = {stats['hm_tstat']:>+.2f}")
    print(f"  Control mean (265d):    {stats['rest_mean']:>+7.2f}%  HAC t = {stats['rest_tstat']:>+.2f}")
    print(f"  Paired diff (HM-Ctrl):  {stats['diff_mean']:>+7.2f} pp  HAC t = {stats['diff_tstat']:>+.2f}")
    print(f"  Positive rate:          {stats['pos_rate']:.0%}")
    print()

    print("=== First-year vs second-year extended test ===")
    print(f"  n terms: {fy.get('n_terms', 0)}")
    print(f"  Year-1 mean: {fy.get('yr1_mean', float('nan')):>+.2f}%")
    print(f"  Year-2 mean: {fy.get('yr2_mean', float('nan')):>+.2f}%")
    print(f"  Diff (yr1-yr2): {fy.get('diff_mean', float('nan')):>+.2f} pp  HAC t = {fy.get('diff_tstat', float('nan')):>+.2f}")
    print()

    # Post-2001 subsample
    df_post = df[df.index.year >= 2001]
    periods_post = st.period_returns(df_post)
    stats_post = st.honeymoon_stats(periods_post)
    print("=== Post-2001 subsample ===")
    print(f"  n terms: {stats_post['n']}")
    print(f"  HM mean: {stats_post['hm_mean']:>+.2f}%  Ctrl mean: {stats_post['rest_mean']:>+.2f}%")
    print(f"  Diff: {stats_post['diff_mean']:>+.2f} pp  HAC t = {stats_post['diff_tstat']:>+.2f}")
    print()

    print(f"fingerprint: {fp}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
