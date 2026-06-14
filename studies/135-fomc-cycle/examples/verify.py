"""Real-tape verification — Study 135 (FOMC-Cycle). Regenerates docs/results.md numbers.

Loads (or refreshes) the SPY daily total-return series and the hardcoded FOMC meeting
schedule, assigns each trading day its cycle week (0-5), and runs the CMV even-vs-odd
test plus the post-publication decay split. Network is touched only with --fetch.

    python studies/135-fomc-cycle/examples/verify.py            # cache-only
    python studies/135-fomc-cycle/examples/verify.py --fetch    # refresh SPY prices
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fomc_cycle import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    frame, fomc_dates = data.fetch_panel(fetch=fetch)
    fp = data.fingerprint(frame)
    print(f"SPY daily returns  {frame.index[0].date()} to {frame.index[-1].date()}")
    print(f"Fingerprint: {fp}  n={len(frame):,}")
    print()

    # --- Even vs Odd ---
    tbl = st.even_odd_table(frame)
    print("=== Even-week vs Odd-week (full sample 1994-2026) ===")
    print(f"Even  n={tbl['even']['n']:5d}  mean={tbl['even']['mean_bps']:+6.2f}bps  "
          f"t={tbl['even']['tstat']:+5.2f}  sharpe_ann={tbl['even']['sharpe_ann']:+5.2f}")
    print(f"Odd   n={tbl['odd']['n']:5d}  mean={tbl['odd']['mean_bps']:+6.2f}bps  "
          f"t={tbl['odd']['tstat']:+5.2f}  sharpe_ann={tbl['odd']['sharpe_ann']:+5.2f}")
    print(f"All   n={tbl['all']['n']:5d}  mean={tbl['all']['mean_bps']:+6.2f}bps  "
          f"t={tbl['all']['tstat']:+5.2f}  sharpe_ann={tbl['all']['sharpe_ann']:+5.2f}")
    print(f"Diff  {tbl['diff_bps']:+.2f}bps   Welch-t={tbl['welch_t']:+.2f}")
    print()

    # --- Week by week ---
    print("=== Week-by-week breakdown ===")
    wbw = st.week_by_week(frame)
    for w, row in wbw.iterrows():
        tag = " (EVEN)" if row["is_even"] else " (odd) "
        print(f"Week {w}{tag}  n={row['n']:5d}  mean={row['mean_bps']:+6.2f}bps  "
              f"t={row['tstat']:+5.2f}")
    print()

    # --- Pre vs Post publication ---
    era = st.split_by_era(frame)
    pre = era["pre_publication"]
    post = era["post_publication"]
    print("=== Pre- vs Post-publication decay (cut: 2019-01-01) ===")
    print(f"Pre-2019  diff={pre['diff_bps']:+.2f}bps  Welch-t={pre['welch_t']:+.2f}  "
          f"n_even={pre['n_even']}  n_odd={pre['n_odd']}")
    print(f"Post-2019 diff={post['diff_bps']:+.2f}bps  Welch-t={post['welch_t']:+.2f}  "
          f"n_even={post['n_even']}  n_odd={post['n_odd']}")
    print()

    # --- Placebo ---
    plac = st.random_week_split(frame, n_seeds=500)
    print("=== Placebo (random week permutation) ===")
    print(f"Real gap:    {plac['real_gap_bps']:+.2f}bps")
    print(f"Placebo:     mean={plac['placebo_mean_bps']:+.2f}bps  "
          f"std={plac['placebo_std_bps']:.2f}bps  p-value={plac['p_value']:.3f}")
    print()

    # --- Cost sweep ---
    cs = st.cost_sweep(frame)
    print("=== Cost sweep (long even weeks only) ===")
    for c, row in cs.iterrows():
        print(f"cost={c:.1f}bps  gross={row['gross_ann_pct']:+.1f}%/yr  "
              f"net={row['net_ann_pct']:+.1f}%/yr")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
