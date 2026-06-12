"""Real-tape verification — Study 79 (Sleigh-Ride). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the ^GSPC and SPY daily tapes, runs the Santa Claus Rally
window analysis, measures the mean vs baseline, the 'If Santa Fails' follow-up claim,
and the random-window baseline. Network is touched only with --fetch.

    python studies/79-sleigh-ride/examples/verify.py            # cache-only
    python studies/79-sleigh-ride/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sleigh_ride import data, strategy as st  # noqa: E402

TICKERS = ["^GSPC", "SPY"]
STARTS = {"^GSPC": "1950-01-01", "SPY": "1993-01-29"}


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, start=STARTS[t], fetch=True)
            print(f"{t:6s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    bars = data.fetch_daily("^GSPC", fetch=False)
    bars_spy = data.fetch_daily("SPY", fetch=False)

    print("=== ^GSPC (1950-2026): Santa window summary ===")
    ws = st.santa_window_summary(bars)
    print(
        f"n_windows={ws['n_windows']}  mean={ws['mean_window_bps']:+.1f}bps  "
        f"median={ws['median_window_bps']:+.1f}bps  "
        f"win_rate={ws['win_rate']:.1%}  t={ws['tstat_vs_zero']:+.2f}"
    )

    print("\n=== ^GSPC: daily return comparison Santa vs baseline ===")
    res = st.compare_santa_vs_baseline(bars)
    print(
        f"Santa   n={res['santa_n']}  mean={res['santa_mean_bps']:+.2f}bps  t={res['santa_tstat']:+.2f}"
    )
    print(
        f"Baseline n={res['base_n']}  mean={res['base_mean_bps']:+.2f}bps  t={res['base_tstat']:+.2f}"
    )
    print(f"Diff={res['diff_bps']:+.2f}bps  diff_t={res['diff_tstat']:+.2f}")

    print("\n=== ^GSPC: 'If Santa Fails' forward-return test ===")
    isf = st.if_santa_fails(bars)
    print(
        f"After positive Santa: n={isf['n_pos_santa']}  fwd mean={isf['fwd_mean_after_pos_bps']:+.1f}bps"
    )
    print(
        f"After negative Santa: n={isf['n_neg_santa']}  fwd mean={isf['fwd_mean_after_neg_bps']:+.1f}bps"
    )
    print(
        f"Diff={isf['fwd_diff_bps']:+.1f}bps  t={isf['fwd_diff_tstat']:+.2f}  "
        f"neg-santa forward win-rate={isf['neg_santa_win_rate_fwd']:.1%}"
    )

    print("\n=== ^GSPC: random-window baseline (1000 draws of 7 consecutive bdays) ===")
    baseline = st.random_window_baseline(bars)
    print(
        f"mean={baseline['baseline_mean_bps']:+.1f}bps  "
        f"std={baseline['baseline_std_bps']:.1f}bps  "
        f"p95={baseline['baseline_p95_bps']:+.1f}bps  "
        f"win_rate={baseline['baseline_win_rate']:.1%}"
    )

    print("\n=== SPY (1993-2026): Santa window summary ===")
    ws_spy = st.santa_window_summary(bars_spy)
    print(
        f"n_windows={ws_spy['n_windows']}  mean={ws_spy['mean_window_bps']:+.1f}bps  "
        f"win_rate={ws_spy['win_rate']:.1%}  t={ws_spy['tstat_vs_zero']:+.2f}"
    )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
