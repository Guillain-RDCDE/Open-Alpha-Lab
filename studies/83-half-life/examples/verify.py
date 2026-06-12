"""Real-tape verification — Study 83 (Half-Life). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the BTC-USD daily tape, runs the post-halving window study
for 90-day, 180-day, and 365-day windows against a random-day control, computes secular
excess returns, and reports the p-value / power analysis. Network is touched only
with --fetch.

    python studies/83-half-life/examples/verify.py            # cache-only
    python studies/83-half-life/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from half_life import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    bars = data.fetch_daily(fetch=fetch)
    if fetch:
        print(f"BTC-USD {bars.index[0].date()}..{bars.index[-1].date()} "
              f"fp={data.fingerprint(bars)}  n={len(bars)} days\n")

    # The halvings available in Yahoo's BTC history (from 2014-09-17)
    halvings_in_range = [h for h in data.HALVING_DATES if bars.index[0] <= h <= bars.index[-1]]
    print(f"Halvings in Yahoo BTC range: {len(halvings_in_range)}")
    for h in halvings_in_range:
        print(f"  {h.date()}")
    print()

    for win in (90, 180, 365):
        print(f"=== {win}-day forward window ===")
        ev = st.window_returns(bars, halvings=halvings_in_range, window_days=win)
        print(f"  n events: {len(ev)}")
        for _, row in ev.iterrows():
            print(f"  {row['halving_date'].date()} entry {row['entry_date'].date()} "
                  f"exit {row['exit_date'].date()}  log_ret={row['log_ret']:+.3f} "
                  f"({row['pct_ret']:+.0%})")
        s = st.summarize(ev["log_ret"].to_numpy(), label=f"{win}d windows")
        print(f"  mean={s['mean']:+.3f}  std={s['std']:.3f}  HAC t={s['tstat']:+.3f}")
        print(f"  LOW POWER WARNING: {s['low_power_warning']} (n={s['n']})")

        pa = st.power_analysis(bars, halvings=halvings_in_range, window_days=win, n_draws=2000)
        print(f"  p-value vs random: {pa['p_value']:.3f}  "
              f"ctrl p90={pa['ctrl_p90']:+.3f}  ctrl p95={pa['ctrl_p95']:+.3f}")

        ex = st.secular_excess(bars, halvings=halvings_in_range, window_days=win)
        print(f"  secular excess mean={ex['excess_log_ret'].mean():+.3f}  "
              f"(trend={ex['trend_log_ret'].mean():+.3f}/window)")
        print()

    print("=== Secular trend vs post-halving excess (365d) ===")
    ex = st.secular_excess(bars, halvings=halvings_in_range, window_days=365)
    for _, row in ex.iterrows():
        print(f"  {row['halving_date'].date()}  raw={row['log_ret']:+.3f}  "
              f"trend={row['trend_log_ret']:+.3f}  excess={row['excess_log_ret']:+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
