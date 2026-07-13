"""Real-tape verification — Study 767 (VIX9D-Term). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the VIX9D, VIX, and SPY daily tape, computes the quintile
forward-return table, the Q5-Q1 spread, the binary timing overlay, and the regime breakdown.
Network is touched only with --fetch.

    python studies/767-vix9d-term/examples/verify.py            # cache-only
    python studies/767-vix9d-term/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vix9d_term import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_daily(fetch=fetch)
    if fetch:
        print(f"^VIX9D/^VIX/SPY  {df.index[0].date()}..{df.index[-1].date()}  fp={data.fingerprint(df)}")
        print()

    print(f"Tape: N={len(df)}, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Fraction contango (slope>0): {(df['slope']>0).mean():.3f}")
    print()

    # --- Q5-Q1 spreads ---
    print("=== Q5-Q1 spread (contango - backwardation) ===")
    for h in [1, 5, 21]:
        res = st.top_minus_bottom(df, horizon=h)
        print(f"h={h:2d}d: n_top={res['n_top']} n_bot={res['n_bot']} "
              f"mean_top={res['mean_top_bps']:+.2f}bps mean_bot={res['mean_bot_bps']:+.2f}bps "
              f"spread={res['spread_bps']:+.2f}bps t_spread={res['t_spread']:+.2f}")

    # --- Full quintile table at h=1 ---
    print("\n=== Quintile table (h=1d) ===")
    tbl = st.quintile_table(df, horizons=[1])
    for _, row in tbl.iterrows():
        print(f"Q{row['quintile']:.0f}: n={row['n']:.0f}  mean={row['mean_bps']:+.2f}bps  "
              f"t={row['t_stat']:+.2f}")

    # --- Timing overlay ---
    print("\n=== Binary timing overlay: long when contango, flat when backwardated ===")
    ov = st.timing_overlay(df, horizon=1, cost_bps=0.0)
    sa = st.summarize(ov["r_active"])
    sp = st.summarize(ov["r_passive"])
    ss = st.summarize(ov["r_spread"])
    n_switches = int(ov["switched"].sum())
    print(f"Active:  mean={sa['mean_bps']:+.2f}bps/d  Sharpe={sa['sharpe_ann']:+.2f}  t={sa['t_stat']:+.2f}")
    print(f"Passive: mean={sp['mean_bps']:+.2f}bps/d  Sharpe={sp['sharpe_ann']:+.2f}  t={sp['t_stat']:+.2f}")
    print(f"Spread:  mean={ss['mean_bps']:+.2f}bps/d  t={ss['t_stat']:+.2f}")
    print(f"Switches: {n_switches} over {len(ov)} days = {n_switches/len(ov)*252:.1f}/yr")

    print("\n=== Cost sweep (spread, active-passive) ===")
    for cost in [0.0, 0.5, 1.0, 2.0, 5.0]:
        ov2 = st.timing_overlay(df, horizon=1, cost_bps=cost)
        ss2 = st.summarize(ov2["r_spread"])
        print(f"cost={cost:.1f}bps: spread={ss2['mean_bps']:+.2f}bps/d  t={ss2['t_stat']:+.2f}")

    # --- Regime stats ---
    print("\n=== Regime statistics ===")
    rs = st.regime_stats(df)
    for _, row in rs.iterrows():
        print(f"{row['regime']:30s}: n={row['n']:.0f} ({row['n_pct']:.1f}%)  "
              f"mean={row['mean_bps']:+.2f}bps  Sharpe={row['sharpe_ann']:+.2f}  t={row['t_stat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
