"""Real-tape verification -- Study 246 (Defensive-Sectors Leadership).
Regenerates docs/results.md numbers.

Fetches (or reads from cache) the XLP/XLU/SPY daily tape, runs the defensive-sectors
analysis: quintile-sorted forward returns, top-minus-bottom spreads at 1d/5d/21d,
regime stats, and a timing-overlay cost sweep. Network is touched only with --fetch.

    python studies/246-defensive-sectors/examples/verify.py            # cache-only
    python studies/246-defensive-sectors/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from defensive_sectors import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
    if fetch:
        df = data.fetch_daily(fetch=True, cache_dir=cache_dir)
        print(f"XLP/XLU/SPY {df.index[0].date()} .. {df.index[-1].date()}  fp={data.fingerprint(df)}")
        print(f"n = {len(df)}")
        print()
    else:
        df = data.fetch_daily(fetch=False, cache_dir=cache_dir)

    print(f"=== Data: {df.index[0].date()} to {df.index[-1].date()} (n={len(df)}) ===")
    print(f"Fingerprint (SPY_close): {data.fingerprint(df)}")
    print()

    # Quintile table
    qt = st.quintile_table(df, horizons=[1, 5, 21])
    print("=== Quintile table (Q1=SPY outperforming/calm, Q5=defensives outperforming/alert) ===")
    for h in [1, 5, 21]:
        sub = qt[qt["horizon"] == h]
        print(f"\nhorizon={h}d:")
        for _, r in sub.iterrows():
            print(f"  Q{int(r['quintile'])}: n={r['n']:5.0f}  mean={r['mean_bps']:+7.2f}bps  t={r['t_stat']:+.2f}")

    # Key spreads
    print()
    print("=== Q1-calm vs Q5-alert spread (decisive test) ===")
    for h in [1, 5, 21]:
        r = st.top_minus_bottom(df, horizon=h)
        print(
            f"h={h:2d}d: Q1-calm={r['mean_calm_bps']:+.2f}bps  Q5-alert={r['mean_alert_bps']:+.2f}bps  "
            f"spread={r['spread_bps']:+.2f}bps  t={r['t_spread']:+.2f}"
        )

    # Regime stats
    print()
    print("=== Regime stats (prior-day combined RS > median = alert) ===")
    rs = st.regime_stats(df)
    for _, r in rs.iterrows():
        print(
            f"  {r['regime']:26s}: n={r['n']:5d} ({r['n_pct']:4.1f}%)  "
            f"mean={r['mean_bps']:+.2f}bps  sharpe={r['sharpe_ann']:+.3f}  t={r['t_stat']:+.2f}"
        )

    # Timing overlay
    print()
    print("=== Timing overlay (go to cash when top 20% defensive RS alert) ===")
    ov_gross = st.timing_overlay(df, cost_bps=0.0)
    spr_g = st.summarize(ov_gross["r_spread"])
    act_g = st.summarize(ov_gross["r_active"])
    pas_g = st.summarize(ov_gross["r_passive"])
    print(f"  passive (buy-and-hold): sharpe={pas_g['sharpe_ann']:+.3f}  mean={pas_g['mean_bps']:+.2f}bps")
    print(f"  active  (gross):        sharpe={act_g['sharpe_ann']:+.3f}  mean={act_g['mean_bps']:+.2f}bps")
    print(f"  spread  (gross):        mean={spr_g['mean_bps']:+.2f}bps  t={spr_g['t_stat']:+.2f}")

    print()
    print("=== Cost sweep (timing overlay) ===")
    for c in [0.0, 0.5, 1.0, 2.0, 5.0]:
        ov = st.timing_overlay(df, cost_bps=c)
        spr = st.summarize(ov["r_spread"])
        act = st.summarize(ov["r_active"])
        print(
            f"  cost={c:.1f}bps: spread={spr['mean_bps']:+.2f}bps  t={spr['t_stat']:+.2f}  "
            f"active_sharpe={act['sharpe_ann']:+.3f}"
        )

    n_switches = ov_gross["switched"].sum()
    n_alerts = ov_gross["is_alert"].fillna(False).sum()
    print(f"\n  alert days: {n_alerts} ({100*n_alerts/len(ov_gross):.1f}%)  switches: {n_switches}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
