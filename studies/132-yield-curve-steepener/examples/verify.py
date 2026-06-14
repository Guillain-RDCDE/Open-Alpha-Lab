"""Real-tape verification — Study 132 (Yield-Curve-Steepener).
Regenerates docs/results.md numbers.

Fetches (or reads from cache) the daily yield-curve / bond-ETF tape (TLT, IEF, SHY,
^TNX, ^IRX), runs the Q5-Q1 quintile-spread test and the binary timing overlay, and
computes regime-conditioned Sharpe statistics.  Network is touched only with --fetch.

    python studies/132-yield-curve-steepener/examples/verify.py            # cache-only
    python studies/132-yield-curve-steepener/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from yield_curve_steepener import data, strategy as st  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "_cache")


def main(fetch: bool) -> None:
    if fetch:
        df = data.fetch_daily(fetch=True, cache_dir=CACHE_DIR)
        print(f"Fetched {len(df)} rows: {df.index[0].date()} → {df.index[-1].date()}")
        print(f"Fingerprint (TLT_close): {data.fingerprint(df)}")
        print()
    else:
        df = data.fetch_daily(fetch=False, cache_dir=CACHE_DIR)

    print(f"=== tape: {df.index[0].date()} to {df.index[-1].date()}, n={len(df)}, fp={data.fingerprint(df)} ===")
    print(f"Slope: mean={df['slope'].mean():.3f} pp, std={df['slope'].std():.3f} pp")
    print(f"Steep (slope>0): {(df['slope']>0).mean():.1%}  |  Inverted/flat: {(df['slope']<=0).mean():.1%}")
    print()

    print("=== Q5-Q1 quintile spread (the central test) ===")
    for h in [1, 5, 21]:
        r = st.top_minus_bottom(df, horizon=h)
        print(
            f"  horizon={h:2d}d: n_top={r['n_top']} n_bot={r['n_bot']} "
            f"spread={r['spread_bps']:+.2f}bps  t_spread={r['t_spread']:+.2f}  "
            f"(Q5={r['mean_top_bps']:+.2f}  Q1={r['mean_bot_bps']:+.2f})"
        )

    print()
    print("=== binary timing overlay (threshold=0) ===")
    for cost in [0.0, 2.0, 5.0]:
        ov = st.timing_overlay(df, threshold=0.0, cost_bps=cost)
        a = st.summarize(ov["r_active"], label=f"active cost={cost}")
        p = st.summarize(ov["r_passive"], label="passive")
        spr = st.summarize(ov["r_spread"])
        print(
            f"  cost={cost:.1f}bps  active: {a['mean_bps']:+.3f}bps/d t={a['t_stat']:+.2f} Sharpe={a['sharpe_ann']:+.3f}  "
            f"spread: {spr['mean_bps']:+.3f}bps/d t={spr['t_stat']:+.2f}"
        )
    ov_ref = st.timing_overlay(df, threshold=0.0, cost_bps=2.0)
    print(f"  Switches per year: {ov_ref['switched'].sum() / (len(df)/252):.1f}")

    print()
    print("=== regime stats ===")
    rs = st.regime_stats(df)
    for _, row in rs.iterrows():
        print(
            f"  {row['regime']:30s}: n={row['n']:5.0f} ({row['n_pct']:.1f}%)  "
            f"mean={row['mean_bps']:+.3f}bps  Sharpe_ann={row['sharpe_ann']:+.3f}  t={row['t_stat']:+.2f}"
        )

    print()
    print("=== quintile table, horizon=1 ===")
    tbl = st.quintile_table(df, horizons=[1])
    for _, row in tbl.iterrows():
        print(
            f"  Q{row['quintile']:.0f}: n={row['n']:5.0f}  mean={row['mean_bps']:+.3f}bps  t={row['t_stat']:+.2f}"
        )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
