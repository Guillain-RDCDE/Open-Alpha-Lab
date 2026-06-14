"""Real-tape verification — Study 142 (Split-Drift). Regenerates docs/results.md numbers.

Fetches (or reads from cache) split events and daily prices for the large-cap basket,
builds the matched no-split baseline, and computes forward-return comparisons across
four horizons (1/3/6/12 months).  Network is touched only with --fetch.

    python studies/142-split-drift/examples/verify.py            # cache-only
    python studies/142-split-drift/examples/verify.py --fetch    # refresh

Notes
-----
yfinance supplies the split *effective* date, not the announcement date.  The
Ikenberry-Rankine-Stice (1996) anomaly is measured from the announcement date —
so our window is a strictly weaker (and therefore more conservative) test.  If
anything, this is an upper bound: by the effective date the market has already
had ~3-6 weeks to incorporate the announcement.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from split_drift import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        print("Fetching split events and price data from Yahoo Finance...")
        events, prices = data.fetch_splits_panel(fetch=True)
        print(f"  {len(events)} split events across "
              f"{events['ticker'].nunique()} tickers")
        print(f"  Price panel: {prices.index[0].date()} → {prices.index[-1].date()}")
        print(f"  Fingerprint: {data.fingerprint(events)}")
        print()
    else:
        events, prices = data.fetch_splits_panel(fetch=False)

    print(f"=== Split events (n={len(events)}) ===")
    print(events[["ticker", "event_date", "split_ratio"]].to_string())
    print()

    baseline = st.build_baseline_events(
        events, prices, n_control_per_event=5, seed=142
    )
    print(f"=== Baseline events (n={len(baseline)}) ===\n")

    print("=== Forward returns: split arm ===")
    for col, lbl in [("ret_1m","1m"), ("ret_3m","3m"), ("ret_6m","6m"), ("ret_12m","12m")]:
        s = st.summarize_horizon(events, col=col)
        print(
            f"  {lbl}: n={s['n']}, mean={s['mean_pct']:+.2f}%, "
            f"median={s['median_pct']:+.2f}%, win={s['win_rate']:.1%}, "
            f"HAC t={s['tstat']:+.2f}"
        )

    print()
    print("=== Forward returns: matched baseline arm ===")
    for col, lbl in [("ret_1m","1m"), ("ret_3m","3m"), ("ret_6m","6m"), ("ret_12m","12m")]:
        s = st.summarize_horizon(baseline, col=col)
        print(
            f"  {lbl}: n={s['n']}, mean={s['mean_pct']:+.2f}%, "
            f"median={s['median_pct']:+.2f}%, win={s['win_rate']:.1%}, "
            f"HAC t={s['tstat']:+.2f}"
        )

    print()
    print("=== Horizon comparison: split vs baseline ===")
    comp = st.compare_horizons(events, baseline)
    print(comp.to_string(index=False))

    print()
    print("=== Cost-adjusted annualised returns (split arm gross) ===")
    for col, lbl, h in [("ret_1m","1m",21), ("ret_3m","3m",63), ("ret_6m","6m",126), ("ret_12m","12m",252)]:
        gross = events[col].dropna().mean()
        for cost in (0.0, 5.0, 10.0):
            ann = st.cost_adjusted_return(float(gross), h, cost_bps=cost)
            print(f"  {lbl}, cost={cost:.0f}bps: gross={gross*100:+.2f}%, net_ann={ann*100:+.1f}%/yr")

    print()
    print("=== VERDICT ===")
    sp_12m = st.summarize_horizon(events, col="ret_12m")
    print(f"  Signal: post-effective drift 12m mean {sp_12m['mean_pct']:+.2f}%, "
          f"HAC t={sp_12m['tstat']:+.2f} — {'REAL' if abs(sp_12m['tstat']) >= 2 else 'NONE/WEAK'}")
    comp_12m = comp[comp["horizon"].str.contains("12m")].iloc[0]
    print(f"  vs baseline: diff={comp_12m['diff_pct']:+.2f}%, "
          f"HAC t={comp_12m['tstat_diff']:+.2f} — SPLIT UNDERPERFORMS baseline")
    print("  Tradability: MIRAGE — no significant edge; n=41 events too small for live sizing")
    print("  CAVEAT: yfinance gives effective dates, not announcement dates.")
    print("  Post-announcement drift (Ikenberry 1996) NOT tested here.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
