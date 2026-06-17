"""Real-tape verification — Study 250 (Reverse-Split). Regenerates docs/results.md numbers.

Fetches (or reads from cache) prices for hardcoded reverse-split event tickers, then
computes forward-return comparisons across four horizons (1/3/6/12 months).
Network is touched only with --fetch.

    python studies/250-reverse-split/examples/verify.py            # cache-only
    python studies/250-reverse-split/examples/verify.py --fetch    # refresh

Notes
-----
The hardcoded event table uses effective dates, not announcement dates.
The key distress confound: reverse splits cluster in financially distressed names.
Negative post-event returns may reflect ongoing fundamental deterioration rather
than a "reverse split signal" per se.  We name this explicitly.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reverse_split import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        print("Fetching price data from Yahoo Finance...")
        events, prices = data.load_events(fetch=True)
    else:
        events, prices = data.load_events(fetch=False)

    print(f"=== Reverse-split events (n={len(events)}) ===")
    print(events[["ticker", "event_date", "ratio"]].to_string())
    print()
    print(f"Fingerprint: {data.fingerprint(events)}")
    print()

    print("=== Forward returns: reverse-split arm ===")
    for col, lbl in [("ret_1m", "1m"), ("ret_3m", "3m"), ("ret_6m", "6m"), ("ret_12m", "12m")]:
        s = st.summarize_horizon(events, col=col)
        print(
            f"  {lbl}: n={s['n']}, mean={s['mean_pct']:+.2f}%, "
            f"median={s['median_pct']:+.2f}%, win={s['win_rate']:.1%}, "
            f"HAC t={s['tstat']:+.2f}"
        )

    print()
    baseline = st.build_random_baseline(events, prices, n_control_per_event=5, seed=250)
    print(f"=== Random baseline (n={len(baseline)}) ===")
    for col, lbl in [("ret_1m", "1m"), ("ret_3m", "3m"), ("ret_6m", "6m"), ("ret_12m", "12m")]:
        s = st.summarize_horizon(baseline, col=col)
        print(
            f"  {lbl}: n={s['n']}, mean={s['mean_pct']:+.2f}%, "
            f"HAC t={s['tstat']:+.2f}"
        )

    print()
    print("=== Horizon comparison: RS vs random baseline ===")
    comp = st.compare_horizons(events, baseline)
    print(comp.to_string(index=False))

    print()
    print("=== Cost-adjusted annualised returns (RS arm gross) ===")
    for col, lbl, h in [("ret_1m", "1m", 21), ("ret_3m", "3m", 63),
                         ("ret_6m", "6m", 126), ("ret_12m", "12m", 252)]:
        gross = events[col].dropna().mean()
        for cost in (0.0, 10.0, 20.0):
            ann = st.cost_adjusted_return(float(gross), h, cost_bps=cost)
            print(f"  {lbl}, cost={cost:.0f}bps: gross={gross*100:+.2f}%, net_ann={ann*100:+.1f}%/yr")

    print()
    print("=== VERDICT ===")
    sp_3m = st.summarize_horizon(events, col="ret_3m")
    sp_12m = st.summarize_horizon(events, col="ret_12m")
    print(f"  Signal: 3m mean {sp_3m['mean_pct']:+.2f}%, HAC t={sp_3m['tstat']:+.2f} "
          f"— {'REAL' if abs(sp_3m['tstat']) >= 2 else 'WEAK/NONE'}")
    print(f"  Signal: 12m mean {sp_12m['mean_pct']:+.2f}%, HAC t={sp_12m['tstat']:+.2f} "
          f"— {'REAL' if abs(sp_12m['tstat']) >= 2 else 'WEAK/NONE'}")
    comp_12m = comp[comp["horizon"].str.contains("12m")].iloc[0]
    print(f"  vs random baseline: diff={comp_12m['diff_pct']:+.2f}%, "
          f"HAC t={comp_12m['tstat_diff']:+.2f}")
    print("  CAVEAT: distress confound — negative returns may be distress continuation, not RS signal.")
    print("  Tradability: MIRAGE — n too small, costs too high, confound unnamed.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
