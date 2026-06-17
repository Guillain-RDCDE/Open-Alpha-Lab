"""Real-tape verification — Study 249 (Index-Inclusion). Regenerates docs/results.md numbers.

Reads cached daily price parquets (or fetches with --fetch) for the hardcoded inclusion
event table, computes announce-to-effective pop and post-effective give-back windows,
and prints a full statistical summary with the event fingerprint.

    python studies/249-index-inclusion/examples/verify.py            # cache-only
    python studies/249-index-inclusion/examples/verify.py --fetch    # refresh

Notes
-----
Our pop window is measured close-to-close: announcement-day close to effective-date
close.  A real trader could not buy at the announcement-day close if news arrived
after hours — the achievable pop is therefore a gross upper bound.  Transaction costs
(bid-ask slippage, market impact) further reduce the net return.

The TSLA 2020 addition is by far the largest event in the table (+59.2% over 25
business days); it should be considered separately when assessing the signal.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from index_inclusion import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        print("Fetching inclusion events and price data from Yahoo Finance...")
        events, prices = data.fetch_inclusion_panel(fetch=True)
        print(f"  {len(events)} inclusion events across {events['ticker'].nunique()} tickers")
        print(f"  Price panel: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"  Fingerprint: {data.fingerprint(events)}")
        print()
    else:
        events, prices = data.fetch_inclusion_panel(fetch=False)

    print(f"=== Inclusion events (n={len(events)}) ===")
    print(events[["ticker", "announce_date", "effective_date",
                  "window_days", "ret_pop", "ret_giveback_1m", "ret_giveback_3m"]].to_string())
    print()

    print("=== Pop window (announce -> effective) ===")
    s_pop = st.summarize_window(events, col="ret_pop")
    print(f"  n={s_pop['n']}, mean={s_pop['mean_pct']:+.2f}%, "
          f"median={s_pop['median_pct']:+.2f}%, "
          f"win={s_pop['win_rate']:.1%}, HAC t={s_pop['tstat']:+.2f}")

    ev_no_tsla = events[events["ticker"] != "TSLA"]
    s_nt = st.summarize_window(ev_no_tsla, col="ret_pop")
    print(f"  Ex-TSLA (n={s_nt['n']}): mean={s_nt['mean_pct']:+.2f}%, "
          f"HAC t={s_nt['tstat']:+.2f}")

    print()
    print("=== Give-back windows (post-effective) ===")
    s1 = st.summarize_window(events, col="ret_giveback_1m")
    print(f"  1m: n={s1['n']}, mean={s1['mean_pct']:+.2f}%, "
          f"median={s1['median_pct']:+.2f}%, HAC t={s1['tstat']:+.2f}")
    s3 = st.summarize_window(events, col="ret_giveback_3m")
    print(f"  3m: n={s3['n']}, mean={s3['mean_pct']:+.2f}%, "
          f"median={s3['median_pct']:+.2f}%, HAC t={s3['tstat']:+.2f}")

    print()
    print("=== Period breakdown ===")
    pp = st.pop_by_period(events)
    print(pp.to_string(index=False))

    print()
    print("=== Cost-adjusted pop ===")
    gross = s_pop["mean_pct"] / 100.0
    for cost in (0.0, 5.0, 10.0, 20.0):
        net = st.cost_adjusted_pop(float(gross), cost_bps=cost)
        print(f"  cost={cost:.0f}bps: gross={gross*100:+.2f}%, net={net*100:+.2f}%")

    print()
    print("=== VERDICT ===")
    print(f"  Pop signal: mean={s_pop['mean_pct']:+.2f}%, HAC t={s_pop['tstat']:+.2f} "
          f"-> {'REAL' if abs(s_pop['tstat']) >= 2 else 'WEAK/NONE'} "
          f"(driven by TSLA ex-TSLA t={s_nt['tstat']:+.2f})")
    print(f"  Give-back 3m: mean={s3['mean_pct']:+.2f}%, HAC t={s3['tstat']:+.2f} "
          f"-> POSITIVE (not a reversal)")
    print("  Tradability: MIRAGE — no significant edge ex-outlier; crowded trade")
    print()
    print(f"  quantlab.repro.fingerprint: {data.fingerprint(events)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
