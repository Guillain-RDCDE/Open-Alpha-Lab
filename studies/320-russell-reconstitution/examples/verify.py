#!/usr/bin/env python
"""Reproducible headline run for Study 320 (Russell-Reconstitution).

Prints the fingerprinted real-tape numbers that back docs/results.md. Cache-only by
default (no network); pass --fetch to refresh the IWM cache from Yahoo!.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # refresh IWM cache, then run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from russell_reconstitution import data, strategy as st  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="refresh the IWM cache first")
    ap.add_argument("--ticker", default="IWM")
    args = ap.parse_args()

    bars = data.fetch_daily(args.ticker, fetch=args.fetch)
    print(f"{args.ticker}: {bars.index.min().date()} -> {bars.index.max().date()} "
          f"({len(bars)} days)  fingerprint {data.fingerprint(bars)}")

    win = st.build_windows(bars, runup_days=5)
    print(f"reconstitution events: {len(win)} ({win['year'].min()}-{win['year'].max()})")

    s = st.summarize(bars, runup_days=5, n_permutations=10_000, seed=320)
    print("\n--- 5-session run-up vs unconditional baseline ---")
    print(f"  n                 : {s['n']}")
    print(f"  run-up mean        : {s['runup_mean_pct']:+.4f}%   baseline {s['base_mean_pct']:+.4f}%")
    print(f"  excess             : {s['excess_bps']:+.2f} bps")
    print(f"  up-rate            : {s['up_rate_pct']:.1f}% (base {s['base_up_rate_pct']:.1f}%)")
    print(f"  one-sample t vs base: {s['t_vs_base']:+.3f}")
    print(f"  HAC t (excess)     : {s['hac_t_excess']:+.3f}   <- the |t|>=2 hurdle")
    print(f"  permutation p      : {s['perm_p']:.4f}")
    print(f"  reconstitution-day : {s['event_day_mean_pct']:+.4f}% (HAC t {s['event_day_hac_t']:+.3f})")
    print(f"  give-back (post)   : {s['post_mean_pct']:+.4f}% (HAC t {s['post_hac_t']:+.3f})")
    print(f"  net per event      : {s['net_mean_per_event_bps']:+.2f} bps")

    print("\nVerdict: Signal NONE | Tradability MIRAGE | Front-runnable? NOT SUPPORTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
