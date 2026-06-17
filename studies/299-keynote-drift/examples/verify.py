"""Real-data verification -- Study 299 (Keynote-Drift). Regenerates docs/results.md numbers.

Loads the cached AAPL daily price frame (cache-only), joins it with the hardcoded
Apple keynote table, and runs the full event study: pre/post cumulative abnormal
returns (CARs), one-sample t-tests, a permutation null, and the tradable
'buy the rumor' backtest (gross and net of costs).

The AAPL parquet lives at the study-level _cache/aapl_daily.parquet. Populate it once
with ``data.fetch_prices(fetch=True)`` (the only network call in this study). Then run:

    python studies/299-keynote-drift/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keynote_drift import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 299 -- Keynote-Drift -- real-data verification\n")
    print("Data: AAPL daily adjusted close (price-only), cache-only")
    print("      _cache/aapl_daily.parquet (fetch once with fetch_prices(fetch=True))")
    print("      Keynote dates: hardcoded in data.py (2008-2025)\n")

    px = data.fetch_prices(fetch=False)
    kt = data.keynote_table()
    fp = data.fingerprint(px)

    print(f"Data stamp: {px.index.min().date()}-{px.index.max().date()}, "
          f"rows={len(px)}, fingerprint={fp}")
    print(f"Keynotes: {len(kt)} events\n")

    cs = st.car_stats(px, kt, pre_window=5, post_window=5, n_permutations=10_000, seed=299)
    print("=== Event study (constant-mean abnormal CARs, AAPL vs its own mean) ===")
    print(f"  n usable events:        {cs['n_events']}")
    print(f"  Mean pre-keynote CAR:   {cs['mean_pre_car']*100:+.3f}%  "
          f"(t={cs['t_pre']:+.3f}, p={cs['p_pre']:.4f})")
    print(f"  Mean post-keynote CAR:  {cs['mean_post_car']*100:+.3f}%  "
          f"(t={cs['t_post']:+.3f}, p={cs['p_post']:.4f})")
    print(f"  Permutation p (pre):    {cs['perm_p']:.4f}\n")

    sd = float(np.std(cs["pre_cars"], ddof=1)) * 100
    print(f"  Per-event CAR std:      {sd:.3f}%")
    from scipy.stats import t as t_dist
    n = cs["n_events"]
    mde = (t_dist.ppf(0.975, n-1) + t_dist.ppf(0.80, n-1)) * sd / np.sqrt(n)
    print(f"  Min detectable CAR:     {mde:.3f}%/event (80% power)\n")

    bt = st.buy_the_rumor_backtest(px, kt, pre_window=5, cost_bps=5.0)
    print("=== Tradable 'buy the rumor' leg (long pre-keynote, one-day lag) ===")
    print(f"  Trades:                 {bt['n_trades']} ({bt['ann_factor']:.1f}/yr)")
    print(f"  Mean per trade -- gross: {bt['mean_gross']*100:+.3f}%  net: {bt['mean_net']*100:+.3f}%")
    print(f"  Hit-rate (gross):       {bt['hit_rate_gross']*100:.1f}%")
    print(f"  Total compounded -- gross: {bt['total_gross']*100:+.1f}%  net: {bt['total_net']*100:+.1f}%\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- a great stock's trend, "
          "mistaken for a keynote omen.")


if __name__ == "__main__":
    main()
