"""Real-tape verification — Study 147 (FX-Momentum). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the G10 FX daily panel, constructs the 12-1 momentum
portfolio (long top-3, short bottom-3), pins it against a random-ranking control,
sweeps costs, and shows a rolling Sharpe to assess decay.

    python studies/147-fx-momentum/examples/verify.py            # cache-only
    python studies/147-fx-momentum/examples/verify.py --fetch    # refresh the panel

Network is touched only with --fetch.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fx_momentum import data, strategy as st  # noqa: E402

N_RANDOM_SEEDS = 20


def main(fetch: bool) -> None:
    # Load and process data.
    daily = data.fetch_panel(fetch=fetch)
    monthly = data.to_monthly(daily)
    fp = data.fingerprint(monthly)
    print(f"Monthly panel: {monthly.index[0].date()} to {monthly.index[-1].date()}")
    print(f"  n_months={len(monthly)}, fingerprint={fp}\n")

    # Momentum ranks and portfolio.
    ranks_m = st.momentum_ranks(monthly)
    led_m = st.build_portfolio(monthly, ranks_m, cost_bps=0)
    s_m = st.summarize(led_m, "ret_gross")

    print("=== Momentum portfolio (gross, 0 cost) ===")
    print(f"  n_months={s_m['n_months']}, ann_mean={s_m['mean_ann_pct']:+.2f}%,"
          f" Sharpe={s_m['sharpe_ann']:+.2f}, t={s_m['tstat']:+.2f}")

    # Random-ranking control (average over many seeds).
    rand_means = []
    for seed in range(N_RANDOM_SEEDS):
        ranks_r = st.random_ranks(monthly, seed=seed)
        led_r = st.build_portfolio(monthly, ranks_r, cost_bps=0)
        s_r = st.summarize(led_r, "ret_gross")
        rand_means.append(s_r["mean_ann_pct"])
    rand_mean = np.mean(rand_means)
    print(f"\n=== Random-ranking control (avg over {N_RANDOM_SEEDS} seeds) ===")
    print(f"  avg ann_mean={rand_mean:+.2f}%")
    print(f"  Momentum - Random = {s_m['mean_ann_pct'] - rand_mean:+.2f}%")

    # Cost sweep.
    print("\n=== Cost sweep (momentum portfolio, net) ===")
    costs = [0, 2, 5, 10, 20]
    for c in costs:
        led = st.build_portfolio(monthly, ranks_m, cost_bps=c)
        s = st.summarize(led, "ret_net")
        print(f"  cost={c:2d} bps/leg: ann_mean={s['mean_ann_pct']:+.2f}%, t={s['tstat']:+.2f}")

    # Sub-period breakdown to show decay.
    print("\n=== Sub-period breakdown (gross) ===")
    for label, start, end in [
        ("2000-2007", "2000-01-01", "2007-12-31"),
        ("2008-2015", "2008-01-01", "2015-12-31"),
        ("2016-2026", "2016-01-01", "2026-12-31"),
    ]:
        sub = led_m[(led_m.index >= start) & (led_m.index <= end)]
        if len(sub) < 12:
            continue
        s = st.summarize(sub, "ret_gross")
        print(f"  {label}: n={s['n_months']}, ann={s['mean_ann_pct']:+.2f}%, t={s['tstat']:+.2f}")

    # Fingerprint stamp.
    print(f"\nFingerprint: {fp} | as-of 2026-06-14")
    print(f"Currencies: {data.CCY_NAMES}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
