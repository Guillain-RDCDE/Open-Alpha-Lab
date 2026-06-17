"""Real-tape verification — Study 215 (Big-Mac-PPP). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the annual FX panel, joins with hardcoded Big Mac
mis-valuation table, runs the long-short PPP portfolio (long under-valued, short
over-valued), pins against a random-ranking control, sweeps costs, and runs the
cross-sectional pooled regression.

    python studies/216-big-mac-ppp/examples/verify.py            # cache-only
    python studies/216-big-mac-ppp/examples/verify.py --fetch    # refresh FX panel

Network is touched only with --fetch.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from big_mac_ppp import data, strategy as st  # noqa: E402

N_RANDOM_SEEDS = 20


def main(fetch: bool) -> None:
    # Load data
    misval = data.hardcoded_misval()
    fx_annual = data.fetch_fx(fetch=fetch)

    print(f"Big Mac mis-val table: {len(misval)} years x {len(misval.columns)} currencies")
    print(f"FX annual panel: {len(fx_annual)} years, columns: {list(fx_annual.columns)}\n")

    # Build joint panel
    panel = data.build_panel(misval, fx_annual)
    fp = data.fingerprint(panel)
    print(f"Joint panel: {len(panel)} obs, fingerprint={fp}\n")

    # Cross-sectional pooled regression
    reg = st.cross_sectional_regression(panel)
    print("=== Cross-sectional pooled regression: mis_val -> fwd_log_ret ===")
    print(f"  n_obs={reg['n_obs']}, slope={reg['slope']:.4f}, "
          f"t={reg['t_slope']:+.2f}, R2={reg['r_squared']:.4f}")
    print(f"  Interpretation: 1% over-valuation -> {reg['slope']*100:.3f}% change in FX (next year)")

    # PPP portfolio (gross, 0 cost)
    ranks_p = st.misval_ranks(misval)
    led_p = st.build_portfolio(misval, fx_annual, ranks=ranks_p, cost_bps=0)
    s_p = st.summarize(led_p, "ret_gross")

    print("\n=== PPP long-short portfolio (long under-valued, short over-valued, gross) ===")
    print(f"  n_years={s_p['n_years']}, ann_mean={s_p['mean_ann_pct']:+.2f}%,"
          f" Sharpe={s_p['sharpe_ann']:+.2f}, t={s_p['tstat']:+.2f}")

    # Random-ranking control
    rand_means = []
    for seed in range(N_RANDOM_SEEDS):
        ranks_r = st.random_ranks(misval, seed=seed)
        led_r = st.build_portfolio(misval, fx_annual, ranks=ranks_r, cost_bps=0)
        s_r = st.summarize(led_r, "ret_gross")
        rand_means.append(s_r["mean_ann_pct"])
    rand_mean = np.mean(rand_means)
    print(f"\n=== Random-ranking control (avg over {N_RANDOM_SEEDS} seeds) ===")
    print(f"  avg ann_mean={rand_mean:+.2f}%")
    print(f"  PPP - Random = {s_p['mean_ann_pct'] - rand_mean:+.2f}%")

    # Cost sweep
    print("\n=== Cost sweep (PPP portfolio, net) ===")
    costs = [0, 5, 10, 20, 50]
    for c in costs:
        led = st.build_portfolio(misval, fx_annual, ranks=ranks_p, cost_bps=c)
        s = st.summarize(led, "ret_net")
        print(f"  cost={c:3d} bps/leg: ann_mean={s['mean_ann_pct']:+.2f}%, t={s['tstat']:+.2f}")

    # Sub-period breakdown
    print("\n=== Sub-period breakdown (gross) ===")
    for label, y_start, y_end in [
        ("2000-2007", 2000, 2007),
        ("2008-2015", 2008, 2015),
        ("2016-2023", 2016, 2023),
    ]:
        sub = led_p[(led_p.index >= y_start) & (led_p.index <= y_end)]
        if len(sub) < 4:
            continue
        s = st.summarize(sub, "ret_gross")
        print(f"  {label}: n={s['n_years']}, ann={s['mean_ann_pct']:+.2f}%, t={s['tstat']:+.2f}")

    # Fingerprint stamp
    print(f"\nFingerprint: {fp} | as-of 2026-06-16")
    print(f"Currencies: {data.CURRENCIES}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
