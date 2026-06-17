"""Real-tape verification -- Study 262 (Short-Interest). Regenerates docs/results.md numbers.

Joins the hardcoded short-interest snapshot with cached (or freshly fetched)
forward returns for the basket, sorts the cross-section into high-SI / low-SI
quantiles, and reports the spread, the OLS slope, a label-shuffle null, a
bootstrap CI, a horizon sweep, and a cost-adjusted net spread.
Network is touched only with --fetch.

    python studies/262-short-interest/examples/verify.py          # cache-only
    python studies/262-short-interest/examples/verify.py --fetch  # refresh forward returns
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from short_interest import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    si = data.short_interest_table()
    print(f"Short-interest snapshot: {len(si)} tickers (as-of {data.SNAPSHOT_DATE})")
    print(f"SI fingerprint: {data.fingerprint(si)}")

    fwd = data.fetch_forward_returns(fetch=fetch)
    print(f"Forward-return panel: {fwd.shape[0]} tickers x {fwd.shape[1]} horizons")
    print(f"Forward fingerprint: {data.fingerprint(fwd)}\n")

    panel = data.real_panel(horizon_col="fwd_3m")
    qs = st.quantile_spread(panel, q=0.20)
    sl = st.cross_sectional_slope(panel)

    print("=== Primary spec: short_pct_float -> 3-month forward return ===")
    print(f"n = {qs['n']} ({qs['n_high']} high-SI, {qs['n_low']} low-SI)")
    print(f"HIGH-SI mean fwd: {qs['high']*100:+.2f}%")
    print(f"LOW-SI  mean fwd: {qs['low']*100:+.2f}%")
    print(f"SPREAD (high-low): {qs['spread']*100:+.2f}%   cross-sectional t = {qs['tstat']:+.2f}")
    print(f"OLS slope (per +1 SD SI): {sl['slope']*100:+.2f}%  t = {sl['tstat']:+.2f}  r2 = {sl['r2']:.3f}")

    sh = st.shuffle_null(panel, q=0.20, n_shuffles=2000, seed=262)
    bc = st.bootstrap_spread_ci(panel, q=0.20, n_boot=2000, seed=262)
    print(f"\nLabel-shuffle p-value: {sh['p_value']:.3f}")
    print(f"Bootstrap 95% CI on spread: [{bc['ci_low']*100:+.1f}%, {bc['ci_high']*100:+.1f}%]  "
          f"(frac opposite sign: {bc['frac_opposite']:.2f})")

    print("\n=== Horizon sweep ===")
    for col in ("fwd_1m", "fwd_3m", "fwd_6m", "fwd_12m"):
        try:
            p = data.real_panel(horizon_col=col)
        except KeyError:
            continue
        q_h = st.quantile_spread(p, q=0.20)
        print(f"  {col}: spread={q_h['spread']*100:+.2f}%  t={q_h['tstat']:+.2f}  n={q_h['n']}")

    print("\n=== Costs (3-month horizon) ===")
    ca = st.cost_adjusted_spread(panel, q=0.20, one_way_bps=25.0, borrow_ann_pct=8.0, horizon_months=3)
    print(f"Gross: {ca['gross']*100:+.2f}%  Cost: {ca['cost']*100:.2f}%  Net: {ca['net']*100:+.2f}%")

    print("\n=== Synthetic positive control ===")
    for beta in (-0.04, 0.0, 0.04):
        df, _ = data.synthetic_panel(n_firms=200, si_beta=beta, seed=262)
        q_s = st.quantile_spread(df, q=0.20)
        print(f"  si_beta={beta:+.2f}: spread={q_s['spread']*100:+.2f}%  t={q_s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
