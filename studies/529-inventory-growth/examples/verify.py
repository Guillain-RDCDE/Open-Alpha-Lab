"""Real-tape verification — Study 529 (Inventory-Growth). Regenerates docs/results.md numbers.

Reads this study's own cache (``_cache/inv_balance_sheets.parquet`` and
``_cache/inv_prices.parquet``), builds the inventory-growth signal, runs the quintile sort,
the long-low/short-high hedge with a one-sample HAC t-stat, the label-shuffle placebo null,
the random-portfolio control, costs + borrow, and the seed-robust synthetic positive control.
PRINTS every number. Cache-only — no network unless you pass ``--fetch``.

    python studies/529-inventory-growth/examples/verify.py
    python studies/529-inventory-growth/examples/verify.py --fetch   # refresh the cache
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401  (kept for parity / ad-hoc poking)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inventory_growth import data, strategy as st  # noqa: E402


def main() -> None:
    fetch = "--fetch" in sys.argv
    invg, fwd = data.fetch_panel(fetch=fetch)
    if invg.empty:
        print("ERROR: cache empty. Run once with --fetch to populate _cache/.")
        sys.exit(1)

    n_names = int(invg.notna().any(axis=0).sum())
    print(f"Basket: {len(data.BASKET)} requested, {n_names} with usable INVG")
    print(f"INVG panel: {invg.shape[0]} signal years {list(invg.index)}  fingerprint {data.fingerprint(invg)}")
    print(f"Names with valid INVG per year: {invg.notna().sum(axis=1).to_dict()}")
    print()
    print("** DATA LIMITATION: Yahoo serves ~4-5 annual balance sheets per name, so only")
    print("   a handful of signal years survive the one-year reporting lag + complete-year")
    print("   return filter. The hedge has very few annual observations. Stated openly. **")
    print()

    h = st.quintile_returns(invg, fwd, q=0.20, min_names=10)

    print("=== quintile returns (INVG low->high, forward 1yr) ===")
    for q in ["q1", "q2", "q3", "q4", "q5", "market"]:
        s = st.summary(h[q])
        label = {"q1": "low-INVG", "q5": "high-INVG", "market": "EW mkt"}.get(q, q)
        print(f"  {label:10s} mean={s['mean']*100:+.1f}%/yr  Sharpe={s['sharpe']:+.2f}"
              f"  HAC t={s['tstat']:+.2f}  hit={s['hit_rate']:.0%}  n={s['n']}")

    gross = st.summary(h["hedge"])
    net_series = st.net_hedge(h["hedge"])
    net = st.summary(net_series)
    print()
    print("=== long low-INVG / short high-INVG hedge ===")
    print(f"  GROSS mean={gross['mean']*100:+.1f}%/yr  Sharpe={gross['sharpe']:+.2f}  "
          f"HAC t={gross['tstat']:+.2f}  hit={gross['hit_rate']:.0%}  n={gross['n']}")
    cost = (2 * 5.0 + 50.0) / 1e4
    print(f"  cost drag = 2 legs x 5 bps + 50 bps borrow = {cost*100:.1f}%/yr")
    print(f"  NET   mean={net['mean']*100:+.1f}%/yr  Sharpe={net['sharpe']:+.2f}  "
          f"HAC t={net['tstat']:+.2f}")

    print()
    print("=== placebo / label-shuffle null (1000 shuffles) ===")
    p_val, null_means = st.placebo_hedge_t(invg, fwd, n_shuffles=1000, min_names=10)
    print(f"  Real hedge mean={gross['mean']*100:+.1f}%/yr beats placebo with p={p_val:.3f}")
    print(f"  Null hedge mean={null_means.mean()*100:+.2f}%/yr  std={null_means.std()*100:.1f}%")

    print()
    print("=== random-portfolio control (1000 draws) ===")
    rand = st.random_portfolio_returns(invg, fwd, n_draws=1000, min_names=10)
    lo_ex = float((h["q1"] - h["market"]).mean())
    pct = float((rand < lo_ex).mean())
    print(f"  Low-INVG excess vs EW mkt = {lo_ex*100:+.1f}%/yr; beats {pct:.0%} of random draws")
    print(f"  Random control mean={rand.mean()*100:+.2f}%  std={rand.std()*100:.1f}%")

    print()
    print("=== synthetic positive control (seed-robust, 20 seeds) ===")
    for prem in [0.0, -0.03, -0.06, -0.10]:
        r = st.seed_robust_synthetic_t(prem, n_seeds=20, n_firms=200, n_years=20)
        print(f"  premium={prem:+.2f}  mean_hedge={r['mean_hedge']*100:+.1f}%/yr  "
              f"mean_t={r['mean_t']:+.2f}  frac(t>2)={r['frac_t_gt2']:.0%}")

    print()
    print("=== year-by-year ===")
    print(f"{'year':>6}  {'n':>4}  {'q1 (lo)':>9}  {'q5 (hi)':>9}  {'mkt':>9}  {'hedge':>9}")
    for y, row in h.iterrows():
        print(f"{y:>6}  {row['n']:>4.0f}  {row['q1']*100:>+8.1f}%"
              f"  {row['q5']*100:>+8.1f}%  {row['market']*100:>+8.1f}%"
              f"  {row['hedge']*100:>+8.1f}%")

    print()
    print("SURVIVORSHIP-BIAS WARNING: basket is inventory-heavy names STILL TRADING in 2026.")
    print("Failed over-builders are absent; results are upper bounds on a live strategy.")


if __name__ == "__main__":
    main()
