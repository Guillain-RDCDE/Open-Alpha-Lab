"""Real-tape verification — Study 522 (Percent Operating Accruals). Prints every number.

Reads this study's own ``_cache/`` (EDGAR fundamentals + yfinance prices), computes the
HLVW (2011) percent-operating-accruals signal, runs the quintile sort, the long-short hedge,
HAC t-stats, the label-shuffle placebo, the random-portfolio null, costs/turnover, and the
synthetic positive control. If the cache is absent it pulls once from the network and caches.

    python studies/522-percent-operating-accruals/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from percent_operating_accruals import data, strategy as st  # noqa: E402


def main() -> None:
    if not data.have_real():
        print("Cache absent — pulling EDGAR fundamentals + yfinance prices once...")
        data.fetch_panel()

    sig, fwd = data.load_panel()
    if sig.empty:
        print("ERROR: real tape unavailable.")
        sys.exit(1)

    print(f"Panel: {len(sig.columns)} tickers, {len(sig)} signal years "
          f"({int(sig.index.min())}-{int(sig.index.max())})")
    print(f"Fingerprint (percent-accruals): {data.fingerprint(sig)}")
    print(f"Fingerprint (fwd returns):      {data.fingerprint(fwd)}")
    print()

    h = st.quintile_returns(sig, fwd, q=0.20, min_names=15)

    print("=== quintile returns (percent-accruals low->high, forward 1yr) ===")
    for q in ["q1", "q2", "q3", "q4", "q5", "market"]:
        s = st.summary(h[q])
        label = {"q1": "low-PA", "q5": "high-PA", "market": "EW mkt"}.get(q, q)
        print(f"  {label:8s} mean={s['mean']*100:+.1f}%/yr  Sharpe={s['sharpe']:+.2f}"
              f"  HAC t={s['tstat']:+.2f}  hit={s['hit_rate']:.0%}")

    hedge = st.summary(h["hedge"])
    print()
    print("=== long low-PA / short high-PA hedge (Q1 - Q5) ===")
    print(f"  GROSS mean={hedge['mean']*100:+.1f}%/yr  Sharpe={hedge['sharpe']:+.2f}  "
          f"HAC t={hedge['tstat']:+.2f}  hit={hedge['hit_rate']:.0%}  n={hedge['n']}")

    # Costs
    turn = st.turnover_series(sig, fwd, q=0.20, min_names=15)
    net = st.apply_costs(h["hedge"], turn, one_way_bps=10.0, borrow_bps=50.0)
    nets = st.summary(net)
    print()
    print("=== costs (10 bps one-way x 2 legs x turnover + 50 bps/yr borrow) ===")
    print(f"  mean turnover (long leg): {turn.mean()*100:.0f}%/yr")
    print(f"  NET mean={nets['mean']*100:+.1f}%/yr  Sharpe={nets['sharpe']:+.2f}  "
          f"HAC t={nets['tstat']:+.2f}  hit={nets['hit_rate']:.0%}")

    # Excess legs
    lo_x = st.summary(h["q1"] - h["market"])
    hi_x = st.summary(h["q5"] - h["market"])
    print()
    print("=== excess returns vs EW market ===")
    print(f"  Low-PA (q1) excess:  mean={lo_x['mean']*100:+.1f}%/yr  HAC t={lo_x['tstat']:+.2f}")
    print(f"  High-PA (q5) excess: mean={hi_x['mean']*100:+.1f}%/yr  HAC t={hi_x['tstat']:+.2f}")

    # Placebo
    print()
    print("=== label-shuffle placebo (1000 perms) ===")
    p, null = st.placebo_hedge_t(sig, fwd, q=0.20, n_perm=1000, min_names=15, seed=522)
    print(f"  real hedge HAC t = {hedge['tstat']:+.2f}")
    print(f"  placebo |t| >= |real t|:  p = {p:.3f}  (null mean t={null.mean():+.2f}, "
          f"std={null.std():.2f})")

    # Random-portfolio null
    print()
    print("=== random-portfolio null (long-leg size, 500 draws/yr) ===")
    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=500, seed=522)
    actual_lo = float((h["q1"] - h["market"]).mean())
    pct_beaten = float((rand < actual_lo).mean())
    print(f"  random excess mean={rand.mean()*100:+.2f}%  std={rand.std()*100:.1f}%")
    print(f"  low-PA portfolio beats {pct_beaten:.0%} of random same-size draws")

    # Seed robustness of the placebo p (quick)
    print()
    print("=== seed robustness (placebo p across 5 seeds) ===")
    ps = []
    for sd in (1, 7, 42, 101, 522):
        pp, _ = st.placebo_hedge_t(sig, fwd, q=0.20, n_perm=300, min_names=15, seed=sd)
        ps.append(pp)
    print("  placebo p:", ", ".join(f"{x:.3f}" for x in ps),
          f" (mean {np.mean(ps):.3f})")

    # Synthetic positive control
    print()
    print("=== synthetic positive control (engine faithfulness) ===")
    t_null = st.synthetic_hedge_t(premium=0.0, seed=522)
    t_strong = st.synthetic_hedge_t(premium=-0.05, seed=522)
    # average a Welch-style stat over >=20 seeds for the synthetic claim
    null_ts = [st.synthetic_hedge_t(0.0, seed=s) for s in range(20)]
    strong_ts = [st.synthetic_hedge_t(-0.05, seed=s) for s in range(20)]
    print(f"  premium=0 (null):     t={t_null:+.2f}   mean over 20 seeds={np.mean(null_ts):+.2f}")
    print(f"  premium=-5%/yr (live): t={t_strong:+.2f}  mean over 20 seeds={np.mean(strong_ts):+.2f}")

    # Year-by-year
    print()
    print("=== year-by-year summary ===")
    print(f"{'year':>6}  {'n':>4}  {'q1(lo)':>8}  {'q5(hi)':>8}  {'mkt':>8}  {'hedge':>8}")
    for y, row in h.iterrows():
        print(f"{int(y):>6}  {int(row['n']):>4d}  {row['q1']*100:>+7.1f}%"
              f"  {row['q5']*100:>+7.1f}%  {row['market']*100:>+7.1f}%"
              f"  {row['hedge']*100:>+7.1f}%")

    print()
    print("SURVIVORSHIP BIAS WARNING: basket = fixed large-caps still trading 2026.")
    print("All real-tape numbers are UPPER BOUNDS on a live strategy.")


if __name__ == "__main__":
    main()
