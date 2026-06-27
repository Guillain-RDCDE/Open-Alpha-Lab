"""Real-tape verification — Study 524 (Operating-Leverage). Regenerates results.md.

Reads the cached EDGAR fundamentals + Yahoo daily prices, computes OL = OperatingCosts /
Assets, runs the tercile sort, the long-short hedge (long high-OL / short low-OL), the
placebo / label-shuffle null, the random-portfolio null, costs, and the seed-robust
synthetic positive control — and PRINTS every number. Cache-only by default (no network);
pass ``--fetch`` to refresh.

    python studies/524-operating-leverage/examples/verify.py
    python studies/524-operating-leverage/examples/verify.py --fetch
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from operating_leverage import data, strategy as st  # noqa: E402


def main() -> None:
    fetch = "--fetch" in sys.argv
    ol, fwd = data.fetch_panel(fetch=fetch)
    if ol.empty:
        print("ERROR: no cache. Run once with --fetch to populate _cache/.")
        sys.exit(1)
    ol = ol.dropna(how="all")

    h = st.sorted_returns(ol, fwd, n_groups=3)
    print(f"Panel: {h['n'].mean():.0f} names/yr avg, {len(h)} sort years "
          f"({int(h.index.min())}-{int(h.index.max())})")
    print(f"Fingerprint (OL):  {data.fingerprint(ol)}")
    print(f"Fingerprint (FWD): {data.fingerprint(fwd)}")
    print()

    print("=== tercile returns (OL low->high, 12-mo forward, report-lagged) ===")
    for g, lab in [("g1", "low-OL"), ("g2", "mid-OL"), ("g3", "high-OL"), ("market", "EW mkt")]:
        s = st.summary(h[g])
        print(f"  {lab:8s} mean={s['mean']*100:+.1f}%/yr  Sharpe={s['sharpe']:+.2f}"
              f"  HAC t={s['tstat']:+.2f}  hit={s['hit_rate']:.0%}")

    hedge = st.summary(h["hedge"])
    print()
    print("=== long high-OL / short low-OL hedge (g3 - g1) ===")
    print(f"  mean={hedge['mean']*100:+.2f}%/yr  Sharpe={hedge['sharpe']:+.2f}  "
          f"HAC t={hedge['tstat']:+.2f}  hit={hedge['hit_rate']:.0%}  n={hedge['n']}")

    hi_ex = st.summary(h["g3"] - h["market"])
    lo_ex = st.summary(h["g1"] - h["market"])
    print()
    print("=== excess vs EW market ===")
    print(f"  High-OL (g3) excess: mean={hi_ex['mean']*100:+.1f}%/yr  HAC t={hi_ex['tstat']:+.2f}")
    print(f"  Low-OL (g1) excess:  mean={lo_ex['mean']*100:+.1f}%/yr  HAC t={lo_ex['tstat']:+.2f}")

    print()
    print("=== placebo / label-shuffle null (1000 shuffles) ===")
    p, sh = st.placebo_hedge_tstats(ol, fwd, n_groups=3, n_shuffles=1000, seed=524)
    print(f"  Real hedge mean = {hedge['mean']*100:+.2f}%/yr")
    print(f"  Shuffle mean = {sh.mean()*100:+.2f}%/yr  std = {sh.std()*100:.2f}%/yr")
    print(f"  Placebo p (P[shuffle >= real]) = {p:.3f}  (>0.05 => indistinguishable from noise)")

    print()
    print("=== random-portfolio null (500 draws/yr) ===")
    rand = st.random_portfolio_returns(ol, fwd, n_groups=3, n_draws=500, seed=524)
    hi_excess = float((h["g3"] - h["market"]).mean())
    print(f"  Random mean={rand.mean()*100:+.2f}%  std={rand.std()*100:.1f}%")
    print(f"  High-OL beats {(rand < hi_excess).mean():.0%} of random same-size draws")

    print()
    print("=== costs (gross vs net) ===")
    c = st.net_of_costs(hedge["mean"], one_way_bps=10.0, annual_turnover=2.0, borrow_bps=50.0)
    print(f"  Gross hedge   = {c['gross']*100:+.2f}%/yr")
    print(f"  Trading drag  = {c['cost_drag']*100:.2f}%/yr (10bps x 2.0 turnover)")
    print(f"  Borrow drag   = {c['borrow_drag']*100:.2f}%/yr (50bps short leg)")
    print(f"  NET hedge     = {c['net']*100:+.2f}%/yr")

    print()
    print("=== synthetic positive control (seed-robust, 25 seeds) ===")
    for prem in (0.0, 0.06, 0.10):
        r = st.seed_robust_synthetic(prem, n_seeds=25)
        print(f"  premium={prem:+.2f}/yr -> mean hedge={r['mean_hedge']*100:+.1f}%/yr  "
              f"mean HAC t={r['mean_tstat']:+.2f}")

    print()
    print("=== year-by-year ===")
    print(f"{'year':>6} {'n':>4} {'g1(lo)':>8} {'g2':>8} {'g3(hi)':>8} {'mkt':>8} {'hedge':>8}")
    for y, row in h.iterrows():
        print(f"{int(y):>6} {int(row['n']):>4} {row['g1']*100:>+7.1f}% {row['g2']*100:>+7.1f}%"
              f" {row['g3']*100:>+7.1f}% {row['market']*100:>+7.1f}% {row['hedge']*100:>+7.1f}%")

    print()
    print("SURVIVORSHIP-BIAS WARNING: basket = large-cap names still trading in 2026.")
    print("High-operating-leverage firms whose demand collapsed are absent. Results are UPPER BOUNDS.")


if __name__ == "__main__":
    main()
