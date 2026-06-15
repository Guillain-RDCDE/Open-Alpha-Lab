"""Real-tape verification — Study 205 (Three-Fund). Regenerates docs/results.md numbers.

Reads (or fetches) VTI, VXUS, BND, SPY daily prices, runs the three-way tournament
(three-fund vs 60/40 vs 100% US), and reports the key metrics.

    python studies/205-three-fund/examples/verify.py            # cache-only
    python studies/205-three-fund/examples/verify.py --fetch    # refresh prices
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from three_fund import data, strategy as st  # noqa: E402
from quantlab import stats                   # noqa: E402


def main(fetch: bool) -> None:
    prices = data.load_real(fetch=fetch)

    print("=== Data stamp ===")
    print(f"Tickers:     {list(prices.columns)}")
    print(f"Window:      {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"n_days:      {len(prices)}")
    print(f"Fingerprint: {data.fingerprint(prices)}\n")

    cost = 5.0  # 5 bps one-way; annual rebalance means turnover is tiny
    navs = st.run_all_portfolios(prices, cost_bps=cost)

    print(f"=== Portfolio comparison (cost={cost:.0f}bps one-way, annual rebalance) ===")
    print(f"\n{'Strategy':>14} | {'CAGR%':>7} | {'Sharpe':>7} | {'Vol%':>6} | {'MaxDD%':>8} | {'Calmar':>7} | {'t-ann':>7}")
    print("-" * 76)
    metrics = {}
    for name, nav in navs.items():
        m = st.summarize(nav)
        metrics[name] = m
        print(
            f"{name:>14} | {m['cagr']*100:>7.2f} | {m['sharpe']:>7.3f} | "
            f"{m['ann_vol']*100:>6.2f} | {m['max_dd']*100:>8.2f} | "
            f"{m['calmar']:>7.3f} | {m['tstat_annual']:>+7.2f}"
        )

    print("\n=== Pairwise diffs (three_fund minus baseline, HAC t on annual diffs) ===")
    diff_vs_us   = st.hac_tstat_diff(navs["three_fund"], navs["us_only"])
    diff_vs_6040 = st.hac_tstat_diff(navs["three_fund"], navs["sixty_forty"])
    bonf = st.bonferroni_threshold(n_tests=3, alpha=0.05)
    print(f"  vs US-only (SPY):   {diff_vs_us['mean_diff']*100:+.2f}%/yr, t={diff_vs_us['tstat']:+.2f}  (n={diff_vs_us['n']} years)")
    print(f"  vs 60/40 (SPY+BND): {diff_vs_6040['mean_diff']*100:+.2f}%/yr, t={diff_vs_6040['tstat']:+.2f}  (n={diff_vs_6040['n']} years)")
    print(f"  Bonferroni |t| threshold (3 tests, 5%): {bonf:.3f}")

    print("\n=== International attribution (3-fund vs 90/10 VTI+BND, no VXUS) ===")
    intl = st.international_attribution(prices, cost_bps=cost)
    print(f"  VXUS sleeve: {intl['mean_diff']*100:+.2f}%/yr, t={intl['tstat']:+.2f} (n={intl['n']} years)")

    print("\n=== Bootstrap Sharpe CI (three-fund) ===")
    r_3f = st.nav_to_returns(navs["three_fund"])
    ci = stats.sharpe_ci_bootstrap(r_3f, periods_per_year=252, seed=205)
    print(f"  Sharpe = {ci['sharpe']:.3f}, 95% CI = [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}], frac_neg = {ci['frac_negative']*100:.1f}%")

    print("\n=== Sub-period Sharpe breakdown ===")
    print(f"{'Period':>11} | {'3f Sharpe':>10} | {'US Sharpe':>10} | {'60/40 Sharpe':>12} | {'3f CAGR%':>9} | {'US CAGR%':>9}")
    for period, start, end in [
        ("2011-2015", "2011-01-01", "2016-01-01"),
        ("2016-2020", "2016-01-01", "2021-01-01"),
        ("2021-2026", "2021-01-01", "2027-01-01"),
    ]:
        sub = prices[(prices.index >= start) & (prices.index < end)]
        if len(sub) < 100:
            continue
        navs_sub = st.run_all_portfolios(sub, cost_bps=cost)
        ms = {k: st.summarize(v) for k, v in navs_sub.items()}
        print(
            f"{period:>11} | {ms['three_fund']['sharpe']:>10.3f} | "
            f"{ms['us_only']['sharpe']:>10.3f} | {ms['sixty_forty']['sharpe']:>12.3f} | "
            f"{ms['three_fund']['cagr']*100:>9.1f} | {ms['us_only']['cagr']*100:>9.1f}"
        )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
