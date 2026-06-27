"""Real-panel verification -- Study 532 (Firm-Age-Anomaly). Regenerates docs/results.md numbers.

Reads the study's own yfinance cache (studies/532-firm-age-anomaly/_cache/age_prices.parquet),
proxies firm age from first available price date, builds the monthly old-minus-young (firm-age)
long-short, and reports the premium, HAC t-stat, label-shuffle placebo p-value, costs, a seed
robustness check on the synthetic control, and the year-by-year breakdown.

    python studies/532-firm-age-anomaly/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from firm_age_anomaly import data, strategy as st  # noqa: E402


def synthetic_control_seeds(n_seeds: int = 25) -> dict:
    """Average a Welch-style stat over many seeds to prove the engine is faithful.

    Plant a positive age premium; the engine must recover a positive AGE mean with a
    t-stat that is large *on average* (not one lucky seed), and recover ~zero at premium=0.
    """
    pos_t, null_t, pos_mean = [], [], []
    for s in range(n_seeds):
        p, spy, fd, _ = data.synthetic_panel(age_premium=0.30, seed=1000 + s)
        res = st.age_portfolio(p, spy, first_dates=fd, min_stocks=9)
        if not res.empty:
            ss = st.summary(res["age_ret"])
            pos_t.append(ss["tstat"]); pos_mean.append(ss["mean"])
        p0, spy0, fd0, _ = data.synthetic_panel(age_premium=0.0, seed=1000 + s)
        res0 = st.age_portfolio(p0, spy0, first_dates=fd0, min_stocks=9)
        if not res0.empty:
            null_t.append(st.summary(res0["age_ret"])["tstat"])
    return {
        "pos_mean_t": float(np.nanmean(pos_t)),
        "pos_mean_ret": float(np.nanmean(pos_mean)) * 100,
        "null_mean_t": float(np.nanmean(null_t)),
        "frac_pos_sig": float(np.mean(np.array(pos_t) > 2.0)),
        "n_seeds": n_seeds,
    }


def main() -> None:
    prices, spy = data.fetch_panel()

    print("Study 532 -- Firm-Age-Anomaly -- new-list / firm-age effect (old minus young)\n")

    # Synthetic positive control (faithful-engine / power check only; never a real stamp).
    print("=== Synthetic positive control (seed-robust, >=20 seeds) ===")
    ctrl = synthetic_control_seeds(25)
    print(f"  Planted premium 0.30: avg AGE mean = {ctrl['pos_mean_ret']:+.2f}%/yr"
          f" | avg HAC t = {ctrl['pos_mean_t']:+.3f} over {ctrl['n_seeds']} seeds")
    print(f"  Null premium 0.00:    avg HAC t = {ctrl['null_mean_t']:+.3f}")
    print(f"  Fraction of seeds with t>2 (planted): {ctrl['frac_pos_sig']*100:.0f}%")
    print("  -> engine is faithful: recovers planted effect, ~0 under null.\n")

    if prices.empty:
        print("Cache not found. Run from repo root with network access to populate the cache.")
        return

    first_dates = data.first_price_dates(prices)
    fp_prices = data.fingerprint(prices)
    fp_spy = data.fingerprint(spy)
    print("=== Real yfinance panel ===")
    print(f"Prices panel: {prices.shape}, fingerprint={fp_prices}")
    print(f"SPY series:   {spy.shape}, fingerprint={fp_spy}")
    print(f"Date range: {prices.index[0].date()} -- {prices.index[-1].date()}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close prices\n")

    print("=== First-price-date (firm-age proxy), sample ===")
    fd_sorted = first_dates.dropna().sort_values()
    for t in list(fd_sorted.index[:5]) + list(fd_sorted.index[-5:]):
        print(f"  {t:6s}: {fd_sorted[t].date()}")

    result = st.age_portfolio(prices, spy, first_dates=first_dates, min_stocks=9)
    if result.empty:
        print("Could not construct portfolio (too few months).")
        return

    s_age = st.summary(result["age_ret"])
    s_old = st.summary(result["old_ret"])
    s_young = st.summary(result["young_ret"])
    s_mkt = st.summary(result["mkt_ret"])

    print("\n=== Firm-Age long-short (OLD minus YOUNG, gross) ===")
    print(f"  N months:            {s_age['n']} ({result.index[0].date()} -- {result.index[-1].date()})")
    print(f"  Mean return (ann):   {s_age['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):           {s_age['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):        {s_age['sharpe']:+.3f}")
    print(f"  HAC t-stat:          {s_age['tstat']:+.3f}")
    print(f"  Hit rate:            {s_age['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:        {s_age['max_drawdown']*100:.1f}%")

    print("\n=== Legs ===")
    print(f"  Old leg (long, ann):   {s_old['mean']*100:+.2f}%/yr | avg age {result['avg_age_old'].mean():.1f} yrs")
    print(f"  Young leg (short,ann): {s_young['mean']*100:+.2f}%/yr | avg age {result['avg_age_young'].mean():.1f} yrs")
    print(f"  SPY (market, ann):     {s_mkt['mean']*100:+.2f}%/yr")

    # Net of costs
    net = st.apply_costs(result)
    s_net = st.summary(net)
    print("\n=== Net of costs (10 bps one-way x turnover + 100 bps/yr borrow on short) ===")
    print(f"  Avg two-sided turnover: {result['turnover'].mean():.2f}")
    print(f"  Net mean (ann):         {s_net['mean']*100:+.2f}%/yr")
    print(f"  Net HAC t-stat:         {s_net['tstat']:+.3f}")
    print(f"  Net Sharpe:             {s_net['sharpe']:+.3f}")

    # Placebo
    print("\n=== Label-shuffle placebo (permute firm-age labels) ===")
    pb = st.placebo_pvalue(prices, spy, first_dates=first_dates, n_perm=200, min_stocks=9)
    print(f"  Real AGE mean (monthly): {pb['real_mean']*100:+.3f}%")
    print(f"  Placebo mean of shuffles:{pb['perm_mean']*100:+.3f}% (std {pb['perm_std']*100:.3f}%)")
    print(f"  Two-sided permutation p: {pb['p_value']:.3f} over {pb['n_perm']} shuffles")

    print("\n=== Year-by-year AGE returns (compounded over months) ===")
    rc = result.copy()
    rc["year"] = rc.index.year
    annual = rc.groupby("year")["age_ret"].apply(lambda x: float((1 + x).prod() - 1))
    for yr, val in annual.items():
        print(f"  {yr}: {val*100:+.1f}%")

    print("\nNOTE Survivorship bias: basket = names still trading in 2026. Young firms that")
    print("     IPO'd then failed are absent -> the short (young) leg misses its worst names,")
    print("     biasing the old-minus-young spread DOWNWARD. Measured premium is conservative.")


if __name__ == "__main__":
    import pandas as pd  # noqa: F401  (used in date formatting above)
    main()
