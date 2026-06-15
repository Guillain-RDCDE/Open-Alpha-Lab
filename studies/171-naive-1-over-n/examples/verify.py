"""Real-tape verification — Study 171 (Naive-1-Over-N). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the eleven SPDR sector ETF daily price series, runs the
out-of-sample tournament (1/N vs min-variance, max-Sharpe, MVO), and reports the key
numbers. Network is touched only with --fetch.

    python studies/171-naive-1-over-n/examples/verify.py            # cache-only
    python studies/171-naive-1-over-n/examples/verify.py --fetch    # refresh prices
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from naive_1_over_n import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        prices = data.load_real(fetch=True)
        print(f"Downloaded {prices.shape[1]} tickers, {len(prices)} days")
        print(f"Window: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Fingerprint: {data.fingerprint(prices)}\n")
    else:
        prices = data.load_real(fetch=False)

    print(f"=== Data stamp ===")
    print(f"Tickers: {list(prices.columns)}")
    print(f"Window:  {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Fingerprint: {data.fingerprint(prices)}")
    print(f"n_days: {len(prices)}\n")

    returns = prices.pct_change().dropna()
    lookback = 60  # 60 months = 5 years
    cost = 10.0    # 10 bps one-way

    print("=== Out-of-sample tournament (lookback=60m, cost=10bps OW) ===")
    portfolio = st.run_all_strategies(returns, lookback=lookback, cost_bps=cost)
    print(f"\n{'Strategy':>12} | {'CAGR%':>7} | {'Sharpe':>7} | {'MaxDD%':>8} | {'t-ann':>7}")
    print("-" * 55)
    metrics = {}
    for col in portfolio.columns:
        m = st.summarize(portfolio[col])
        metrics[col] = m
        cagr = m["cagr"] * 100
        mdd = m["max_dd"] * 100
        print(
            f"{col:>12} | {cagr:>7.2f} | {m['sharpe']:>7.3f} | "
            f"{mdd:>8.2f} | {m['tstat_annual']:>+7.2f}"
        )

    print("\n=== Pairwise: optimiser minus 1/N (annual return diff, HAC t) ===")
    port_1n = portfolio["1n"]
    for strat in ("minvar", "maxsharpe", "mvo"):
        diff = st.hac_tstat_diff(portfolio[strat], port_1n)
        sign = "+" if diff["mean_diff"] >= 0 else ""
        print(
            f"{strat:>12} - 1/N: {sign}{diff['mean_diff']*100:+.2f}% ann, "
            f"HAC t={diff['tstat']:+.2f} (n_years={diff['n']})"
        )

    print("\n=== Sensitivity: varying lookback (cost=10bps) ===")
    print(f"{'lookback':>9} | {'1N sharpe':>10} | {'minvar':>8} | {'maxsharpe':>10} | {'mvo':>6}")
    for lb in (24, 36, 60, 84):
        p = st.run_all_strategies(returns, lookback=lb, cost_bps=cost)
        sharpes = {c: st.summarize(p[c])["sharpe"] for c in p.columns}
        print(
            f"{lb:>9}m | {sharpes['1n']:>10.3f} | {sharpes['minvar']:>8.3f} | "
            f"{sharpes['maxsharpe']:>10.3f} | {sharpes['mvo']:>6.3f}"
        )

    print("\n=== Sensitivity: varying rebalance cost ===")
    print(f"{'cost_bps':>9} | {'1N sharpe':>10} | {'minvar':>8} | {'maxsharpe':>10} | {'mvo':>6}")
    for c in (0.0, 5.0, 10.0, 20.0):
        p = st.run_all_strategies(returns, lookback=lookback, cost_bps=c)
        sharpes = {col: st.summarize(p[col])["sharpe"] for col in p.columns}
        print(
            f"{c:>9.1f} | {sharpes['1n']:>10.3f} | {sharpes['minvar']:>8.3f} | "
            f"{sharpes['maxsharpe']:>10.3f} | {sharpes['mvo']:>6.3f}"
        )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
