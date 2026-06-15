"""Real-tape verification — Study 204 (Talmud-Portfolio). Regenerates docs/results.md numbers.

Loads (or fetches) the daily total-return prices for VNQ, SPY, and BND, runs the
Talmud 1/3–1/3–1/3 portfolio against 60/40 (SPY/BND), 100 % SPY, and the 50/50
SPY/VNQ mix. Prints the headline stats and HAC t-stats.

    python studies/204-talmud-portfolio/examples/verify.py            # cache-only
    python studies/204-talmud-portfolio/examples/verify.py --fetch    # refresh prices
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from talmud_portfolio import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    prices = data.load_real(fetch=fetch)
    if fetch:
        print(f"Fetched: {', '.join(prices.columns.tolist())}")
        print(f"Window:  {prices.index[0].date()} -> {prices.index[-1].date()}")
        for c in prices.columns:
            print(f"  {c}: fp={data.fingerprint(prices[[c]])}")
        print()

    ret = st.to_returns(prices)
    rf = ret["BND"]

    # All portfolios at 10 bps one-way rebalance cost
    tal   = st.talmud_portfolio(ret, cost_bps=10.0)
    s6040 = st.sixty_forty(ret, cost_bps=10.0)
    spy   = st.spy_only(ret)
    vnq50 = st.spy_vnq_50_50(ret, cost_bps=10.0)

    print("=== Headline stats (10 bps one-way rebalance cost) ===")
    print(f"{'Portfolio':22s} {'CAGR':>7} {'Sharpe':>7} {'Vol':>7} {'MaxDD':>7} {'WorstYr':>8}")
    print("-" * 65)
    for name, port in [
        ("Talmud 1/3-1/3-1/3", tal),
        ("60/40 (SPY/BND)", s6040),
        ("SPY 100%", spy),
        ("SPY/VNQ 50/50", vnq50),
    ]:
        s = st.portfolio_stats(port, rf=rf)
        print(f"{name:22s} {s['cagr']*100:>6.1f}% {s['sharpe']:>+7.2f} "
              f"{s['vol']*100:>6.1f}% {s['max_dd']*100:>6.1f}% {s['worst_year']*100:>7.1f}%")

    print()
    print("=== HAC t-stats on annual return differences (Talmud vs benchmark) ===")
    print(f"Talmud vs 60/40:       t = {st.hac_tstat_annual(tal, s6040):+.3f}")
    print(f"Talmud vs SPY 100%:    t = {st.hac_tstat_annual(tal, spy):+.3f}")
    print(f"Talmud vs SPY/VNQ:     t = {st.hac_tstat_annual(tal, vnq50):+.3f}")

    print()
    print("=== Equity crash episodes (SPY peak-to-trough >= -15%) ===")
    episodes = st.drawdown_episodes(ret, "SPY", thresh=-0.15)
    print(f"{'Peak':12s} {'Trough':12s} {'SPY':>7} {'VNQ':>7} {'BND':>7}")
    for ep in episodes:
        print(f"{str(ep['peak'].date()):12s} {str(ep['trough'].date()):12s} "
              f"{ep['stock_loss']*100:>6.1f}%"
              f"{ep['others'].get('VNQ', float('nan'))*100:>7.1f}%"
              f"{ep['others'].get('BND', float('nan'))*100:>7.1f}%")

    print()
    print(f"Data fingerprint: {data.fingerprint(prices)}")
    print(f"Date range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"n_days: {len(prices)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
