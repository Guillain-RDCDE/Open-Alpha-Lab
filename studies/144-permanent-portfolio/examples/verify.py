"""Real-tape verification — Study 144 (Permanent-Portfolio). Regenerates docs/results.md.

Loads the shared cross-asset ETF cache (SPY / TLT / GLD / SHY), runs the
25/25/25/25 Permanent Portfolio with annual rebalance against 100% SPY and 60/40,
and prints the headline risk-adjusted stats that populate docs/results.md.

    python studies/144-permanent-portfolio/examples/verify.py            # cache-only
    python studies/144-permanent-portfolio/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from permanent_portfolio import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    # ---- load ----------------------------------------------------------------
    px = data.load_real(fetch=fetch)
    print(f"=== Permanent-Portfolio real-tape results ===")
    print(f"Tickers : SPY / TLT / GLD / SHY")
    print(f"Window  : {px.index[0].date()} -> {px.index[-1].date()}")
    print(f"Fingerprint: {data.fingerprint(px)}")
    print()

    ret = st.to_returns(px)
    rf = ret["SHY"]  # total-return cash proxy

    # ---- portfolios ----------------------------------------------------------
    pp = st.permanent_portfolio(ret, cost_bps=1.0)
    p60 = st.sixty_forty(ret, cost_bps=1.0)
    spy = st.spy_only(ret)

    pp_s = st.portfolio_stats(pp, rf=rf)
    p60_s = st.portfolio_stats(p60, rf=rf)
    spy_s = st.portfolio_stats(spy, rf=rf)

    print("=== Headline stats (annual rebalance, 1 bp one-way cost) ===")
    header = f"{'Portfolio':20s}  {'CAGR':>7}  {'Vol':>6}  {'Sharpe':>7}  {'MaxDD':>7}  {'WorstYr':>8}"
    print(header)
    print("-" * len(header))
    for name, s in [("PP 25/25/25/25", pp_s), ("60/40 SPY/TLT", p60_s), ("100% SPY", spy_s)]:
        print(f"{name:20s}  {s['cagr']*100:>6.2f}%  {s['vol']*100:>5.2f}%  "
              f"{s['sharpe']:>7.3f}  {s['max_dd']*100:>6.1f}%  {s['worst_year']*100:>7.1f}%")
    print()

    # ---- inference -----------------------------------------------------------
    t_vs_spy = st.hac_tstat_annual(pp, spy)
    t_vs_6040 = st.hac_tstat_annual(pp, p60)
    print("=== HAC t-stats on annual return differences ===")
    print(f"PP minus SPY:    HAC t = {t_vs_spy:+.3f}")
    print(f"PP minus 60/40:  HAC t = {t_vs_6040:+.3f}")
    print()

    bst_spy = st.bootstrap_sharpe_diff(pp, spy, rf=rf, seed=144)
    bst_6040 = st.bootstrap_sharpe_diff(pp, p60, rf=rf, seed=144)
    print("=== Bootstrap Sharpe difference (PP vs benchmark) ===")
    print(f"PP vs SPY  : {bst_spy['point']:+.3f}  "
          f"CI95=[{bst_spy['ci95'][0]:+.3f}, {bst_spy['ci95'][1]:+.3f}]  "
          f"PP wins {bst_spy['frac_a_wins']*100:.0f}% of resamples")
    print(f"PP vs 60/40: {bst_6040['point']:+.3f}  "
          f"CI95=[{bst_6040['ci95'][0]:+.3f}, {bst_6040['ci95'][1]:+.3f}]  "
          f"PP wins {bst_6040['frac_a_wins']*100:.0f}% of resamples")
    print()

    # ---- drawdown episodes ---------------------------------------------------
    episodes = st.equity_drawdowns(ret, "SPY", thresh=-0.10)
    print(f"=== SPY drawdown episodes (>10%), n={len(episodes)} ===")
    for ep in episodes:
        p = ep["peak"].strftime("%Y-%m")
        t = ep["trough"].strftime("%Y-%m")
        others = ep["others"]
        pp_during = others.get("TLT", float("nan")) * 0.25 + others.get("GLD", float("nan")) * 0.25
        print(f"  {p}->{t}  SPY={ep['stock_loss']*100:+.1f}%  "
              f"TLT={others.get('TLT',0)*100:+.1f}%  "
              f"GLD={others.get('GLD',0)*100:+.1f}%  "
              f"SHY={others.get('SHY',0)*100:+.1f}%")

    print()
    print("=== Annual returns ===")
    ar_pp = st.annual_returns(pp)
    ar_spy = st.annual_returns(spy)
    ar_6040 = st.annual_returns(p60)
    import pandas as pd
    combined = pd.DataFrame({"PP": ar_pp, "SPY": ar_spy, "60/40": ar_6040})
    print(combined.map(lambda x: f"{x*100:+.1f}%").to_string())


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
