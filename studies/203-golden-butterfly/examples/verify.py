"""Real-tape verification — Study 203 (Golden-Butterfly). Regenerates docs/results.md.

Loads the per-study panel (SPY / IWN / TLT / SHY / GLD), runs Tyler's
20/20/20/20/20 Golden Butterfly with annual rebalance against 100% SPY, 60/40,
and the Permanent Portfolio, and prints the headline risk-adjusted stats that
populate docs/results.md.

    python studies/203-golden-butterfly/examples/verify.py            # cache-only
    python studies/203-golden-butterfly/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from golden_butterfly import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    # ---- load ----------------------------------------------------------------
    px = data.load_real(fetch=fetch)
    print("=== Golden-Butterfly real-tape results ===")
    print("Tickers : SPY / IWN / TLT / SHY / GLD")
    print(f"Window  : {px.index[0].date()} -> {px.index[-1].date()}")
    print(f"Fingerprint: {data.fingerprint(px)}")
    print()

    ret = st.to_returns(px)
    rf = ret["SHY"]  # total-return cash proxy

    # ---- portfolios ----------------------------------------------------------
    gb = st.golden_butterfly(ret, cost_bps=1.0)
    pp = st.permanent_portfolio(ret, cost_bps=1.0)
    p60 = st.sixty_forty(ret, cost_bps=1.0)
    spy = st.spy_only(ret)

    gb_s = st.portfolio_stats(gb, rf=rf)
    pp_s = st.portfolio_stats(pp, rf=rf)
    p60_s = st.portfolio_stats(p60, rf=rf)
    spy_s = st.portfolio_stats(spy, rf=rf)

    print("=== Headline stats (annual rebalance, 1 bp one-way cost) ===")
    header = (
        f"{'Portfolio':22s}  {'CAGR':>7}  {'Vol':>6}  {'Sharpe':>7}  "
        f"{'MaxDD':>7}  {'WorstYr':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, s in [
        ("GB 20/20/20/20/20", gb_s),
        ("PP 25/25/25/25", pp_s),
        ("60/40 SPY/TLT", p60_s),
        ("100% SPY", spy_s),
    ]:
        print(
            f"{name:22s}  {s['cagr']*100:>6.2f}%  {s['vol']*100:>5.2f}%  "
            f"{s['sharpe']:>7.3f}  {s['max_dd']*100:>6.1f}%  "
            f"{s['worst_year']*100:>7.1f}%"
        )
    print()

    # ---- inference -----------------------------------------------------------
    t_vs_spy = st.hac_tstat_annual(gb, spy)
    t_vs_pp = st.hac_tstat_annual(gb, pp)
    t_vs_6040 = st.hac_tstat_annual(gb, p60)
    print("=== HAC t-stats on annual return differences (GB minus benchmark) ===")
    print(f"GB minus SPY:    HAC t = {t_vs_spy:+.3f}")
    print(f"GB minus PP:     HAC t = {t_vs_pp:+.3f}")
    print(f"GB minus 60/40:  HAC t = {t_vs_6040:+.3f}")
    print()

    bst_spy = st.bootstrap_sharpe_diff(gb, spy, rf=rf, seed=203)
    bst_pp = st.bootstrap_sharpe_diff(gb, pp, rf=rf, seed=203)
    bst_6040 = st.bootstrap_sharpe_diff(gb, p60, rf=rf, seed=203)
    print("=== Bootstrap Sharpe difference (GB vs benchmark) ===")
    print(
        f"GB vs SPY  : {bst_spy['point']:+.3f}  "
        f"CI95=[{bst_spy['ci95'][0]:+.3f}, {bst_spy['ci95'][1]:+.3f}]  "
        f"GB wins {bst_spy['frac_a_wins']*100:.0f}% of resamples"
    )
    print(
        f"GB vs PP   : {bst_pp['point']:+.3f}  "
        f"CI95=[{bst_pp['ci95'][0]:+.3f}, {bst_pp['ci95'][1]:+.3f}]  "
        f"GB wins {bst_pp['frac_a_wins']*100:.0f}% of resamples"
    )
    print(
        f"GB vs 60/40: {bst_6040['point']:+.3f}  "
        f"CI95=[{bst_6040['ci95'][0]:+.3f}, {bst_6040['ci95'][1]:+.3f}]  "
        f"GB wins {bst_6040['frac_a_wins']*100:.0f}% of resamples"
    )
    print()

    # ---- drawdown episodes ---------------------------------------------------
    episodes = st.equity_drawdowns(ret, "SPY", thresh=-0.10)
    print(f"=== SPY drawdown episodes (>10%), n={len(episodes)} ===")
    for ep in episodes:
        p = ep["peak"].strftime("%Y-%m")
        t = ep["trough"].strftime("%Y-%m")
        others = ep["others"]
        print(
            f"  {p}->{t}  SPY={ep['stock_loss']*100:+.1f}%  "
            f"IWN={others.get('IWN', 0)*100:+.1f}%  "
            f"TLT={others.get('TLT', 0)*100:+.1f}%  "
            f"GLD={others.get('GLD', 0)*100:+.1f}%  "
            f"SHY={others.get('SHY', 0)*100:+.1f}%"
        )

    print()
    print("=== Annual returns ===")
    import pandas as pd
    ar_gb = st.annual_returns(gb)
    ar_pp = st.annual_returns(pp)
    ar_spy = st.annual_returns(spy)
    ar_6040 = st.annual_returns(p60)
    combined = pd.DataFrame({"GB": ar_gb, "PP": ar_pp, "SPY": ar_spy, "60/40": ar_6040})
    print(combined.map(lambda x: f"{x*100:+.1f}%").to_string())


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
