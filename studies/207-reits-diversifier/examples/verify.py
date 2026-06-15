"""Real-tape verification — Study 207 (REITs-Diversifier). Regenerates docs/results.md numbers.

Reads (or fetches) the cross-asset ETF cache and the Shiller CPI dataset, runs the
REIT-sleeve portfolio analysis against 60/40 and SPY, computes rolling correlations,
crisis drawdown episodes, and CPI-regime breakdowns. Network is touched only with --fetch.

    python studies/207-reits-diversifier/examples/verify.py            # cache-only
    python studies/207-reits-diversifier/examples/verify.py --fetch    # refresh the panel
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reits_diversifier import data, strategy as st  # noqa: E402

SHILLER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_cache", "shiller_sp500.parquet")
)


def main(fetch: bool) -> None:
    prices = data.load_real(fetch=fetch)
    print(f"Window: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Fingerprint: {data.fingerprint(prices)}")
    print(f"Shape: {prices.shape}\n")

    ret = st.to_returns(prices)
    rf = ret["SHY"]

    # ------------------------------------------------------------------ portfolios
    sleeve = st.reit_sleeve(ret, reit_wt=0.20, eq_wt=0.60, cost_bps=1.0)
    bench = st.sixty_forty(ret, cost_bps=1.0)
    spy = st.spy_only(ret)

    s_sleeve = st.portfolio_stats(sleeve, rf=rf)
    s_bench = st.portfolio_stats(bench, rf=rf)
    s_spy = st.portfolio_stats(spy, rf=rf)

    print("=== Portfolio Stats (1 bp one-way cost) ===")
    for name, s in [("60/20/20 REIT sleeve", s_sleeve), ("60/40 benchmark", s_bench), ("SPY only", s_spy)]:
        print(
            f"{name:22s}: CAGR={s['cagr']:+.2%} vol={s['vol']:.2%} "
            f"Sharpe={s['sharpe']:.3f} maxDD={s['max_dd']:.1%} worst_yr={s['worst_year']:.1%}"
        )

    # ------------------------------------------------------------------ inference
    t_vs_bench = st.hac_tstat_annual(sleeve, bench)
    t_vs_spy = st.hac_tstat_annual(sleeve, spy)
    print(f"\nHAC t (sleeve vs 60/40): {t_vs_bench:+.3f}")
    print(f"HAC t (sleeve vs SPY):   {t_vs_spy:+.3f}")

    bst = st.bootstrap_sharpe_diff(sleeve, bench, rf=rf, seed=207)
    print(
        f"\nBootstrap Sharpe diff (sleeve − 60/40): "
        f"point={bst['point']:+.3f} 95% CI=[{bst['ci95'][0]:+.3f},{bst['ci95'][1]:+.3f}] "
        f"sleeve wins {bst['frac_a_wins']*100:.1f}% of resamples"
    )

    # ------------------------------------------------------------------ correlations
    full_corr = ret["VNQ"].corr(ret["SPY"])
    rc_63 = st.rolling_correlation(ret, "VNQ", "SPY", window=63).dropna()
    rc_252 = st.rolling_correlation(ret, "VNQ", "SPY", window=252).dropna()
    print(f"\nVNQ-SPY full-sample correlation: {full_corr:.3f}")
    print(f"Rolling 63-day: mean={rc_63.mean():.3f} min={rc_63.min():.3f} max={rc_63.max():.3f}")
    print(f"Rolling 252-day: mean={rc_252.mean():.3f} min={rc_252.min():.3f} max={rc_252.max():.3f}")
    print(f"VNQ-TLT correlation: {ret['VNQ'].corr(ret['TLT']):.3f}")

    # ------------------------------------------------------------------ crisis episodes
    episodes = st.equity_drawdown_episodes(ret, stock="SPY", thresh=-0.15)
    print(f"\n=== Crisis Episodes (SPY drawdown > −15%) ===")
    for ep in episodes:
        vnq = ep["others"].get("VNQ", float("nan"))
        tlt = ep["others"].get("TLT", float("nan"))
        gld = ep["others"].get("GLD", float("nan"))
        print(
            f"  {ep['peak'].date()} → {ep['trough'].date()}: "
            f"SPY={ep['stock_loss']:.1%}  VNQ={vnq:.1%}  TLT={tlt:.1%}  GLD={gld:.1%}"
        )
    vnq_cushioned = sum(
        1 for ep in episodes if ep["others"].get("VNQ", 0) > ep["stock_loss"] + 0.05
    )
    print(f"VNQ cushioned (outperformed SPY by >5pp) in {vnq_cushioned}/{len(episodes)} episodes")

    # ------------------------------------------------------------------ CPI regimes
    if os.path.exists(SHILLER_PATH):
        regimes = st.cpi_regimes(SHILLER_PATH)
        reg_vnq = regimes[(regimes.index >= 2004) & (regimes.index <= 2026)]
        vnq_reg = st.returns_by_cpi_regime(ret["VNQ"], reg_vnq)
        spy_reg = st.returns_by_cpi_regime(ret["SPY"], reg_vnq)
        print("\n=== Annual Mean Return by CPI Regime (2004–2026) ===")
        for label in ("low", "medium", "high"):
            vr, sr = vnq_reg[label], spy_reg[label]
            print(
                f"  {label:8s}: VNQ mean={vr['mean']:+.1%} (n={vr['n']})  "
                f"SPY mean={sr['mean']:+.1%} (n={sr['n']})"
            )
    else:
        print("\n[Shiller CPI data not found — skipping regime analysis]")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
