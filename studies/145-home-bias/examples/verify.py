"""Real-tape verification — Study 145 (Home-Bias). Regenerates docs/results.md numbers.

Reads the shared cross_asset_etfs.parquet for SPY / EFA / EEM, runs the global-blend
backtest (60/25/15, annual rebalancing), compares Sharpe and drawdown against US-only,
tests crisis-period correlation, and reports the HAC t-stat on the return difference.

    python studies/145-home-bias/examples/verify.py            # cache-only
    python studies/145-home-bias/examples/verify.py --fetch    # refresh prices
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from home_bias import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_panel(fetch=fetch)
    if ret.empty:
        print("No cached panel data."); return

    print(f"\nHome-Bias — SPY / EFA / EEM daily log-returns")
    print(f"Window: {ret.index[0].date()} -> {ret.index[-1].date()}  |  n = {len(ret)} days\n")

    # --- Per-asset stats ---
    print("=== Per-asset statistics ===")
    for col in ret.columns:
        s = st.portfolio_stats(ret[col])
        print(f"  {col:4s}  ann={s['ann_return']:+.1%}  vol={s['ann_vol']:.1%}  "
              f"sharpe={s['sharpe']:+.2f}  maxDD={s['max_dd']:.1%}")

    print()

    # --- Portfolio comparison ---
    tbl = st.compare_portfolios(ret)
    print("=== US-only vs global blend (60/25/15 SPY/EFA/EEM, annual rebal) ===")
    t = tbl.copy()
    t["ann_return"] = (t["ann_return"] * 100).round(2)
    t["ann_vol"] = (t["ann_vol"] * 100).round(2)
    t["sharpe"] = t["sharpe"].round(3)
    t["max_dd"] = (t["max_dd"] * 100).round(1)
    print(t[["ann_return", "ann_vol", "sharpe", "max_dd", "n"]].to_string())
    print()

    # --- Mean excess return and HAC t-stat ---
    blend = st.rolling_blend(ret)
    us = st.us_only(ret)
    hac = st.mean_excess_hac(blend, us)
    print(f"=== Mean excess return: blend - US-only ===")
    print(f"  mean = {hac['mean_bps']:+.2f} bps/day  |  HAC t = {hac['tstat']:+.2f}  "
          f"|  n = {hac['n']}")
    print()

    # --- Sharpe difference bootstrap ---
    boot = st.sharpe_diff_bootstrap(blend, us, n_boot=2000, seed=145)
    print(f"=== Sharpe difference (blend - US): block-bootstrap 95% CI ===")
    print(f"  point estimate: {boot['sharpe_diff']:+.3f}")
    print(f"  95% CI: [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]")
    print(f"  frac of resamples where blend < US: {boot['frac_blend_below_us']:.0%}")
    print()

    # --- Crisis vs calm correlation ---
    print("=== Crisis vs calm correlation (bottom 20% SPY days = crisis) ===")
    crisis = st.crisis_vs_calm(ret)
    print(crisis.round(3).to_string())
    print()

    # --- Rolling correlation extremes ---
    corr_efa = st.rolling_correlation(ret, "SPY", "EFA")
    corr_eem = st.rolling_correlation(ret, "SPY", "EEM")
    print(f"=== 63-day rolling correlation (min / mean / max) ===")
    print(f"  SPY-EFA:  min={corr_efa.min():.2f}  mean={corr_efa.mean():.2f}  max={corr_efa.max():.2f}")
    print(f"  SPY-EEM:  min={corr_eem.min():.2f}  mean={corr_eem.mean():.2f}  max={corr_eem.max():.2f}")
    print()

    # --- Fingerprint ---
    fp = data.fingerprint(ret)
    print(f"Panel fingerprint: {fp}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
