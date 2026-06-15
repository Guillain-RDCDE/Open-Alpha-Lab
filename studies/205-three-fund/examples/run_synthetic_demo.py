"""Offline, deterministic demo — Study 205 (Three-Fund).

No network. Builds synthetic price paths and shows the study's spine:
  - Under null (intl_alpha=0): the three-fund's Sharpe is very close to the
    US-only and 60/40 baselines — no statistically distinguishable edge.
  - With positive intl_alpha: three-fund wins on CAGR.
  - With negative intl_alpha: three-fund lags.

Run:
    python studies/205-three-fund/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from three_fund import data, strategy as st  # noqa: E402


def run_scenario(intl_alpha: float, n_days: int = 2520, seed: int = 205) -> None:
    prices, truth = data.synthetic_prices(n_days=n_days, intl_alpha=intl_alpha, seed=seed)
    navs = st.run_all_portfolios(prices, cost_bps=5.0)

    print(f"\n{'Strategy':>14} | {'CAGR%':>7} | {'Sharpe':>7} | {'MaxDD%':>7} | {'t-ann':>7}")
    print("-" * 58)
    for name, nav in navs.items():
        m = st.summarize(nav)
        cagr  = m["cagr"]  * 100
        mdd   = m["max_dd"] * 100
        print(
            f"{name:>14} | {cagr:>7.2f} | {m['sharpe']:>7.3f} | "
            f"{mdd:>7.2f} | {m['tstat_annual']:>+7.2f}"
        )

    # HAC pairwise diffs
    diff_vs_us  = st.hac_tstat_diff(navs["three_fund"], navs["us_only"])
    diff_vs_6040 = st.hac_tstat_diff(navs["three_fund"], navs["sixty_forty"])
    intl_attr    = st.international_attribution(prices, cost_bps=5.0)
    print(f"\n  three_fund vs us_only:   mean_diff={diff_vs_us['mean_diff']*100:+.2f}% ann, "
          f"HAC t={diff_vs_us['tstat']:+.2f}")
    print(f"  three_fund vs sixty_40:  mean_diff={diff_vs_6040['mean_diff']*100:+.2f}% ann, "
          f"HAC t={diff_vs_6040['tstat']:+.2f}")
    print(f"  intl attribution (3f vs 90/10): mean_diff={intl_attr['mean_diff']*100:+.2f}% ann, "
          f"HAC t={intl_attr['tstat']:+.2f}")


def main() -> None:
    print("Study 205 -- Three-Fund -- synthetic spine")
    print("=" * 72)

    print("\nSCENARIO A: null (intl_alpha=0.0) -- international earns same as US")
    print("  Expect: three-fund sharpe ~ us-only sharpe; no clear winner")
    run_scenario(intl_alpha=0.0)

    print("\n\nSCENARIO B: positive international alpha (+2%/yr)")
    print("  Expect: three-fund wins on CAGR, intl attribution positive")
    run_scenario(intl_alpha=0.02)

    print("\n\nSCENARIO C: negative international alpha (-2%/yr) -- the real-world 2011-2026")
    print("  Expect: three-fund lags US-only; intl attribution negative")
    run_scenario(intl_alpha=-0.02)

    print("\n")
    print("Key insight: the three-fund's edge vs US-only is determined almost")
    print("entirely by whether international stocks outperform. On the actual")
    print("2011-2026 sample, US equity dominated and VXUS was a return drag,")
    print("making the three-fund's Sharpe lower -- but its drawdown also lower.")
    print("The honest verdict: sensible, not an edge vs simpler alternatives.")


if __name__ == "__main__":
    main()
