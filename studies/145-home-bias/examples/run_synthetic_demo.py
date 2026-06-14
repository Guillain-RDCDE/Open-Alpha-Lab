"""Offline, deterministic demo — Study 145 (Home-Bias).

No network. Builds synthetic daily return panels and shows the study's spine
in one screen: the global blend (60/25/15 SPY/EFA/EEM) improves the Sharpe
ratio only when correlations are low and US and international have the same
expected return. When the US has a structural edge (post-2009 regime), the
blend drags on return while delivering only modest risk reduction.

Run:
    python studies/145-home-bias/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from home_bias import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 145 — Home-Bias — synthetic positive / null controls\n")

    scenarios = [
        ("Equal mu, low corr (null + diversification benefit)",
         dict(mu_us=0.07 / 252, mu_intl=0.07 / 252, corr_intl=0.40)),
        ("Equal mu, high corr (no diversification benefit)",
         dict(mu_us=0.07 / 252, mu_intl=0.07 / 252, corr_intl=0.90)),
        ("US dominant +4%/yr, moderate corr (post-2009 regime)",
         dict(mu_us=0.11 / 252, mu_intl=0.07 / 252, corr_intl=0.70)),
        ("Perfect correlation (upper bound on US advantage)",
         dict(mu_us=0.07 / 252, mu_intl=0.07 / 252, corr_intl=1.00)),
    ]

    header = (
        f"{'Scenario':<50} | {'US Sharpe':>9} {'Blend Sharpe':>12} "
        f"{'Delta':>7} | {'US maxDD':>9} {'Blend maxDD':>11}"
    )
    print(header)
    print("-" * len(header))

    for label, kw in scenarios:
        ret, _ = data.synthetic_panel(n_days=5000, seed=145, **kw)
        us = st.us_only(ret)
        blend = st.rolling_blend(ret)
        su = st.portfolio_stats(us)
        sb = st.portfolio_stats(blend)
        delta = sb["sharpe"] - su["sharpe"]
        print(
            f"{label:<50} | {su['sharpe']:>+9.3f} {sb['sharpe']:>+12.3f} "
            f"{delta:>+7.3f} | {su['max_dd']:>9.1%} {sb['max_dd']:>11.1%}"
        )

    print()
    print("The blend beats US-only on Sharpe only when correlation is low AND")
    print("returns are equal. US structural advantage (mu_spread > 0) reduces")
    print("the blend's return advantage below zero even as risk falls slightly.")
    print("On the real tape (2003-2026), the US outperformed by ~4%/yr — see docs/results.md.")


if __name__ == "__main__":
    main()
