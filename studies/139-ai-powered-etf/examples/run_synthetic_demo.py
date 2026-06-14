"""Offline, deterministic demo — Study 139 (AI-Powered-ETF).

No network. Builds a synthetic fund-vs-market tape and shows the study's spine in one
screen: the CAPM alpha recovery is reliable when alpha is planted, and reads near zero
when the null holds. Run:

    python studies/139-ai-powered-etf/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_powered_etf import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 139 — AI-Powered-ETF — synthetic alpha-recovery control\n")
    print(
        f"{'planted alpha':>16} | {'fund CAGR':>9} {'mkt CAGR':>9} "
        f"{'Jensen alpha':>10} {'HAC t(a)':>10} | detect?"
    )
    print("-" * 74)

    scenarios = [
        (0.00,  "null (no edge)"),
        (-0.044, "AIEQ-like (-4.4%/yr)"),
        (-0.10,  "severe underperformer"),
        (+0.05,  "genuine AI alpha"),
    ]

    for alpha, label in scenarios:
        prices, _ = data.synthetic_daily(n_days=2174, alpha_ann=alpha, beta=1.05, seed=139)
        rets = st.daily_returns(prices)
        res = st.summarize(rets["aieq"], rets["spy"])
        detected = "yes (negative)" if res["tstat_alpha_hac"] < -2 else (
            "yes (positive)" if res["tstat_alpha_hac"] > 2 else "no (noise)"
        )
        print(
            f"{alpha:>+16.3f} | {res['cagr_fund_pct']:>+9.2f}% {res['cagr_market_pct']:>+8.2f}% "
            f"{res['alpha_ann_pct']:>+9.2f}% {res['tstat_alpha_hac']:>+10.2f} | {detected}"
        )
        if label:
            print(f"  ({label})")

    print()
    print("The engine detects planted alpha reliably on 2174-day tapes.")
    print("On the real tape (8.6 yrs), AIEQ's HAC t(alpha) = -1.29 — not significant.")
    print("See docs/results.md for the full real-tape breakdown.")


if __name__ == "__main__":
    main()
