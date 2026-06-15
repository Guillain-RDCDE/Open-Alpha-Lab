"""Offline, deterministic demo — Study 169 (Fluent-Tickers).

No network. Builds a synthetic cross-sectional panel and shows the study's spine in one
screen: the fluency long-short portfolio only beats a random-label control when a
premium is actually planted in the data — and on a martingale (premium = 0) the two
are indistinguishable. Run:

    python studies/169-fluent-tickers/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fluent_tickers import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 169 — Fluent-Tickers — synthetic positive control\n")
    print(
        f"{'planted premium':>18} | {'fluent bps/yr':>13} {'t':>6} | "
        f"{'random bps/yr':>13} {'t':>6} | beats random?"
    )
    print("-" * 85)

    for premium in (0.00, 0.05, 0.10, -0.05):
        prices, mask, _ = data.synthetic_panel(
            n_stocks=40, n_fluent=20, n_days=2520,
            fluency_premium=premium, seed=169
        )
        port_fluent = st.build_ls_portfolio(prices, mask, cost_bps=0.0)
        port_random = st.build_random_ls_portfolio(prices, mask, seed=42, cost_bps=0.0)

        annual_fluent = st.annual_ls_returns(port_fluent)
        annual_random = st.annual_ls_returns(port_random)

        sf = st.summarize(annual_fluent)
        sr = st.summarize(annual_random)

        # "beats random?" means the fluency arm's mean exceeds the coin arm's mean
        # AND the absolute level is above noise (|t|>1 on the fluency arm).
        beats = "yes" if sf["mean_bps"] > sr["mean_bps"] and abs(sf.get("tstat", 0) or 0) > 1 else "no"
        print(
            f"{premium:>18.2f} | {sf['mean_bps']:>+13.0f} {sf['tstat']:>+6.2f} | "
            f"{sr['mean_bps']:>+13.0f} {sr['tstat']:>+6.2f} | {beats}"
        )

    print()
    print("The fluency L-S portfolio is a faithful premium harvester: it only beats")
    print("the random-label control when a premium is planted (+0.05/+0.10 annual).")
    print("On a null panel (0.00) the two are statistically indistinguishable.")
    print()
    print("On the real S&P 500 survivor panel the fluency premium is absent —")
    print("see docs/results.md for the honest verdict.")

    # Also show the fluency score distribution for the S&P 500 tickers.
    print()
    print("--- Fluency score distribution for a sample universe ---")
    sample_tickers = [
        "AAPL", "MSFT", "KO", "CAT", "KAR", "RDO", "BRK", "NVDA", "GPN",
        "VLO", "DVN", "XOM", "MCD", "V", "MA", "UNH", "IBM", "AIG",
    ]
    scores = data.score_tickers(sample_tickers)
    mask_sample = data.classify_tickers(sample_tickers)
    print(f"{'Ticker':>8} {'Score':>7} {'Label':>10}")
    print("-" * 30)
    for t in sorted(sample_tickers, key=lambda x: scores[x], reverse=True):
        label = "FLUENT" if mask_sample[t] else "NON-FLUENT"
        print(f"{t:>8} {scores[t]:>7.3f} {label:>10}")


if __name__ == "__main__":
    main()
