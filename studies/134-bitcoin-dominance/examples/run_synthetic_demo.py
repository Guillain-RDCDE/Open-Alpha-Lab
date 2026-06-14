"""Offline, deterministic demo — Study 134 (Bitcoin-Dominance).

No network. Builds a synthetic crypto panel and shows the study's spine in one screen:
the dominance→alt-spread signal only beats the base-rate control when a genuine
relationship is planted — on a null tape (dom_signal=0) the strategy is a coincident
rotation indicator, not a forecaster. Run:

    python studies/134-bitcoin-dominance/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bitcoin_dominance import data, strategy as st  # noqa: E402


def _run_signal(panel, lookback: int = 10, horizon: int = 5) -> dict:
    sig = st.dominance_signal(panel, lookback=lookback)
    pairs = st.signal_outcome_pairs(panel, sig, horizon=horizon)
    # Conditional: alt-season periods (falling dominance, sig < 0)
    alt_season = pairs[pairs["signal"] < 0]["spread_fwd"]
    # Unconditional base rate
    base_rate = pairs["spread_fwd"]
    # Regression
    reg = st.regression_tstat(pairs["spread_fwd"], pairs["signal"])
    s_cond = st.summarize(alt_season, label="alt-season")
    s_base = st.summarize(base_rate, label="base-rate")
    return {
        "n_total": len(pairs),
        "n_alt_season": len(alt_season),
        "cond_mean_pct": s_cond["mean_pct"],
        "cond_t": s_cond["tstat"],
        "base_mean_pct": s_base["mean_pct"],
        "base_t": s_base["tstat"],
        "slope": reg["slope"],
        "reg_t": reg["tstat"],
    }


def main() -> None:
    print("Study 134 — Bitcoin-Dominance — synthetic positive control\n")
    print(
        f"{'dom_signal':>12} | {'n':>5} {'n_alts':>7} | "
        f"{'cond mean%':>10} {'cond t':>7} | "
        f"{'base mean%':>10} {'base t':>7} | "
        f"{'reg slope':>9} {'reg t':>7}"
    )
    print("-" * 90)

    for dom_sig in (0.0, 0.5, 1.0, 2.0):
        panel, _ = data.synthetic_daily(n_years=5, dom_signal=dom_sig, seed=134)
        r = _run_signal(panel)
        print(
            f"{dom_sig:>12.1f} | {r['n_total']:>5} {r['n_alt_season']:>7} | "
            f"{r['cond_mean_pct']:>+10.3f} {r['cond_t']:>+7.2f} | "
            f"{r['base_mean_pct']:>+10.3f} {r['base_t']:>+7.2f} | "
            f"{r['slope']:>+9.4f} {r['reg_t']:>+7.2f}"
        )

    print()
    print("At dom_signal=0 (null): regression slope is near zero, no conditional edge.")
    print("With dom_signal>0 (planted): slope turns more negative (falling dom -> higher spread).")
    print("The real tape has no planted signal -- see docs/results.md for the verdict.")


if __name__ == "__main__":
    main()
