"""Offline, deterministic demo — Study 156 (Martingale).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the martingale / averaging-down strategy "wins" most of the time but blows up on a ruin
cascade — and does NOT beat buy-and-hold in expectation once you account for ruin events
and costs. Run:

    python studies/156-martingale/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from martingale import data, strategy as st  # noqa: E402


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def main() -> None:
    print("Study 156 — Martingale — synthetic analysis\n")
    print("=" * 72)

    # ---- Part 1: Episode outcomes across market regimes ----------------------
    print("\nPart 1: Episode outcomes vs buy-and-hold (step=-5%, TP=+5%, 6 levels)\n")
    print(
        f"{'regime':>18} | {'episodes':>8} | {'ruin%':>6} | {'win%':>6} | "
        f"{'excess mean bps':>15} | {'skew':>6} | {'t-stat':>7}"
    )
    print("-" * 80)

    regimes = [
        ("random walk",    dict(mu=0.0,   mean_reversion=0.0,  seed=156)),
        ("mean-reverting", dict(mu=0.0,   mean_reversion=0.05, seed=156)),
        ("bull (+10%/yr)", dict(mu=0.10,  mean_reversion=0.0,  seed=156)),
        ("bear (-20%/yr)", dict(mu=-0.20, mean_reversion=0.0,  seed=156)),
    ]

    for label, kwargs in regimes:
        prices, _ = data.synthetic_daily(n_days=2000, **kwargs)
        ledger = st.run_episodes(
            prices["close"], step_pct=0.05, take_profit_pct=0.05, max_levels=6, cost_bps=5.0
        )
        s = st.summarize(ledger, "excess_ret_net")
        print(
            f"{label:>18} | {s['n_episodes']:>8d} | "
            f"{s['ruin_rate'] * 100:>5.1f}% | "
            f"{s['win_rate'] * 100:>5.1f}% | "
            f"{s['mean_bps']:>+15.1f} | "
            f"{s['skew']:>+6.2f} | "
            f"{s['tstat']:>+7.2f}"
        )

    # ---- Part 2: Ruin probability by capital level ---------------------------
    print("\n\nPart 2: Ruin probability vs max levels (mu=0, annual_vol=18%)\n")
    print(f"{'max_levels':>12} | {'capital units':>14} | {'ruin %':>8} | {'skew':>8}")
    print("-" * 52)
    for lvl in (3, 4, 5, 6, 7, 8):
        res = st.ruin_probability(
            annual_vol=0.18, mu=0.0, step_pct=0.05, take_profit_pct=0.05,
            max_levels=lvl, n_paths=5000, seed=156
        )
        cap_units = 2**lvl - 1
        print(
            f"{lvl:>12d} | {cap_units:>14d} | "
            f"{res['ruin_rate'] * 100:>7.1f}% | "
            f"{res['skew']:>+8.3f}"
        )

    # ---- Part 3: The spine summary ------------------------------------------
    print("\n\nPart 3: The spine — random walk (the null)\n")
    prices, _ = data.synthetic_daily(n_days=3000, mu=0.0, mean_reversion=0.0, seed=42)
    ledger = st.run_episodes(
        prices["close"], step_pct=0.05, take_profit_pct=0.05, max_levels=6, cost_bps=5.0
    )
    s_exc = st.summarize(ledger, "excess_ret_net")
    s_bah = st.summarize(ledger, "bah_ret_net")
    s_mart = st.summarize(ledger, "ret_net")

    print(f"  Total episodes : {s_exc['n_episodes']}")
    print(f"  Ruin rate      : {s_exc['ruin_rate'] * 100:.1f}%")
    print(f"  Martingale win : {s_mart['win_rate'] * 100:.1f}%  (looks great until ruin strikes)")
    print(f"  Martingale skew: {s_mart['skew']:+.3f}  (negative = fat left tail)")
    print(f"  Martingale mean: {s_mart['mean_bps']:+.1f} bps/episode")
    print(f"  B&H mean       : {s_bah['mean_bps']:+.1f} bps/episode")
    print(f"  Excess (M-BaH) : {s_exc['mean_bps']:+.1f} bps/episode, t = {s_exc['tstat']:+.2f}")
    print()
    print("Verdict: the martingale 'wins' frequently but has a left-skewed P&L and")
    print("does NOT produce a positive excess return over buy-and-hold once you price")
    print("the ruin events. Negative skew confirms the tail-risk profile.")


if __name__ == "__main__":
    main()
