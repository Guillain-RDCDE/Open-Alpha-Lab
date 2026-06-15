"""Offline, deterministic demo — Study 202 (Fifty-Two-Week-Low).

No network.  Builds a synthetic daily close panel and shows the study's spine
in one screen: the Q1-minus-Q5 long-short spread (near-52w-low minus near-52w-high)
is positive under mean-reversion (contrarian wins), near zero under a martingale,
and *negative* under momentum (losers keep losing).  Run:

    python studies/202-fifty-two-week-low/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fifty_two_week_low import data, strategy as st  # noqa: E402


def _spread_stats(reversion: float) -> tuple[float, float, int]:
    """Return (Q1_minus_Q5_mean_bps, HAC_t, n) for a given reversion."""
    panel, _ = data.synthetic_panel(n_stocks=20, n_days=2500, reversion=reversion, seed=42)
    prox = st.proximity_to_52w_low(panel, window=252)
    fwd = st.forward_returns(panel, hold_days=5)
    q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)
    spread = q["Q1_minus_Q5"].dropna()
    s = st.summarize(spread)
    return s["mean_bps"], s["tstat"], s["n"]


def main() -> None:
    print("Study 202 — Fifty-Two-Week-Low — synthetic positive control")
    print("Metric: Q1 (near 52w-low) minus Q5 (near 52w-high) per-period return\n")
    print(f"{'planted AR(1)':>14} | {'description':>18} | {'spread (bps)':>13} {'HAC t':>7} | regime")
    print("-" * 72)
    cases = [
        (-0.40, "strong momentum"),
        (-0.20, "mild momentum"),
        ( 0.00, "martingale"),
        ( 0.20, "mild reversion"),
        ( 0.40, "strong reversion"),
    ]
    for rev, desc in cases:
        mean_bps, t, _ = _spread_stats(rev)
        regime = ("losers-keep-losing" if mean_bps < -5
                  else "contrarian-wins" if mean_bps > 5
                  else "noise")
        print(f"{rev:>14.2f} | {desc:>18s} | {mean_bps:>+13.2f} {t:>+7.2f} | {regime}")

    print()
    print("Key: the engine is a faithful mean-reversion detector.")
    print("  Strong momentum (-0.40):  losers keep losing — contrarian LOSES badly.")
    print("  Martingale (0.00):        no structure — spread near zero, insignificant.")
    print("  Strong reversion (+0.40): fallen stocks bounce — contrarian WINS.")
    print()
    print("On the real S&P 500 tape, see docs/results.md for the honest verdict.")
    print("(Spoiler: the real tape is closer to the momentum / martingale regime.)")


if __name__ == "__main__":
    main()
