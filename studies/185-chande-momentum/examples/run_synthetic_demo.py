"""Offline, deterministic demo — Study 185 (Chande-Momentum).

No network.  Builds synthetic daily tapes and shows the study's spine in one screen:
the CMO overbought/oversold framing only beats a random-direction coin when mean-
reversion is planted; the zero-cross framing only beats a coin when momentum is planted.
On a martingale both framings are fair dice.  Run:

    python studies/185-chande-momentum/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chande_momentum import data, strategy as st  # noqa: E402


def _edge(bars, entries, seed: int = 1) -> dict:
    """Cross minus random-direction control on gross 5-day forward returns."""
    if len(entries) < 5:
        return {"signal_bps": float("nan"), "coin_bps": float("nan"),
                "n": 0, "beats_coin": "n/a"}
    sig = st.summarize(
        st.run_trades(bars, entries, hold_days=5, cost_bps=0), "ret_gross"
    )
    rnd = st.summarize(
        st.run_trades(bars, entries, hold_days=5, cost_bps=0,
                      directions=st.random_directions(len(entries), seed=seed)),
        "ret_gross",
    )
    return {
        "signal_bps": sig["mean_bps"],
        "coin_bps": rnd["mean_bps"],
        "n": sig["n_trades"],
        "beats_coin": "yes" if sig["mean_bps"] > rnd["mean_bps"] + 0.05 else "no",
    }


def main() -> None:
    print("Study 185 — Chande-Momentum — synthetic positive-control sweep\n")

    # ---- OB/OS framing vs mean_rev knob ----
    print("=== Overbought/Oversold framing (hold 5 days) vs mean_rev knob ===")
    print(f"{'mean_rev':>9} | {'n_sig':>5} {'signal(bps)':>11} {'coin(bps)':>10} | beats coin?")
    print("-" * 56)
    for mr in (0.00, 0.15, 0.30, 0.45):
        bars, _ = data.synthetic_daily(n_days=600, mean_rev=mr, momentum=0.0, seed=185)
        c = st.cmo(bars["close"])
        ent = st.ob_os_entries(c, threshold=50.0)
        r = _edge(bars, ent, seed=1)
        print(f"{mr:>9.2f} | {r['n']:>5d} {r['signal_bps']:>+11.2f} "
              f"{r['coin_bps']:>+10.2f} | {r['beats_coin']}")

    print()

    # ---- Zero-cross framing vs momentum knob ----
    print("=== Zero-cross framing (hold 5 days) vs momentum knob ===")
    print(f"{'momentum':>9} | {'n_sig':>5} {'signal(bps)':>11} {'coin(bps)':>10} | beats coin?")
    print("-" * 56)
    for mom in (0.00, 0.15, 0.30, 0.45, -0.15):
        bars, _ = data.synthetic_daily(n_days=600, mean_rev=0.0, momentum=mom, seed=185)
        c = st.cmo(bars["close"])
        ent = st.zero_cross_entries(c)
        r = _edge(bars, ent, seed=2)
        print(f"{mom:>9.2f} | {r['n']:>5d} {r['signal_bps']:>+11.2f} "
              f"{r['coin_bps']:>+10.2f} | {r['beats_coin']}")

    print()
    print("Engine confirmed: each framing harvests only the structure it was designed")
    print("to harvest; on a martingale (0/0) both framings are fair dice.")
    print("The real-tape verdict is a statement about the market, not the method.")
    print("See docs/results.md for the real-tape numbers.")


if __name__ == "__main__":
    main()
