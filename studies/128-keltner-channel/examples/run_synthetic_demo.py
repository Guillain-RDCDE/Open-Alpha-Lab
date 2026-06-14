"""Offline, deterministic demo — Study 128 (Keltner-Channel).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the contradictory Keltner rules (breakout vs reversion) vs a random-entry baseline, across
a sweep of planted momentum values.

    python studies/128-keltner-channel/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keltner_channel import data, strategy as st  # noqa: E402


def _arm(bars, side, hold=20, seed=0):
    ent = st.channel_entries(bars, side=side)
    if len(ent) < 5:
        return dict(n_trades=0, mean_bps=float("nan"), tstat=float("nan"))
    sig = st.summarize(st.run_trades(bars, ent, hold_days=hold, cost_bps=0), "ret_gross")
    rand_ent = st.random_entries(bars, n=len(ent), seed=seed)
    rand = st.summarize(st.run_trades(bars, rand_ent, hold_days=hold, cost_bps=0), "ret_gross")
    sig["rand_mean"] = rand["mean_bps"]
    sig["delta"] = sig["mean_bps"] - rand["mean_bps"]
    return sig


def main() -> None:
    print("Study 128 — Keltner-Channel — synthetic positive control\n")
    header = f"{'momentum':>10} | {'side':>10} | {'n':>5} | {'mean bps':>9} | {'vs random':>10} | {'t':>6} | signal?"
    print(header)
    print("-" * len(header))

    for mom in (-0.20, -0.10, 0.0, 0.10, 0.20):
        bars, _ = data.synthetic_daily(n_days=1000, momentum=mom, seed=128)
        for side in ("breakout", "reversion"):
            r = _arm(bars, side)
            signal = "yes" if abs(r.get("tstat", 0) or 0) > 1.5 and (r.get("delta", 0) or 0) > 0 else "no"
            print(
                f"{mom:>10.2f} | {side:>10} | {r['n_trades']:>5} | "
                f"{r['mean_bps']:>+9.2f} | {r.get('delta', float('nan')):>+10.2f} | "
                f"{r['tstat']:>+6.2f} | {signal}"
            )

    print()
    print("Spine: on a martingale (momentum=0) both arms are fair coins vs random.")
    print("With momentum>0 the breakout arm edges random; with momentum<0 the reversion arm does.")
    print("The real tape result is in docs/results.md — most likely both arms are noise.")


if __name__ == "__main__":
    main()
