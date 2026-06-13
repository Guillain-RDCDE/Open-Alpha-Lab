"""Offline, deterministic demo - Study 104 (Bollinger-Reversion).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the lower-band entry only beats a random-day control when the tape actually carries
mean-reversion -- and on a martingale (reversion = 0) it is a fair coin.  Run:

    python studies/104-bollinger-reversion/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bollinger_reversion import data, strategy as st  # noqa: E402


def _empty_summary() -> dict:
    """Return a zero-filled summary when there are no trades."""
    return {"n_trades": 0, "mean_bps": float("nan"), "tstat": float("nan")}


def _safe_summarize(led: "pd.DataFrame", col: str = "ret_gross") -> dict:
    """Return summarize() or an empty dict if the ledger has no rows/col."""
    if led.empty or col not in led.columns:
        return _empty_summary()
    return st.summarize(led, col)


def _run(bars, rev_label: str, max_hold: int = 40) -> None:
    ent_rev = st.band_entries(bars["close"], side="reversion")
    ent_brk = st.band_entries(bars["close"], side="breakout")
    n_rev = len(ent_rev)
    ent_rand = st.random_entries(bars, n=max(n_rev, 1), seed=42)

    led_rev = st.run_trades(bars, ent_rev, exit_mode="mid", cost_bps=0.0, max_hold=max_hold)
    led_brk = st.run_trades(bars, ent_brk, exit_mode="mid", cost_bps=0.0, max_hold=max_hold)
    led_rand = st.run_trades(bars, ent_rand, exit_mode="time", cost_bps=0.0, max_hold=max_hold)

    s_rev = _safe_summarize(led_rev)
    s_brk = _safe_summarize(led_brk)
    s_rand = _safe_summarize(led_rand)

    beats = (
        "YES (band wins)" if s_rev["mean_bps"] > s_rand["mean_bps"] + 5 else
        "no  (~random)"
    )
    print(
        f"  {rev_label:22s} | n_rev={n_rev:4d}  reversion={s_rev['mean_bps']:+7.1f}bps "
        f"t={s_rev['tstat']:+5.2f}  rand={s_rand['mean_bps']:+7.1f}bps  "
        f"breakout={s_brk['mean_bps']:+7.1f}bps | beats random? {beats}"
    )


def main() -> None:
    print("Study 104 - Bollinger-Reversion - synthetic positive control\n")
    print(
        f"  {'planted reversion':22s} | {'n_rev':>6}  {'rev mean':>14}  "
        f"{'t':>6}  {'rand mean':>14}  {'brk mean':>16} | beats random?"
    )
    print("  " + "-" * 110)

    params = [
        ("reversion=0.00 (RW)", 0.00),
        ("reversion=0.05", 0.05),
        ("reversion=0.10", 0.10),
        ("reversion=0.20", 0.20),
        ("reversion=-0.05 (mom)", -0.05),
    ]
    for label, rev in params:
        bars, _ = data.synthetic_daily(n_days=1500, reversion=rev, seed=104)
        _run(bars, label)

    print()
    print("The lower-band entry is a faithful mean-reversion harvester:")
    print("  - At reversion=0 (random walk) it matches the random baseline -- a fair coin.")
    print("  - At reversion>0 it outperforms -- the band pierce locates the cheapest buys.")
    print("  - At reversion<0 (momentum) it underperforms -- the wrong side of the move.")
    print()
    print("On the real SPY daily tape (see docs/results.md) the effect is WEAK: positive")
    print("mean but HAC |t| < 2, and it does not clearly beat a random-day control.")


if __name__ == "__main__":
    main()
