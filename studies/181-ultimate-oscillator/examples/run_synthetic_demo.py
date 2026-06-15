"""Offline, deterministic demo — Study 181 (Ultimate-Oscillator).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the Ultimate Oscillator's oversold (<30) buy rule only beats a random-direction coin
when the tape actually carries mean reversion — and on a martingale (mean_rev=0) it
is a fair coin. Run:

    python studies/181-ultimate-oscillator/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultimate_oscillator import data, strategy as st  # noqa: E402


def _pool_spine(mean_rev: float, n_tapes: int = 20, hold: int = 5, seed: int = 0) -> dict:
    """Pool UO threshold signals across multiple synthetic tapes to get stable statistics."""
    from ultimate_oscillator import data as d
    all_sig, all_rand = [], []
    for s in range(n_tapes):
        bars, _ = d.synthetic_daily(n_days=500, mean_rev=mean_rev, seed=seed + s)
        uo = st.ultimate_oscillator(bars)
        ent = st.threshold_entries(uo, oversold=30.0, overbought=70.0)
        if len(ent) == 0:
            continue
        all_sig.append(st.run_trades(bars, ent, hold_days=hold, cost_bps=0))
        all_rand.append(st.run_trades(bars, ent, hold_days=hold, cost_bps=0,
                                      directions=st.random_directions(len(ent), seed=s + 99)))
    if not all_sig:
        return {"n": 0, "uo_mean": float("nan"), "rand_mean": float("nan"),
                "uo_t": float("nan"), "beats": "—"}
    import pandas as pd
    sig_all = pd.concat(all_sig, ignore_index=True)
    rand_all = pd.concat(all_rand, ignore_index=True)
    uo_s = st.summarize(sig_all, "ret_gross")
    rand_s = st.summarize(rand_all, "ret_gross")
    beats = "yes" if uo_s["mean_bps"] > rand_s["mean_bps"] + 0.5 else "no"
    return {
        "n": uo_s["n_trades"],
        "uo_mean": uo_s["mean_bps"],
        "rand_mean": rand_s["mean_bps"],
        "uo_t": uo_s["tstat"],
        "beats": beats,
    }


def main() -> None:
    print("Study 181 — Ultimate-Oscillator — synthetic positive control")
    print("(Pooled across 20 x 500-day tapes to stabilize small-n statistics)\n")
    print(
        f"{'mean_rev':>10} | {'n_trades':>8} | {'UO bps':>8} {'HAC-t':>7} | "
        f"{'coin bps':>9} | beats coin?"
    )
    print("-" * 70)
    for mr in (0.00, 0.10, 0.20, -0.10):
        r = _pool_spine(mr)
        beats_str = r["beats"]
        t_str = f"{r['uo_t']:>+7.2f}" if r["uo_t"] == r["uo_t"] else "     nan"
        print(
            f"{mr:>10.2f} | {r['n']:>8} | {r['uo_mean']:>+8.2f} {t_str} | "
            f"{r['rand_mean']:>+9.2f} | {beats_str}"
        )

    print()
    print("The UO oscillator fires very infrequently on synthetic tapes — the AR(1)")
    print("mean reversion is too fast for the 28-bar lookback to accumulate extreme")
    print("pressure readings.  On the real tape the signal exists but is too rare for")
    print("statistical significance.  See docs/results.md for the real-tape verdict.")


if __name__ == "__main__":
    main()
