"""Offline core — the machinery proof, no network, fixed seeds.

Two tapes, one harness (bench rule: a pipeline that can't bank a planted signal proves
nothing by finding nothing):

  * **Planted** — every armed signal adds a known +8 bp to the next day's mean. The
    harness must show a monotone lift table, an HAC *t* ≥ 2 armed stream, and a net
    book that banks it through the full CFD overlay (sizing, stop, costs).
  * **Null** — a seeded random walk with the same bar geometry. Everything must read
    as noise.

Synthetic numbers are machinery evidence only — they never support the Signal stamp
(that's the real tape's job, examples/verify.py).
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from ambush import strategy, synth  # noqa: E402
from quantlab.analytics import mean_tstat_hac  # noqa: E402


def run(plant: float, seed: int, label: str) -> None:
    spy, vix = synth.synthetic_tape(plant_bps_per_signal=plant, seed=seed)
    rf = synth.flat_rf(spy.index)
    lift = strategy.lift_table(spy, vix)
    t = mean_tstat_hac(strategy.armed_stream(spy, vix, k=3))
    led = strategy.book(spy, vix, rf, k=3)
    s = strategy.summary(led["net_excess"], led)
    print(f"--- {label} (plant {plant:+.0f} bp/signal, seed {seed}) ---")
    print(lift.round(2).to_string())
    print(
        f"armed K>=3: {t['mean_bps']:+.1f} bp/day, HAC t = {t['tstat']:+.2f} (n={t['n']}) | "
        f"net book Sharpe {s['sharpe']:+.2f}, {s['trades_per_year']:.0f} trades/yr\n"
    )


if __name__ == "__main__":
    run(plant=8.0, seed=0, label="PLANTED — the harness must light up")
    run(plant=0.0, seed=1, label="NULL — the harness must stay dark")
