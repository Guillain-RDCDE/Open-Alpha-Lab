"""Offline demo — the whole Social-Oracle pipeline on synthetic data, no network.

Builds a toy universe of micro-caps with a *baked-in pump-and-fade* (mentions land
after a run-up, pop on the day, then bleed back), then runs the full gauntlet:
feed -> events (+ coverage) -> event study -> random-day null -> the momentum
control -> fade curve -> clustering bootstrap -> name jackknife -> a cost-charged
backtest -> capacity.

The pump-and-fade is real (we put it there), so this is also a sanity check that the
machinery *detects* the signature it's meant to. Run on a real feed, the same code
asks whether the signature is actually there.

    python examples/run_synthetic_demo.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from social_oracle import backtest, benchmark, data, eventstudy, mentions, robustness

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main():
    panel, feed = data.synthetic_panel(seed=0)
    print(f"synthetic universe: {len(panel)} names, {len(feed)} mentions\n")

    events, cov = mentions.to_events(feed, panel)
    print("[coverage] mentions -> events:", cov)

    es = eventstudy.event_study(panel, events, horizon=21, pre=5)
    print(f"\n[event study] {es['n_events']} events; abnormal CAR at rel days -5/0/1/5/21:")
    print(es["summary"].loc[[-5, 0, 1, 5, 21]].round(4))

    print("\n[random-day null] mention vs a random (name, day)")
    print(benchmark.conditional_vs_unconditional(panel, events, n_iter=500).round(4))

    print("\n[momentum control] mention vs a name that was already hot")
    hot = mentions.hot_streak_events(panel)
    print(benchmark.excess_vs_alternative(panel, events, hot, n_iter=500).round(4))

    print("\n[fade curve] mean abnormal CAR by horizon (pop then fade?)")
    print(robustness.fade_curve(panel, events).round(4))

    print("\n[block bootstrap] h=5 (clustering-aware)")
    print({k: round(v, 4) for k, v in
           robustness.block_bootstrap_excess(panel, events, horizon=5, n_iter=500).items()})

    print("\n[name jackknife] drop the most-mentioned names, h=5")
    print(robustness.name_jackknife(panel, events, horizon=5).round(4))

    print("\n[backtest] buy next open, hold 10d, micro-cap costs")
    res = backtest.run(panel, events, hold_days=10)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.stats.items()})

    print("\n[cost sweep] mean net trade vs half-spread (bps)")
    print(backtest.cost_sweep(panel, events).round(4))

    # No positive edge exists in this synthetic fade — illustrate capacity with a
    # nominal 50bp edge so the square-root-impact maths is visible.
    print("\n[capacity] $ size where impact eats a nominal 50bp edge:")
    print({k: (round(v, 1) if isinstance(v, float) else v)
           for k, v in backtest.capacity(panel, events, edge_bps=50.0).items()})

    print("\nOK — full pipeline ran offline.")


if __name__ == "__main__":
    main()
