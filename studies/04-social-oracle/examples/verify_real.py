"""Real-data verification — point it at a mentions CSV and reproduce the tables.

Unlike Studies 01–03, the signal here isn't a clean Yahoo series; it's a *feed* of
influencer mentions you bring yourself (exactly the CSV the open-source "Serenity
skill" scrapers emit: a timestamp column and a ``$SYMBOL`` column, optional score).
Prices for the mentioned names are pulled from Yahoo! (cached after the first run);
a broad benchmark (default SPY) defines the abnormal return.

    python examples/verify_real.py path/to/mentions.csv
    python examples/verify_real.py path/to/mentions.csv IWM   # small-cap benchmark

Drops are reported, not hidden (no silent caps): how many mentions had no tradeable
price, how many sat too close to the sample edge. The verdict tables are the
random-day null, the momentum control, the fade curve, the clustering bootstrap,
the name jackknife, and the cost-charged backtest — the numbers behind each stamp.

No CSV? This script can't fabricate a feed. Run ``examples/run_synthetic_demo.py``
for the offline pipeline, then bring real mentions here.
"""

import os
import sys

import pandas as pd

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY_DIR, "..", "..")))  # repo root, for quantlab

from social_oracle import backtest, benchmark, data, mentions, robustness
from quantlab.repro import DEFAULT_AS_OF, data_stamp

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    feed_path = sys.argv[1]
    bench = sys.argv[2] if len(sys.argv) > 2 else data.DEFAULT_MARKET

    feed = data.load_feed(feed_path)
    names = sorted(feed["ticker"].unique())
    print(f"feed: {len(feed):,} mentions on {len(names)} names  "
          f"{feed['timestamp'].min().date()} -> {feed['timestamp'].max().date()}")

    # Fetch the benchmark and each mentioned name (cached after first run).
    market_ret = data.market_frame(bench, end=DEFAULT_AS_OF)
    raw = {}
    for t in names:
        try:
            raw[t] = data.fetch(t, end=DEFAULT_AS_OF)
        except Exception as exc:  # delisted / bad symbol — a survivorship signal
            print(f"  · no price for {t}: {exc}".split(".")[0])
    panel = data.build_panel(raw, market_ret=market_ret)
    print(data_stamp(bench, market_ret.to_frame(), cols=["r_mkt"]))

    events, cov = mentions.to_events(feed, panel)
    print("\n[coverage] mentions -> events:", cov)
    if events.empty:
        print("No usable events after the join. Check symbols and dates.")
        return

    print("\n[random-day null] mention vs a random (name, day)")
    null = benchmark.conditional_vs_unconditional(panel, events)
    print(null.round(4))

    print("\n[momentum control] mention vs a name that was already hot")
    hot = mentions.hot_streak_events(panel)
    print(benchmark.excess_vs_alternative(panel, events, hot).round(4))

    print("\n[fade curve] mean abnormal CAR by horizon")
    print(robustness.fade_curve(panel, events).round(4))

    print("\n[block bootstrap h=5] clustering-aware excess")
    print({k: round(v, 4) for k, v in
           robustness.block_bootstrap_excess(panel, events, horizon=5).items()})

    print("\n[name jackknife h=5] is one name carrying it?")
    print(robustness.name_jackknife(panel, events, horizon=5).round(4))

    print("\n[backtest] buy next open, hold 10d, micro-cap costs")
    res = backtest.run(panel, events, hold_days=10)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.stats.items()})
    print("\n[cost sweep] mean net trade vs half-spread (bps)")
    print(backtest.cost_sweep(panel, events).round(4))


if __name__ == "__main__":
    main()
