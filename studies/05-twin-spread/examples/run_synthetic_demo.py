"""Offline demo — the whole pairs machine on a synthetic universe with true twins.

No network, fixed seed. Builds a toy market with baked-in cointegrated pairs hidden
among noise names, then runs: formation (does the selector find the twins?) → trade →
decay-by-year → the bid-ask-bounce wait rule → market-neutrality. This is the worked
*method*; the live verdict is `examples/verify_real.py` on the cached universe.

    python examples/run_synthetic_demo.py
"""

import os
import sys

import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from pairs_trading import backtest, data, pairs, robustness

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main():
    panel, frames, true_pairs = data.synthetic_universe(seed=0)
    print(f"universe: {panel.shape[1]} names, {panel.shape[0]} sessions, "
          f"{len(true_pairs)} true twins baked in")

    # 1) formation on the first year — should surface the twins
    formation = panel.iloc[:252]
    selected = pairs.select_pairs(formation, top_n=len(true_pairs))
    recall = robustness.selection_recall(selected, true_pairs)
    print(f"\n[formation] top-{len(true_pairs)} by SSD recovers "
          f"{recall:.0%} of the true twins")
    for p in selected[:5]:
        print(f"   {p.a:>7} ~ {p.b:<7}  ssd={p.ssd:.3f}  sigma={p.sigma:.4f}")

    # 2) full backtest, honest 1-day execution lag
    res = backtest.run(panel, top_n=8, form_len=252, trade_len=126, wait=1)
    print("\n[backtest wait=1]")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.stats.items()})

    # 3) decay by year
    print("\n[decay by year]")
    print(robustness.decay_by_year(panel, top_n=8).round(4))

    # 4) the bid-ask-bounce wait rule (wait=1 is the GGR headline; more lag = past the bounce)
    print("\n[wait rule — does the edge fade as execution lag grows?]")
    print(robustness.wait_rule_effect(panel, top_n=8).round(4))

    # 5) market neutrality
    mkt = data.market_return(panel)
    print("\n[market neutrality]")
    print({k: round(v, 4) for k, v in robustness.market_neutrality(res.daily, mkt).items()})


if __name__ == "__main__":
    main()
