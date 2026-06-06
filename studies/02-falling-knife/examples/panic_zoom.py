"""Panic zoom — the only place the dip edge looked real: deep selloffs (-5%/-7%).

The threshold sweep showed -3% is folklore but the bounce grows and turns
significant as the drop deepens. This script zooms in on that tail and asks the
three questions that decide whether it's *tradeable*, not just *true*:

  1. SHAPE  — do deeper drops bounce harder? (overlaid event studies + benchmark)
  2. CAPACITY — how often do these events even happen, and how clustered are they?
     (a real edge you can touch twice a decade is barely an edge.)
  3. STABILITY — does a fixed deep-dip rule survive out-of-sample, and across the
     2008/2020 regimes that dominate the sample?

Run:  python examples/panic_zoom.py     (uses cached data; network on first call)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falling_knife import data, triggers, exits, eventstudy, benchmark, backtest, robustness, plots

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 150)

THRESHOLDS = [-0.03, -0.05, -0.07]


def capacity(events: pd.Series, ohlc: pd.DataFrame) -> dict:
    """Frequency and clustering of a fresh-event series."""
    dates = events.index[events.to_numpy()]
    years = ohlc.shape[0] / 252.0
    n = len(dates)
    by_year = pd.Series(dates).dt.year.value_counts().sort_values(ascending=False)
    top2_share = by_year.head(2).sum() / n if n else np.nan
    return {
        "n_events": n,
        "per_decade": 10 * n / years if years else np.nan,
        "avg_years_between": years / n if n else np.nan,
        "top2_years": list(by_year.head(2).index),
        "top2_share": top2_share,
    }


def fixed_rule_oos(ohlc, raw_signal, rule, costs, frac=0.6):
    """In-sample vs out-of-sample for ONE fixed rule (no selection)."""
    is_o, oos_o = robustness.split_sample(ohlc, frac)
    out = {}
    for tag, seg in (("IS", is_o), ("OOS", oos_o)):
        sig = triggers.first_crossings(raw_signal.reindex(seg.index), cooldown=max(20, rule.max_hold))
        r = backtest.run(seg, sig, rule, costs)
        out[tag] = (r.stats["n_trades"], r.stats["cagr"], r.stats["sharpe"], r.stats["max_drawdown"])
    return out


def study(label, ticker):
    ohlc = data.fetch(ticker, mode="split_only")
    ret = data.daily_returns(ohlc)
    print(f"\n{'#'*74}\n#  {label} ({ticker})   {ohlc.index[0].date()}..{ohlc.index[-1].date()}\n{'#'*74}")

    # 1) SHAPE — event study + benchmark at each depth.
    es_by_thr = {}
    print("\n[Shape] bounce by drop depth (excess vs random day):")
    print(f"  {'thr':>5} {'n':>4} {'+1d':>9} {'+5d':>9} {'+10d':>9}  {'p(+1d)':>7} {'p(+5d)':>7}  boot+5d 95%CI")
    for thr in THRESHOLDS:
        ev = triggers.first_crossings(triggers.close_to_close(ret, thr), cooldown=20)
        es_by_thr[f"{thr:.0%}"] = eventstudy.event_study(ohlc, ev, horizon=20, pre=3)
        b = benchmark.conditional_vs_unconditional(ohlc, ev, horizons=(1, 5, 10), n_iter=2000)
        bb = robustness.block_bootstrap_excess(ohlc, ev, horizon=5)
        if 5 in b.index:
            print(f"  {thr:>5.0%} {int(b.loc[1,'n_events']):>4} "
                  f"{b.loc[1,'excess']:>+9.3%} {b.loc[5,'excess']:>+9.3%} {b.loc[10,'excess']:>+9.3%}  "
                  f"{b.loc[1,'p_greater']:>7.3f} {b.loc[5,'p_greater']:>7.3f}  "
                  f"[{bb['ci_low']:+.2%}, {bb['ci_high']:+.2%}]")

    # 2) CAPACITY — how tradeable is the tail?
    print("\n[Capacity] can you actually deploy this?")
    for thr in THRESHOLDS:
        ev = triggers.first_crossings(triggers.close_to_close(ret, thr), cooldown=20)
        c = capacity(ev, ohlc)
        print(f"  {thr:>5.0%}: {c['n_events']:>3} events | {c['per_decade']:>4.1f}/decade | "
              f"~1 every {c['avg_years_between']:>4.1f}y | top-2 yrs {c['top2_years']} "
              f"= {c['top2_share']:.0%} of all events")

    # 3) STABILITY — fixed -5% buy/hold-5d: regimes + IS/OOS.
    raw5 = triggers.close_to_close(ret, -0.05)
    ev5 = triggers.first_crossings(raw5, cooldown=20)
    rule = exits.ExitRule(max_hold=5)
    res5 = backtest.run(ohlc, ev5, rule, backtest.CostModel())
    print(f"\n[Stability] fixed rule: buy -5% close, hold<=5d, retail costs")
    print(f"  full sample: {res5.stats['n_trades']} trades, CAGR {res5.stats['cagr']:.2%}, "
          f"Sharpe {res5.stats['sharpe']:.2f}, maxDD {res5.stats['max_drawdown']:.1%}")
    oos = fixed_rule_oos(ohlc, raw5, rule, backtest.CostModel())
    print(f"  IS  (first 60%): {oos['IS'][0]:>2} trades, CAGR {oos['IS'][1]:+.2%}, Sharpe {oos['IS'][2]:.2f}")
    print(f"  OOS (last  40%): {oos['OOS'][0]:>2} trades, CAGR {oos['OOS'][1]:+.2%}, Sharpe {oos['OOS'][2]:.2f}")
    print("  regime split:")
    print(robustness.regime_split(res5.daily).to_string(float_format=lambda v: f"{v:.4f}"))

    # Figure: overlaid event studies.
    plots.plot_event_overlay(
        es_by_thr, title=f"{label}: deeper drops bounce harder (mean path)",
        path=os.path.join(OUT_DIR, f"out_panic_{ticker.replace('^','').lower()}.png"))
    return es_by_thr


def main():
    print("FALLING-KNIFE — panic zoom (deep selloffs, spot indices for depth)")
    for label, (spot, _etf) in data.INDEX_PAIRS.items():
        study(label, spot)
    print(f"\n{'='*74}\n  TAKEAWAY\n{'='*74}")
    print("  The bounce is real and significant in deep panic (-5%/-7%) and grows\n"
          "  with depth on BOTH indices — but those events are rare and dominated by\n"
          "  2008/2020. It's a true effect with tiny capacity: a disciplined deep-dip\n"
          "  entry, not a standalone alpha engine. Size it for a handful of trades a\n"
          "  decade, into the teeth of a crash, or don't trade it at all.")


if __name__ == "__main__":
    main()
