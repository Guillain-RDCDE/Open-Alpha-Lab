"""Real-data run for the pre-earnings-runup study.

    python examples/verify.py --fetch          # first run: download yfinance data, cache locally
    python examples/verify.py                  # subsequent runs: fully offline from the cache

Runs the equal-weight pre-earnings book on a fixed large-cap universe: buy N trading days before
each scheduled earnings announcement, exit just before the announcement. Measures gross/net book
returns and the per-day pre-event drift curve. Prints the fingerprint and headline numbers.

Survivorship caveat: the universe is current large-cap membership, so delisted names are absent,
which biases measured runup magnitudes upward.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from pre_earnings_runup import data, strategy

COST_BPS = 2.0
PRE_DAYS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download fresh data via yfinance and cache")
    ap.add_argument("--rebuild", action="store_true", help="re-download even if cache exists")
    args = ap.parse_args()

    bundle = data.fetch_real_data(fetch=args.fetch, rebuild=args.rebuild)
    if not bundle:
        print("[skip] real cache not found — run: python examples/verify.py --fetch")
        return

    panel = bundle["panel"]
    events = bundle["events"]

    # Pin the sample to the panel's last session so reruns don't drift as the calendar moves.
    as_of = str(panel.index[-1].date())
    try:
        from quantlab import repro
        panel = repro.as_of(panel, as_of)
    except Exception:
        pass
    events = events[events["date"] <= panel.index[-1]].reset_index(drop=True)

    n_names = events["ticker"].nunique()
    n_events = len(events)

    print(f"\nPre-earnings-runup panel: {panel.index[0].date()} - {panel.index[-1].date()} "
          f"({n_names} names, {n_events} events, pre_days={PRE_DAYS})\n")

    def line(nm, x):
        s = strategy.summary(x)
        print(f"  {nm:28} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  vol {s['vol_ann']*100:4.1f}%  "
              f"maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    g = strategy.book_returns(panel, events, pre_days=PRE_DAYS, cost_bps=0.0)
    line("Pre-earnings book (gross)", g)
    line(f"Pre-earnings (net @{COST_BPS}bp)", strategy.book_returns(panel, events, pre_days=PRE_DAYS, cost_bps=COST_BPS))
    line("Pre-earnings (net @5bp)", strategy.book_returns(panel, events, pre_days=PRE_DAYS, cost_bps=5.0))

    mkt = strategy.market_return(panel)
    line("Equal-weight market", mkt)

    nw = strategy.newey_west_t(g, lags=5)
    plain = strategy.newey_west_t(g, lags=0)
    to = strategy.turnover(panel, events, pre_days=PRE_DAYS)
    be = strategy.breakeven_cost_bps(panel, events, pre_days=PRE_DAYS)

    print(f"\nGross mean t-stat: Newey-West(L=5) {nw:+.2f}   plain {plain:+.2f}")
    print(f"Turnover/day: {to:.4f}   Break-even cost: {be:.1f} bp")

    print("\nPre-day sweep (gross Sharpe / net Sharpe @2bp / turnover):")
    sweep = strategy.pre_day_sweep(panel, events, windows=[1, 2, 3, 5, 10], cost_bps=COST_BPS)
    print(sweep.round(3).to_string())

    print("\nMean pre-event return by day-before-announcement (day -5 = first, day -1 = last before news):")
    curve = strategy.mean_pre_event_return(panel, events, pre_days=PRE_DAYS)
    for off, val in curve.items():
        print(f"  day {off:+d}: {val:+.6f}  ({val*100:+.4f}%)")

    try:
        from quantlab import repro
        fp = repro.fingerprint(panel)
        print(f"\nas-of {as_of} · inputs fingerprint {fp}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
