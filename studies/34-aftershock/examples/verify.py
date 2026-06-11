"""Real-data run — is PEAD real and tradable on the tape, and how does the drift decay?

    python examples/verify.py            # offline, reads the cache, prints all headline numbers

Runs the dollar-neutral long-positive / short-negative-SUE PEAD book on the **real tape**: cached EDGAR
quarterly EPS (announcement = earliest filing per quarter) turned into a seasonal-random-walk SUE
(``surprise_q = eps_q − eps_{q−4}``, standardised by the stock's own trailing dispersion), traded against
the cached S&P 500 price panel's split/dividend-adjusted-Close returns, entry lagged one day after the
filing. Fully offline from the cache.

**Survivorship caveat.** The price panel is *today's* S&P 500 membership, so it omits delisted/dropped
names — the magnitudes here are biased upward by the missing losers; the qualitative verdict (small, real
drift; break-even cost inside realistic costs) is robust.
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

from aftershock import costs, data, extension, strategy

COST_BPS = 5.0
HOLDING_DAYS = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="(no-op offline; the cache already exists)")
    ap.add_argument("--rebuild", action="store_true", help="re-assemble the SUE/returns cache from raw EDGAR + panel")
    args = ap.parse_args()

    bundle = data.fetch_earnings_panel(fetch=args.fetch, rebuild=args.rebuild)
    if not bundle:
        print("[skip] PEAD cache not found — expected _cache/edgar_eps.parquet + _cache/panel_503_*.parquet")
        return

    panel, events = bundle["panel"], bundle["events"]
    # Pin the sample to the panel's last session so reruns don't drift as the calendar moves.
    as_of = str(panel.index[-1].date())
    try:
        from quantlab import repro
        panel = repro.as_of(panel, as_of)
    except Exception:
        pass
    events = events[events["date"] <= panel.index[-1]].reset_index(drop=True)

    def line(nm, x):
        s = strategy.summary(x)
        print(f"  {nm:26} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  vol {s['vol_ann']*100:4.1f}%  "
              f"maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    print(f"\nPEAD panel, {panel.index[0].date()} - {panel.index[-1].date()} "
          f"({events['ticker'].nunique()} names, {len(events)} events, hold {HOLDING_DAYS}d)\n")
    g = strategy.book_returns(panel, events, holding_days=HOLDING_DAYS, cost_bps=0.0)
    line("PEAD (gross)", g)
    line("PEAD (net @5bp)", strategy.book_returns(panel, events, holding_days=HOLDING_DAYS, cost_bps=5.0))
    line("PEAD (net @10bp)", strategy.book_returns(panel, events, holding_days=HOLDING_DAYS, cost_bps=10.0))

    nw = strategy.newey_west_t(g, lags=20)
    plain = strategy.newey_west_t(g, lags=0)
    print(f"\nGross mean t-stat: Newey-West(L=20) {nw:+.2f}   plain {plain:+.2f}")
    print(f"Turnover/day: {strategy.turnover(panel, events, holding_days=HOLDING_DAYS):.3f}   "
          f"Break-even cost: {costs.breakeven_cost_bps(panel, events, holding_days=HOLDING_DAYS):.1f} bp")
    print("\nCost sweep (bp -> Sharpe/CAGR):\n"
          + costs.cost_sweep(panel, events, holding_days=HOLDING_DAYS).round(3).to_string())
    print("\nHolding-period sweep:\n" + extension.holding_period_sweep(panel, events).round(3).to_string())

    car = extension.drift_decay_curve(panel, events, window=70)
    print("\nDrift-decay curve (signed CAR by day since announcement):")
    for d in (0, 5, 10, 20, 30, 40, 50, 60, 69):
        print(f"  day {d:2d}: {car.iloc[d]:+.5f}")

    try:
        from quantlab import repro
        print(f"\nas-of {as_of} · inputs fingerprint {repro.fingerprint(panel)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
