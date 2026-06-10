"""Real-data run — is PEAD real and tradable on the tape, and how does the drift decay?

    python examples/verify.py --fetch     # would pull earnings dates + surprises + prices, cache, run
    python examples/verify.py             # offline, cache-only

**Pending-fetch note.** A credible PEAD backtest needs *years* of reported-earnings dates and surprises
per name. No free source supplies that in this sandbox: yfinance exposes only ~6-8 reported quarters
(and no reliable long surprise history is available free — the same wall the desk hit for options
open-interest). So this study has **no pre-populated cache and no wired long-history source**: on the
cache-miss path it prints a skip and the offline synthetic control (run_synthetic_demo.py) is the
validated proof. The real measurement is **pre-registered and pending** — exactly Study 27 (Steamroller)'s
pattern, before its one FRED fetch. When an earnings-history feed is wired, this script downloads it,
runs the same book, and overwrites docs/results.md with the fingerprinted, as-of'd numbers.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()

    bundle = data.fetch_earnings_panel(fetch=args.fetch)
    if not bundle:
        print("[skip] no earnings-history source wired in this environment — see docs/results.md")
        print("       A real PEAD run needs years of reported-earnings dates + surprises per name;")
        print("       no free feed supplies that here (yfinance gives only ~6-8 quarters). The real")
        print("       measurement is pre-registered and PENDING; the offline synthetic core")
        print("       (run_synthetic_demo.py) is the validated proof meanwhile.")
        return

    panel, events = bundle["panel"], bundle["events"]
    try:
        from quantlab import repro
        panel = repro.as_of(panel, "2026-06-10")
    except Exception:
        pass

    def line(nm, x):
        s = strategy.summary(x)
        print(f"  {nm:26} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  vol {s['vol_ann']*100:4.1f}%  "
              f"maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    print(f"\nPEAD panel, {panel.index[0].date()} - {panel.index[-1].date()} "
          f"({panel.shape[1]} names, {len(events)} events)\n")
    line("PEAD (gross)", strategy.book_returns(panel, events, cost_bps=0.0))
    line("PEAD (net @5bp)", strategy.book_returns(panel, events, cost_bps=5.0))
    line("PEAD (net @10bp)", strategy.book_returns(panel, events, cost_bps=10.0))
    print(f"\nTurnover/day: {strategy.turnover(panel, events):.3f}   "
          f"Break-even cost: {costs.breakeven_cost_bps(panel, events):.1f} bp")
    print("\nCost sweep (bp -> Sharpe/CAGR):\n" + costs.cost_sweep(panel, events).round(3).to_string())
    print("\nHolding-period sweep:\n" + extension.holding_period_sweep(panel, events).round(3).to_string())
    print("\nDrift-decay curve (signed CAR by day since event):")
    print(extension.drift_decay_curve(panel, events, window=70).round(4).to_string())

    try:
        from quantlab import repro
        print(f"\nas-of {panel.index[-1].date()} · inputs fingerprint {repro.fingerprint(panel)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
