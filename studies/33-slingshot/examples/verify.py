"""Reproduce the real-data headline run (docs/results.md) — the current S&P 500, 2010–today.

    python examples/verify.py            # cache-only (offline); prints if the panel cache is present
    python examples/verify.py --fetch    # download the S&P 500 panel via quantlab.universe, then run

Runs the dollar-neutral cross-sectional reversal book on the cached S&P 500 returns panel and prints the
gross-vs-net Sharpe, turnover, break-even cost, cost sweep, lookback sweep, holding-period rescue and
sub-period decay that docs/results.md fingerprints.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from slingshot import costs, data, extension, strategy


def main(fetch: bool) -> None:
    r = data.fetch_panel(fetch=fetch)
    if r.empty:
        print("No cached panel. Re-run with --fetch (needs network) to download the S&P 500 panel.")
        return
    try:
        from quantlab import repro
        r = repro.as_of(r, "2026-06-10")
    except Exception:
        pass
    r = r.dropna(how="all")

    def line(nm, x):
        s = strategy.summary(x)
        print(f"  {nm:26} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  vol {s['vol_ann']*100:4.1f}%  "
              f"maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    print(f"\nS&P 500 panel, {r.index[0].date()} - {r.index[-1].date()} ({r.shape[1]} names, {len(r)} days)\n")
    line("reversal (gross)", strategy.book_returns(r, cost_bps=0.0))
    line("reversal (net @5bp)", strategy.book_returns(r, cost_bps=5.0))
    line("reversal (net @10bp)", strategy.book_returns(r, cost_bps=10.0))
    print(f"\nTurnover/day: {strategy.turnover(r):.2f}   Break-even cost: {costs.breakeven_cost_bps(r):.2f} bp")
    print("\nCost sweep (bp -> Sharpe/CAGR):\n" + costs.cost_sweep(r).round(3).to_string())
    print("\nLookback sweep:\n" + extension.lookback_sweep(r).round(3).to_string())
    print("\nHolding-period rescue:\n" + extension.holding_period_sweep(r).round(3).to_string())
    print("\nSub-period Sharpe (decay):\n" + extension.subperiod_sharpe(r, n_splits=3).round(3).to_string())

    try:
        from quantlab import repro
        print(f"\nas-of {r.index[-1].date()} · inputs fingerprint {repro.fingerprint(r)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
