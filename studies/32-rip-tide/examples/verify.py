"""Reproduce the real-data headline run (docs/results.md) — diversified futures, 2000–today.

    python examples/verify.py            # cache-only (offline); prints if cache present
    python examples/verify.py --fetch    # download the futures basket from Yahoo, then run

Runs the equal-risk short-horizon contrarian book on the same 18-futures basket as Study 31, and prints
the gross-vs-net Sharpe, the cost sweep, the break-even cost, turnover, the long-only benchmark, the
holding-period rescue and the sub-period decay that docs/results.md fingerprints.
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

from rip_tide import costs, data, extension, strategy


def main(fetch: bool) -> None:
    r = data.fetch_futures(fetch=fetch)
    if r.empty:
        print("No cached real data. Re-run with --fetch (needs network) to download the futures basket.")
        return
    try:
        from quantlab import repro
        r = repro.as_of(r, "2026-06-10")
    except Exception:
        pass

    def line(nm, x):
        s = strategy.summary(x)
        print(f"  {nm:28} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  vol {s['vol_ann']*100:3.0f}%  "
              f"maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    print(f"\nDiversified futures, {r.index[0].date()} - {r.index[-1].date()} ({r.shape[1]} markets, {len(r)} days)\n")
    gross_book = strategy.book_returns(r, cost_bps=0.0)
    line("contrarian (gross)", gross_book)
    line("contrarian (net @2bp)", strategy.book_returns(r, cost_bps=2.0))
    line("long-only basket", costs.long_only_basket(r))
    try:
        from quantlab.analytics import mean_tstat_hac
        ht = mean_tstat_hac(gross_book.dropna())
        print(f"\nGross book mean: {ht['mean_bps']:+.2f} bp/day, Newey-West t = {ht['tstat']:+.2f} "
              f"({ht['lags']} lags, n={ht['n']})")
    except Exception:
        pass
    print(f"\nTurnover/day: {strategy.turnover(r):.2f}   Break-even cost: {costs.breakeven_cost_bps(r):.2f} bp")
    print("\nCost sweep (bp -> NET Sharpe/CAGR):\n" + costs.cost_sweep(r).round(3).to_string())
    print("\nHolding-period rescue (gross and net @2bp, both labelled):\n"
          + extension.holding_period_sweep(r).round(3).to_string())
    print("\nHorizon sweep (gross and net @2bp, both labelled):\n" + extension.horizon_sweep(r).round(3).to_string())
    print("\nSub-period Sharpe (gross and net @2bp, both labelled):\n"
          + extension.subperiod_sharpe(r, n_splits=3).round(3).to_string())

    try:
        from quantlab import repro
        print(f"\nas-of {r.index[-1].date()} · inputs fingerprint {repro.fingerprint(r)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
