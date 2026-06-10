"""Beat-7 worked complement — breadth is the lever (docs/extension.md).

    python examples/extension.py            # cache-only (offline)
    python examples/extension.py --fetch    # download the wider ETF universe, then run

The main study runs 18 futures and finds the standalone trend book FRAGILE (Sharpe 0.30). The
literature says trend's Sharpe is a *breadth* story. Here we widen to a ~27-market liquid-ETF universe
and sweep the Sharpe against the number of markets — testing whether more breadth lifts the standalone
edge past the benchmarks.
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

from trade_winds import costs, data, extension, strategy


def main(fetch: bool) -> None:
    u = data.fetch_etf_universe(fetch=fetch)
    if u.empty:
        print("No cached wider universe. Re-run with --fetch (needs network) to download the ETF basket.")
        return
    s = strategy.summary(strategy.book_returns(u, cost_bps=2.0))
    print(f"\nWider ETF universe, {u.index[0].date()} - {u.index[-1].date()} ({u.shape[1]} markets, {len(u)} days)\n")
    print(f"  full wider book: Sharpe {s['sharpe']:.2f}  CAGR {s['cagr']*100:.1f}%  "
          f"maxDD {s['max_drawdown']*100:.0f}%  skew {s['skew']:+.2f}")
    print("\nBreadth sweep — mean book Sharpe vs number of markets:")
    print(extension.breadth_sweep(u).round(3).to_string())
    eqcols = [c for c in ("SPY", "QQQ", "IWM", "EFA") if c in u.columns]
    print("\nCrisis alpha (wider):", {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in costs.crisis_alpha(u, eqcols).items()})
    try:
        from quantlab import repro
        print(f"\nas-of {u.index[-1].date()} · inputs fingerprint {repro.fingerprint(u)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
