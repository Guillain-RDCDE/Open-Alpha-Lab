"""Real-data run — is the commodity carry / roll-yield premium real, and how bad is the crash tail?

    python examples/verify.py            # cache-only (offline); prints the pending-fetch skip
    python examples/verify.py --fetch    # reserved for a future term-structure source

**Network / data note.** Computing roll yield needs the **term structure** — at least the front and
first-deferred contract for each commodity, every week (the slope of the curve). The desk's cache holds
only the **front-month continuous** returns (``commodity_futures_weekly.parquet``, 12 commodities), and no
free source in this environment reliably serves the deferred contracts. So this study's real run is
**PENDING a term-structure fetch**, exactly as Study 27 (Steamroller) was pending its FRED download. On a
cache miss this prints the skip and the offline synthetic core (run_synthetic_demo.py) stands as the
validated proof. Pinned with ``quantlab.repro.as_of`` and fingerprinted once a curve is available.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from contango import costs, data, extension, strategy


def main(fetch: bool) -> None:
    curve = data.fetch_curve(fetch=fetch)
    if not curve:
        print("[skip] term structure not available in this environment — see docs/results.md.")
        print("       Roll yield needs the FRONT + first-DEFERRED contract per commodity each week;")
        print("       the cache holds only front-month continuous returns, and no free source here")
        print("       serves the deferred leg. The real run is PENDING a term-structure fetch.")
        basket = data.load_front_month_basket()
        if not basket.empty:
            print(f"\n       (front-month basket present for cross-check: {basket.shape[1]} commodities, "
                  f"{basket.index.min().date()}→{basket.index.max().date()} — front-month only, NO curve.)")
        print("\n       The offline synthetic core (examples/run_synthetic_demo.py) is the validated proof.")
        return

    # --- the real run, once a curve cache is populated (front_*/def_* columns) ---
    front, deferred = curve["front"], curve["deferred"]
    # roll yield ≈ log(front / deferred) per week: positive = backwardation
    import numpy as np
    roll_yield = np.log(front / deferred)
    returns = front.pct_change()

    try:
        from quantlab import repro
        returns = repro.as_of(returns, "2026-06-10")
        roll_yield = repro.as_of(roll_yield, "2026-06-10")
    except Exception:
        pass

    def line(nm, x):
        s = strategy.summary(x)
        print(f"  {nm:24} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  "
              f"maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    pb = strategy.carry_premium_by_bucket(returns, roll_yield)
    print(f"\nCommodity term structure, {returns.index[0].date()} → {returns.index[-1].date()} "
          f"({returns.shape[1]} commodities)\n")
    line("carry (gross)", strategy.book_returns(returns, roll_yield, cost_bps=0.0))
    line("carry (net @5bp)", strategy.book_returns(returns, roll_yield, cost_bps=5.0))
    print(f"\nHigh-minus-low roll-yield bucket: {pb['hml_ann_pct']:+.1f}%/yr   "
          f"turnover/wk {strategy.turnover(roll_yield):.3f}   break-even {costs.breakeven_cost_bps(returns, roll_yield):.0f} bp")
    c = extension.combine(returns, roll_yield, cost_bps=5.0)
    print(f"\ncarry+momentum blend: carry {c['carry_sharpe']:.2f}  momentum {c['momentum_sharpe']:.2f}  "
          f"blend {c['blend_sharpe']:.2f}  (corr {c['correlation']:+.2f})")
    try:
        from quantlab import repro
        print(f"\nas-of {returns.index[-1].date()} · fingerprint {repro.fingerprint(returns.round(6))}")
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true")
    main(ap.parse_args().fetch)
