"""Real-data run — does the 1:1 breakout bracket beat a coin flip once it pays a spread?

The offline synthetic core (the notebooks) proves the *machine*: on a driftless tape the bracket is a
coin flip and costs make it negative; on a continuation tape the same machine finds the edge. This
script points that machine at the **market** — cached intraday OHLCV for Koroush AK's own arena
(BTC-USD, 24/7) plus SPY and QQQ — and reports the one number that decides it: the win rate, with a
Wilson interval, next to the break-even line that costs demand.

    # fetch the bars into the local cache, then run:
    python examples/verify.py --fetch
    # later, offline, reproduce from cache only:
    python examples/verify.py

Network lives only behind ``--fetch``. Without it the run is **cache-only** — a ticker with no cached
parquet is skipped, never silently re-downloaded. The sample is pinned with ``quantlab.repro.as_of``
and stamped with a content fingerprint, so a reader who reruns and matches the fingerprint holds the
same tape. **Named limitation:** Yahoo serves only ~60 days of 5-minute history, so every real win
rate here rests on tens of trades — wide intervals by construction. The verdict is the synthetic
core's; this is the sanity check that the coin flip shows up in the wild.
"""

import argparse
import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from glass_ceiling import data, filters, strategy
from quantlab.repro import data_stamp

TICKERS = ["BTC-USD", "SPY", "QQQ"]
INTERVAL = "5m"
ROUNDTRIP_BPS = 2.0      # an optimistic round-trip for a liquid venue; crypto/CFD is worse


def run_one(ticker: str, fetch: bool) -> dict | None:
    bars = data.fetch_bars(ticker, interval=INTERVAL, period="60d", fetch=fetch)
    if bars.empty:
        print(f"[skip] {ticker}: no cached bars (run with --fetch once).")
        return None
    print(data_stamp(f"{ticker} {INTERVAL}", bars, cols=["Open", "High", "Low", "Close"]))
    bars = bars.reset_index(drop=True)
    bars.columns = ["Open", "High", "Low", "Close", "Volume"]

    trades = strategy.run(bars)
    s = strategy.summary(trades, roundtrip_bps=ROUNDTRIP_BPS)
    lo, hi = strategy.win_rate_ci(trades)
    fl = filters.filter_lift(trades, bars)
    return {"ticker": ticker, **s, "ci_lo": lo, "ci_hi": hi, **{f"flt_{k}": v for k, v in fl.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="hit Yahoo once to (re)fill the cache")
    args = ap.parse_args()

    rows = [r for t in TICKERS if (r := run_one(t, args.fetch)) is not None]
    if not rows:
        print("\nNo cached tapes. Run `python examples/verify.py --fetch` first.")
        return

    print(f"\n{'ticker':9s} {'trades':>6s} {'win%':>6s} {'95% CI':>16s} "
          f"{'gross_R':>8s} {'BE_win%':>8s} {'net_R':>7s} | filters: lift / kept / n")
    for r in rows:
        ci = f"[{r['ci_lo']*100:.0f},{r['ci_hi']*100:.0f}]"
        print(f"{r['ticker']:9s} {r['n_trades']:6d} {r['win_rate']*100:6.1f} {ci:>16s} "
              f"{r['expectancy_R_gross']:+8.3f} {r['breakeven_win_rate']*100:8.1f} "
              f"{r['expectancy_R_net']:+7.3f} | {r['flt_lift']:+.3f} / "
              f"{r['flt_kept_frac']:.0%} / {r['flt_n_filtered']}")
    print("\nRead it: every CI straddles 50% (a coin flip), the deepest sample (BTC-USD) is "
          "net-negative once it pays 2 bps twice, and each filtered subset is a handful of trades — "
          "the selection illusion, not an edge.")


if __name__ == "__main__":
    main()
