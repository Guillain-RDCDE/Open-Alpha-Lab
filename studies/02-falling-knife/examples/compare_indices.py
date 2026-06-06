"""Does the (non-)edge generalise beyond the Nasdaq? Nasdaq-100 vs S&P 500.

If "buy the -3% dip" were a genuine market mechanism it should show up on the
S&P 500 too, not only on the tech-heavy Nasdaq. This script runs the decisive
benchmark and an honest in-sample/out-of-sample test on both indices, spot and
ETF, side by side. Run:

    python examples/compare_indices.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falling_knife import data, triggers, exits, benchmark, robustness

pd.set_option("display.width", 140)


def summarise(label, ticker):
    ohlc = data.fetch(ticker, mode="split_only")
    ret = data.daily_returns(ohlc)
    events = triggers.first_crossings(triggers.close_to_close(ret), cooldown=20)

    bench = benchmark.conditional_vs_unconditional(ohlc, events, horizons=(1, 5, 10, 20))
    row = bench.loc[5] if 5 in bench.index else None

    # Honest IS/OOS over the whole family (selection on first 60%, test on tail).
    sigs = {n: f(ret) for n, f in triggers.TRIGGERS.items()}
    oos = robustness.oos_best_cell(ohlc, sigs, exits.default_grid())

    return {
        "index": label,
        "ticker": ticker,
        "span": f"{ohlc.index[0].date()}..{ohlc.index[-1].date()}",
        "n_events": int(row["n_events"]) if row is not None else 0,
        "excess_5d": float(row["excess"]) if row is not None else float("nan"),
        "p_5d": float(row["p_greater"]) if row is not None else float("nan"),
        "is_sharpe": oos.get("is_sharpe", float("nan")),
        "oos_sharpe": oos.get("oos_sharpe", float("nan")),
        "best_cell": f"{oos.get('trigger','?')} | {oos.get('exit','?')}",
        "oos_verdict": oos.get("verdict", "?"),
    }


def main():
    print("FALLING-KNIFE — Nasdaq-100 vs S&P 500 (generality + IS/OOS)")
    rows = []
    for label, (spot, etf) in data.INDEX_PAIRS.items():
        rows.append(summarise(f"{label} spot", spot))
        rows.append(summarise(f"{label} ETF", etf))

    df = pd.DataFrame(rows)
    print("\n[Decisive benchmark | T1 close-to-close, +5d excess vs random day]")
    print(df[["index", "ticker", "span", "n_events", "excess_5d", "p_5d"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[In-sample vs out-of-sample | best family cell picked on first 60%]")
    print(df[["index", "ticker", "best_cell", "is_sharpe", "oos_sharpe", "oos_verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nIf the dip 'edge' were real it would (a) be significant (p<0.05), "
          "(b) appear on BOTH indices, and (c) survive out-of-sample. Watch how "
          "many of those three it actually clears.")


if __name__ == "__main__":
    main()
