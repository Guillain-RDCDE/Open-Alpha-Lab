"""Real-data run — the energy roll-yield / contango tape, measured from ETF pairs (no FRED, no EIA).

    python examples/verify.py            # cache-only (offline); reads _cache/energy_carry_etfs.parquet
    python examples/verify.py --fetch    # download USO/USL/UNG/UNL from yfinance, cache, then run

Roll yield needs the **term structure** (where on the curve you sit). We observe it without a paid futures
feed by contrasting, for each energy commodity, the **front-month** ETF against the **12-month-laddered** one
(WTI: USO vs USL; gas: UNG vs UNL). The laddered fund barely touches the front roll, so ``laddered − front``
is the realized roll cost of the front contract — positive in contango, negative in backwardation. The
famous USO bleed is this number. See ``docs/results.md`` and :mod:`contango.energy`.

The cross-sectional bucket machinery (long the most-backwardated, short the most-contangoed) is proved on the
offline synthetic panel — ``examples/run_synthetic_demo.py``. With only two liquid energy curves the real
tape is timed per curve instead.
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

from contango import energy


def main(fetch: bool) -> None:
    if fetch:
        energy.fetch_energy_pairs(fetch=True)      # refresh the on-disk cache from yfinance
    prices = energy.load_pairs()                   # always as-of pinned → stable, reproducible numbers

    if prices.empty:
        print("[skip] energy ETF cache not found — run `python examples/verify.py --fetch` once (network)")
        print("       to download USO/USL/UNG/UNL into _cache/energy_carry_etfs.parquet.")
        print("       The offline synthetic machinery proof is `python examples/run_synthetic_demo.py`.")
        return

    end = prices.index.max().date()
    print(f"Energy roll-yield tape — weekly ETF pairs, through {end}\n")

    # --- (A) the contango bleed: the realized roll cost of the front-month contract
    bt = energy.bleed_table(prices)
    print("(A) The contango bleed — front-month vs 12-month-laddered ETF (the roll cost the front pays)")
    for cmd, row in bt.iterrows():
        front, lad = energy.PAIRS[cmd]
        print(f"  {cmd:3} {front}/{lad}  {row['start']}→{end} ({row['years']}y):  "
              f"{front} {row['front_total_pct']:+.0f}%  vs  {lad} {row['lad_total_pct']:+.0f}%  "
              f"(gap {row['gap_pct']:+.0f}%)")
        print(f"      roll drag on front {row['ann_drag_pct']:+.1f}%/yr   "
              f"in contango {row['weeks_in_contango_pct']:.0f}% of weeks   "
              f"HAC t {row['drag_hac_t']:+.2f}")

    # --- (B) the roll-yield timing book: long the front in backwardation, short it in contango
    print("\n(B) Roll-yield timing book — hold the front only when the curve is backwardated")
    print(f"      {'book':14} {'Sharpe':>7} {'CAGR':>7} {'maxDD':>7} {'skew':>6} {'HAC t':>6} {'turn/yr':>8}")
    for cmd in list(energy.PAIRS) + [None]:
        s = energy.book_summary(prices, commodity=cmd, cost_bps=0.0)
        label = "WTI+GAS combo" if cmd is None else cmd
        print(f"      {label:14} {s['sharpe']:+7.2f} {s['cagr']*100:+6.1f}% {s['max_drawdown']*100:+6.0f}% "
              f"{s['skew']:+6.2f} {s['hac_t']:+6.2f} {s['turnover_per_yr']:8.2f}")
        if cmd is not None:
            al = energy.summary(energy.always_long_front(prices, cmd))
            print(f"        (always-long {energy.PAIRS[cmd][0]}: Sharpe {al['sharpe']:+.2f}  "
                  f"CAGR {al['cagr']*100:+.1f}%  maxDD {al['max_drawdown']*100:+.0f}%)")

    # net of a 10 bp round-trip on the combined book
    net = energy.book_summary(prices, commodity=None, cost_bps=10.0)
    print(f"      {'combo @10bp':14} {net['sharpe']:+7.2f} {net['cagr']*100:+6.1f}% "
          f"{net['max_drawdown']*100:+6.0f}% {net['skew']:+6.2f} {net['hac_t']:+6.2f}")

    # --- reproducibility stamp
    try:
        from quantlab import repro
        rets = prices.pct_change().dropna(how="all")
        print(f"\nas-of {end} · inputs fingerprint {repro.fingerprint(rets.round(6))}")
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    main(ap.parse_args().fetch)
