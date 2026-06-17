"""Headline run for Study 247 (Bond-Seasonality) on the real tapes.

Measures the turn-of-month (TOM) effect in long Treasuries - TLT total-return (20+ yr)
and IEF total-return (7-10 yr) - derives the TOM window from the trading index, runs
the TOM-only book against buy-and-hold, and also checks the turn-of-year (TOY) effect.
Stamps the as-of date and content fingerprint. Re-run to reproduce; match the fingerprint
to confirm you hold the same tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from bond_seasonality import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
COST_BPS = 1.0


def _run(ticker: str, label: str) -> None:
    frame = data.load_real(ticker, start="2002-07-30").loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    is_tom = data.tom_mask(close.index, n_end=2, n_start=2)

    print(f"\n=== {label} : {ticker} (total-return) ===")
    print(
        f"[data] {len(close):,} rows  {close.index[0].date()} -> {close.index[-1].date()}  "
        f"as-of {AS_OF}  fingerprint={fp}"
    )
    print(f"[TOM] n_tom={int(is_tom.sum())}  ({int(is_tom.sum())/len(close)*100:.1f}% of days)")

    e = strategy.effect_stats(close, is_tom)
    print(
        f"  TOM mean = {e['mu_tom']*1e4:6.2f} bps/day   "
        f"rest mean = {e['mu_rest']*1e4:5.2f} bps/day   "
        f"gap = {e['gap']*1e4:+.2f} bps"
    )
    print(
        f"  HAC t(TOM) = {e['t_tom']:+.3f}   HAC t(gap) = {e['t_gap']:+.3f}"
    )
    wt = e["wilson_tom"]
    wr = e["wilson_rest"]
    print(
        f"  win-rate TOM = {e['win_tom']*100:.1f}% [{wt[0]*100:.1f}, {wt[1]*100:.1f}]   "
        f"rest = {e['win_rest']*100:.1f}% [{wr[0]*100:.1f}, {wr[1]*100:.1f}]"
    )

    book = strategy.tom_only_book(close, is_tom, cost_bps=COST_BPS)
    bh = strategy.buy_and_hold(close)
    print(
        f"  [TOM book] days invested = {book['days_invested']} "
        f"({book['time_in_market']*100:.1f}% of tape)   "
        f"CAGR = {book['cagr']*100:.2f}%   "
        f"Sharpe(whole) = {book['sharpe_whole']:.3f}   "
        f"Sharpe(per invested day) = {book['sharpe_inv']:.3f}   "
        f"MaxDD = {book['max_dd']*100:.1f}%"
    )
    print(f"  [buy & hold] CAGR = {bh['cagr']*100:.2f}%   Sharpe = {bh['sharpe']:.3f}")

    # Turn-of-year
    import pandas as pd
    ret = close.pct_change().dropna()
    toy = pd.Series(False, index=ret.index)
    for yr in ret.index.year.unique():
        days = ret.index[ret.index.year == yr].sort_values()
        for d in days[:5]:
            toy.loc[d] = True
        for d in days[-5:]:
            toy.loc[d] = True

    etoy = strategy.effect_stats(close, toy)
    print(
        f"  [TOY] gap = {etoy['gap']*1e4:+.2f} bps   "
        f"HAC t(gap) = {etoy['t_gap']:+.3f}   "
        f"n_toy = {etoy['n_tom']}"
    )


def main() -> None:
    _run("TLT", "TLT (20+ yr Treasury)")
    _run("IEF", "IEF (7-10 yr Treasury)")


if __name__ == "__main__":
    main()
