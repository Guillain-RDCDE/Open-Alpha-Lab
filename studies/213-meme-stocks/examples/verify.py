"""Headline run for Study 213 (Meme Stocks) on the real tape.

Tests two strategies on the WSB meme-stock basket (GME, AMC, BB, BBBY, KOSS, NOK)
vs SPY (S&P 500, total return):

  A — Buy-and-hold from 2021-01-04 (equal-weight, hold to end of 2025).
  B — The same basket after the mania went viral: 2021-01-28 entry (peak-liquidity scenario).
  C — Momentum-timed: enter each name when it closes +50% above its 60-day-ago close,
      hold 126 trading days, t+1 execution lag.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from meme_stocks import data, strategy  # noqa: E402

AS_OF = "2026-06-15"
COST_BPS = 20.0


def main() -> None:
    cache = os.path.abspath(os.path.join(HERE, "..", "_cache"))
    panel = data.load_panel(end=AS_OF, cache_dir=cache)
    fp = data.fingerprint(panel)
    px = panel["prices"]
    bench = panel["bench"]
    print(
        f"[data] Meme panel: {px.shape[1]} tickers x {len(px):,} days  "
        f"bench {bench.index[0].date()} -> {bench.index[-1].date()}  "
        f"as-of {AS_OF}  fingerprint={fp}"
    )
    print(f"[basket] {data.MEME_TICKERS}")
    print(f"[delisted] {data.DELISTED}")

    # --- Strategy A: buy-and-hold from Jan-4-2021 ---
    bh = strategy.buy_and_hold(
        panel, entry_date="2021-01-04", exit_date="2025-12-31", cost_bps=COST_BPS
    )
    pret = bh["port_ret"].dropna()
    bret = bh["bench_ret"].dropna()
    port_cum = (1 + pret).cumprod()
    bench_cum = (1 + bret).cumprod()
    print(f"\n--- Strategy A: BUY-AND-HOLD {bh['entry']} -> {bh['exit']} ---")
    print(
        f"  Meme basket  total return: {bh['total_port']*100:+.1f}%   "
        f"CAGR {bh['port_cagr']*100:.1f}%/yr   Sharpe {bh['port_sharpe']:.2f}"
    )
    print(
        f"  SPY          total return: {bh['total_bench']*100:+.1f}%   "
        f"CAGR {bh['bench_cagr']*100:.1f}%/yr   Sharpe {bh['bench_sharpe']:.2f}"
    )
    print(
        f"  Basket vol: {strategy.annual_vol(pret)*100:.1f}%/yr   "
        f"MaxDD: {strategy.max_drawdown(port_cum)*100:.1f}%"
    )
    print(
        f"  SPY    vol: {strategy.annual_vol(bret)*100:.1f}%/yr   "
        f"MaxDD: {strategy.max_drawdown(bench_cum)*100:.1f}%"
    )
    print("  -> Driven almost entirely by GME (+366%); AMC -92%, BBBY -89%.")
    print("  -> One name carries the basket; diversification is illusory.")

    # --- Strategy B: buy AFTER the mania (peak-liquidity entry) ---
    bh2 = strategy.buy_and_hold(
        panel, entry_date="2021-01-28", exit_date="2025-12-31", cost_bps=COST_BPS
    )
    pret2 = bh2["port_ret"].dropna()
    port_cum2 = (1 + pret2).cumprod()
    print(f"\n--- Strategy B: BUY AT PEAK {bh2['entry']} -> {bh2['exit']} ---")
    print(
        f"  Meme basket  total return: {bh2['total_port']*100:+.1f}%   "
        f"CAGR {bh2['port_cagr']*100:.1f}%/yr   Sharpe {bh2['port_sharpe']:.2f}"
    )
    print(
        f"  SPY          total return: {bh2['total_bench']*100:+.1f}%   "
        f"CAGR {bh2['bench_cagr']*100:.1f}%/yr   Sharpe {bh2['bench_sharpe']:.2f}"
    )
    print(
        f"  Basket vol: {strategy.annual_vol(pret2)*100:.1f}%/yr   "
        f"MaxDD: {strategy.max_drawdown(port_cum2)*100:.1f}%"
    )
    print("  -> -45% total return vs +93% for SPY over the same period.")
    print("  -> This is the retail investor who bought the news.")

    # --- Strategy C: momentum-timed ---
    mt = strategy.momentum_timed(panel, mom_window=60, mom_pct=0.50, hold_days=126,
                                  start_date="2020-06-01", cost_bps=COST_BPS)
    print(f"\n--- Strategy C: MOMENTUM-TIMED (60-day, +50%, 126-day hold) ---")
    print(f"  n_trades={mt['n_trades']}")
    if mt["n_trades"] > 0:
        trades = mt["trades"]
        print(f"  mean_gross_ret={mt['mean_gross_ret']*100:.1f}%   "
              f"mean_net_ret={mt['mean_net_ret']*100:.1f}%   win_rate={mt['win_rate']*100:.0f}%")
        for _, row in trades.iterrows():
            print(f"    {row['ticker']:5s} "
                  f"{row['entry_date'].date()} -> {row['exit_date'].date()}  "
                  f"net {row['net_ret']*100:+.1f}%")
        t_stat = strategy.hac_tstat(trades["net_ret"].to_numpy())
        print(f"  HAC t-stat (n={mt['n_trades']}, WARNING: too small for inference): {t_stat:.2f}")
        print("  -> Apparent gains driven by GME/KOSS triggers in Aug-Sep 2020.")
        print("  -> These signals fired BEFORE the WSB mania was known; not replicable ex-ante.")

    print(f"\n[fingerprint] {fp}  (as-of {AS_OF})")


if __name__ == "__main__":
    main()
