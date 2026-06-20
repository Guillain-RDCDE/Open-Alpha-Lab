#!/usr/bin/env python
"""Reproduce the headline numbers in ../docs/results.md (Study 302 — Lithium-Boom).

Cache-first: reads the parquets under ../_cache/ written by a one-off
``data.fetch_daily(t, fetch=True)``. Prints the data stamp (fingerprints), the LIT headline
table, the honest pin (excess-vs-excess + difference HAC t), the random-timing control, the
SPY opportunity cost, and the synthetic positive control. As-of is fixed to 2026-06-17 so the
fingerprints match regardless of when the cache was last refreshed (later bars are clipped).

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lithium_boom import data, strategy as st  # noqa: E402

ASOF = "2026-06-17"
RF = 0.015
RF_D = RF / st.TRADING_DAYS_PER_YEAR


def load(ticker):
    bars = data.fetch_daily(ticker, fetch=False)
    return bars[bars.index <= ASOF]


def main():
    print(f"As-of {ASOF}\n")
    print("Data stamp")
    for t in data.ALL_TICKERS:
        b = load(t)
        print(f"  {t:4s} {b.index[0].date()} -> {b.index[-1].date()}  "
              f"n={len(b):5d}  fp={data.fingerprint(b)}")

    print("\nLIT headline — 200-day trend overlay vs buy-and-hold")
    b = load("LIT")
    sig = st.trend_signal(b, sma_n=200)
    bt = st.backtest(b, sig, cost_bps=10.0, rf_annual=RF)
    bh = st.summarize(bt["asset_ret"].to_numpy(), rf_daily=RF_D)
    g = st.summarize(bt["strat_gross"].to_numpy(), rf_daily=RF_D)
    n = st.summarize(bt["strat_net"].to_numpy(), rf_daily=RF_D)
    print(f"  B&H        CAGR {bh['cagr']*100:+.1f}%  Sharpe {bh['sharpe']:.2f}  "
          f"maxDD {bh['max_dd']*100:.0f}%  HACt {bh['hac_t']:+.2f}")
    print(f"  Trend gross CAGR {g['cagr']*100:+.1f}%  Sharpe {g['sharpe']:.2f}  "
          f"maxDD {g['max_dd']*100:.0f}%  HACt {g['hac_t']:+.2f}")
    print(f"  Trend net   CAGR {n['cagr']*100:+.1f}%  Sharpe {n['sharpe']:.2f}  "
          f"maxDD {n['max_dd']*100:.0f}%  HACt {n['hac_t']:+.2f}")
    lo, hi = st.block_bootstrap_sharpe_ci(bt["strat_net"].to_numpy(), rf_daily=RF_D)
    print(f"  Trend net Sharpe 95% CI [{lo:.2f}, {hi:.2f}]")

    print("\nHonest pin — does the timing beat buy-and-hold?")
    for t in data.THEME_TICKERS:
        bb = load(t)
        ss = st.trend_signal(bb, sma_n=200)
        bbt = st.backtest(bb, ss, cost_bps=10.0, rf_annual=RF)
        diff = bbt["strat_net"].to_numpy() - bbt["asset_ret"].to_numpy()
        print(f"  {t:4s} trend net excess-Sharpe {st.sharpe(bbt['strat_net'].to_numpy(), rf_daily=RF_D):.2f}  "
              f"(Trend-B&H) diff ann {np.nanmean(diff)*252*100:+.2f}%  diff HACt {st.hac_tstat(diff):+.2f}")

    print("\nRandom-timing control (same exposure, random days)")
    for t in data.THEME_TICKERS:
        bb = load(t)
        ss = st.trend_signal(bb, sma_n=200)
        bbt = st.backtest(bb, ss, cost_bps=10.0, rf_annual=RF)
        expo = bbt["pos"].mean()
        tr = st.sharpe(bbt["strat_net"].to_numpy(), rf_daily=RF_D)
        sh = [st.sharpe(st.backtest(bb, st.random_timing_signal(len(bb), expo, seed=s),
                                    cost_bps=10.0, rf_annual=RF)["strat_net"].to_numpy(),
                        rf_daily=RF_D) for s in range(200)]
        pct = (np.array(sh) < tr).mean() * 100
        print(f"  {t:4s} trend net Sharpe {tr:.2f}  random mean {np.nanmean(sh):.2f}  "
              f"trend at {pct:.0f}th pctile")

    print("\nOpportunity cost — SPY")
    sb = load("SPY")
    print(f"  SPY B&H excess-Sharpe {st.sharpe(sb['close'].pct_change().fillna(0).to_numpy(), rf_daily=RF_D):.2f}")

    print("\nSynthetic positive control")
    for ts in [0.0, 0.05, 0.10, 0.15]:
        sbar, _ = data.synthetic_theme(n_days=4000, trend_strength=ts, seed=302)
        sg = st.trend_signal(sbar, sma_n=200)
        sbt = st.backtest(sbar, sg, cost_bps=10.0)
        bh_sh = st.sharpe(sbt["asset_ret"].to_numpy())
        tr_sh = st.sharpe(sbt["strat_net"].to_numpy())
        diff = sbt["strat_net"].to_numpy() - sbt["asset_ret"].to_numpy()
        print(f"  ts={ts:.2f}  B&H Sh {bh_sh:.2f} (DD {st.max_drawdown(sbt['asset_ret'].to_numpy())*100:.0f}%)  "
              f"Trend Sh {tr_sh:.2f} (DD {st.max_drawdown(sbt['strat_net'].to_numpy())*100:.0f}%)  "
              f"edge {tr_sh-bh_sh:+.2f}  diff HACt {st.hac_tstat(diff):+.2f}")


if __name__ == "__main__":
    main()
