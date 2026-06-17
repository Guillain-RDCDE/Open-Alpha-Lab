"""Real-tape verification — Study 212 (Cannabis Stocks). Regenerates docs/results.md numbers.

Loads (or fetches) cannabis tickers and SPY daily adjusted closes, runs the three-pronged analysis:
1. OLS beta/alpha of MSOS on SPY (the alpha claim)
2. Buy-and-hold comparison: SPY, MJ, MSOS, TLRY, CGC, CRON (the wealth-shredder thesis)
3. Cannabis momentum timing rule vs SPY buy-and-hold

Network is touched only with --fetch.

    python studies/212-cannabis-stocks/examples/verify.py            # cache-only
    python studies/212-cannabis-stocks/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cannabis_stocks import data, strategy as st  # noqa: E402

TICKERS = ["MJ", "MSOS", "TLRY", "CGC", "CRON", "SPY"]


def main(fetch: bool) -> None:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_cache")

    if fetch:
        print("Fetching all tickers from Yahoo Finance...")
        for ticker in TICKERS:
            start = data.TICKER_START.get(ticker, "2017-01-01")
            df = data.fetch_prices(ticker, start=start, fetch=True, cache_dir=cache_dir)
            print(f"{ticker}: {df.index[0].date()} -> {df.index[-1].date()}, n={len(df)}")

    # ---- Primary MSOS/SPY pair (since MSOS inception) -------------------------
    pair = data.load_pair("MSOS", fetch=False, cache_dir=cache_dir, start=data.MSOS_START)
    spy_fp = data.fingerprint(pair, "spy")
    can_fp = data.fingerprint(pair, "cannabis")
    print(f"\nMSOS/SPY common tape: {pair.index[0].date()} -> {pair.index[-1].date()}, n={len(pair)}")
    print(f"SPY fingerprint: {spy_fp}")
    print(f"MSOS fingerprint: {can_fp}")

    print("\n=== Test 1: OLS Beta / Alpha (MSOS on SPY) ===")
    ba = st.beta_alpha(pair)
    print(f"n = {ba['n']}")
    print(f"beta = {ba['beta']:.4f}  HAC t = {ba['t_beta']:+.2f}")
    print(f"alpha = {ba['alpha_daily_bps']:+.2f} bps/day = {ba['alpha_ann_pct']:+.2f}%/yr  HAC t = {ba['t_alpha']:+.2f}")
    print(f"R^2 = {ba['r_squared']:.4f}")
    print(f"SPY vol = {ba['spy_vol_ann_pct']:.1f}%/yr  MSOS vol = {ba['can_vol_ann_pct']:.1f}%/yr")
    print(f"Idiosyncratic vol = {ba['idio_vol_ann_pct']:.1f}%/yr")
    print(f"Vol-ratio (implied leverage) = {ba['implied_leverage']:.4f}")

    # ---- Test 2: Multi-ticker performance table --------------------------------
    print("\n=== Test 2: Buy-and-Hold Performance Table ===")
    # Use a common window where most tickers exist — let's align on MSOS start for the ETFs
    # and separately show individual names from their own start
    # Full suite aligned on MSOS start (shortest ETF)
    print("\n-- MSOS-era window (since 2020-09-02) --")
    msos_tickers = ["MSOS", "MJ", "TLRY", "CGC", "CRON", "SPY"]
    try:
        multi = data.load_multi(msos_tickers, fetch=False, cache_dir=cache_dir, start=data.MSOS_START)
        perf = st.perf_table(multi)
        header = f"{'Ticker':<8} {'CAGR':>8} {'Sharpe':>8} {'Vol':>8} {'MaxDD':>8} {'t-stat':>8}"
        print(header)
        print("-" * len(header))
        for t in msos_tickers:
            col = t.lower()
            if col in perf:
                p = perf[col]
                print(f"{t:<8} {p['cagr_pct']:>+7.1f}% {p['sharpe']:>+8.3f} "
                      f"{p['vol_ann_pct']:>7.1f}% {p['max_drawdown_pct']:>+7.1f}% {p['tstat']:>+7.2f}")
    except FileNotFoundError as e:
        print(f"Skipping multi table (cache absent): {e}")

    # MJ-era window (since 2017-11-08)
    print("\n-- MJ-era window (since 2017-11-08) --")
    mj_tickers = ["MJ", "TLRY", "CGC", "CRON", "SPY"]
    try:
        multi_mj = data.load_multi(mj_tickers, fetch=False, cache_dir=cache_dir, start=data.MJ_START)
        perf_mj = st.perf_table(multi_mj)
        header = f"{'Ticker':<8} {'CAGR':>8} {'Sharpe':>8} {'Vol':>8} {'MaxDD':>8} {'t-stat':>8}"
        print(header)
        print("-" * len(header))
        for t in mj_tickers:
            col = t.lower()
            if col in perf_mj:
                p = perf_mj[col]
                print(f"{t:<8} {p['cagr_pct']:>+7.1f}% {p['sharpe']:>+8.3f} "
                      f"{p['vol_ann_pct']:>7.1f}% {p['max_drawdown_pct']:>+7.1f}% {p['tstat']:>+7.2f}")
    except FileNotFoundError as e:
        print(f"Skipping MJ table (cache absent): {e}")

    # ---- Test 3: Timing rule --------------------------------------------------
    print("\n=== Test 3: MSOS Momentum Timing Rule vs SPY Buy-and-Hold ===")
    try:
        res = st.compare_strategies(pair, sma_n=200, cost_bps=5.0)
        print(f"In-market fraction (cannabis): {res['in_market_frac']:.1%}")
        print(f"Number of switches: {res['n_switches']}")
        print()
        for name in ("spy", "cannabis", "timing"):
            s = res[name]
            label = {"spy": "SPY buy-and-hold", "cannabis": "MSOS buy-and-hold", "timing": "MSOS timing"}[name]
            print(f"{label}:")
            print(f"  CAGR = {s['cagr_pct']:+.2f}%  Sharpe = {s['sharpe']:+.4f}  vol = {s['vol_ann_pct']:.1f}%/yr"
                  f"  maxDD = {s['max_drawdown_pct']:.1f}%  HAC t = {s['tstat']:+.2f}")
    except FileNotFoundError as e:
        print(f"Skipping timing test (cache absent): {e}")

    print("\n=== Summary fingerprint ===")
    print(f"SPY fingerprint: {spy_fp}  |  MSOS fingerprint: {can_fp}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
