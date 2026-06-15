"""Real-tape verification — Study 210 (Crypto-Trend). Regenerates docs/results.md numbers.

Fetches (or reads from cache) BTC-USD daily closes back to 2014-09-17, runs the
200-day SMA timing rule vs buy-and-hold and a random-timing control, and reports
Sharpe, CAGR, max drawdown and HAC t-stats. Network is touched only with --fetch.

    python studies/210-crypto-trend/examples/verify.py             # cache-only
    python studies/210-crypto-trend/examples/verify.py --fetch     # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_trend import data, strategy as st  # noqa: E402

TICKER = "BTC-USD"
SMA_N = 200
TBILL_RATE = 0.04       # flat proxy for cash yield
COST_BPS = 10.0         # one-way transaction cost (bps); crypto is more expensive than equity ETFs


def main(fetch: bool) -> None:
    if fetch:
        df = data.fetch_prices(TICKER, start="2014-09-17", fetch=True)
    else:
        df = data.fetch_prices(TICKER, fetch=False)

    print(
        f"{TICKER}: {df.index[0].date()} -> {df.index[-1].date()}  "
        f"n={len(df)}  fp={data.fingerprint(df)}"
    )

    close = df["close"]
    tbill_daily = TBILL_RATE / st.TRADING_DAYS_PER_YEAR

    sig = st.timing_signal(close, sma_n=SMA_N)
    rand_sig = st.random_timing_signal(sig, seed=42)

    bt_timing = st.run_backtest(close, sig, tbill_daily=tbill_daily, cost_bps=COST_BPS)
    bt_random = st.run_backtest(close, rand_sig, tbill_daily=tbill_daily, cost_bps=COST_BPS)

    bh = st.summary(bt_timing["r_bh"])
    timing = st.summary(bt_timing["r_strategy"])
    random = st.summary(bt_random["r_strategy"])

    in_mkt = float(sig.dropna().mean())
    n_switches = int(sig.diff().abs().fillna(0).sum())

    t_timing_vs_bh = st.sharpe_diff_tstat(bt_timing["r_strategy"], bt_timing["r_bh"])
    t_timing_vs_rand = st.sharpe_diff_tstat(bt_timing["r_strategy"], bt_random["r_strategy"])

    def fmt(s: dict) -> str:
        return (
            f"CAGR={s['cagr']:+.2%}  Sharpe={s['sharpe']:+.3f}  "
            f"Vol={s['vol_ann']:.2%}  MaxDD={s['max_drawdown']:+.2%}  "
            f"t={s['tstat']:+.2f}"
        )

    print(f"\nIn-market fraction: {in_mkt:.1%}  |  Total SMA switches: {n_switches}")
    print(f"\nBuy-and-hold (BTC):     {fmt(bh)}")
    print(f"SMA(200) timing:        {fmt(timing)}")
    print(f"Random timing (null):   {fmt(random)}")
    print(f"\nSharpe diff t-stat timing vs BH:     {t_timing_vs_bh:+.2f}")
    print(f"Sharpe diff t-stat timing vs random:  {t_timing_vs_rand:+.2f}")
    print(f"\nDrawdown improvement (timing vs BH):  {bh['max_drawdown'] - timing['max_drawdown']:+.2%}")
    print(f"(random control MaxDD: {random['max_drawdown']:+.2%})")

    # Cost sweep
    print("\n=== cost sweep (SMA timing, varying one-way bps) ===")
    for c in (0.0, 1.0, 5.0, 10.0, 25.0, 50.0):
        bt = st.run_backtest(close, sig, tbill_daily=tbill_daily, cost_bps=c)
        s = st.summary(bt["r_strategy"])
        print(
            f"  cost={c:5.1f}bps  Sharpe={s['sharpe']:+.3f}  "
            f"MaxDD={s['max_drawdown']:+.2%}  CAGR={s['cagr']:+.2%}"
        )

    # Sub-period breakdown
    print("\n=== sub-period breakdown (SMA timing vs BH) ===")
    periods = [
        ("2014-2017", "2014-09-17", "2017-12-31"),
        ("2018-2020", "2018-01-01", "2020-12-31"),
        ("2021-2022", "2021-01-01", "2022-12-31"),
        ("2023-2026", "2023-01-01", "2026-12-31"),
    ]
    for name, s_date, e_date in periods:
        sub = close.loc[s_date:e_date]
        if len(sub) < 250:
            continue
        sig_sub = st.timing_signal(sub, sma_n=SMA_N)
        bt_sub = st.run_backtest(sub, sig_sub, tbill_daily=tbill_daily, cost_bps=COST_BPS)
        t_sub = st.summary(bt_sub["r_strategy"])
        b_sub = st.summary(bt_sub["r_bh"])
        print(
            f"  {name:12s}: BH Sharpe={b_sub['sharpe']:+.2f} MaxDD={b_sub['max_drawdown']:+.1%}"
            f"  |  Timing Sharpe={t_sub['sharpe']:+.2f} MaxDD={t_sub['max_drawdown']:+.1%}"
        )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
