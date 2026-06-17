"""Real-tape verification — Study 301 (Triple-RSI). Regenerates docs/results.md numbers.

Fetches (or reads from cache) SPY daily bars, runs the Triple-RSI recipe (RSI(5)<30 +
down 3 days + RSI[t-3]<60 + close>SMA200; exit when RSI(5)>50), pins it against a
random-direction control on the same entry bars, reports win-rate vs expectancy vs skew
(the "90% win rate" interrogation), sweeps costs, and re-runs with the conservative
next-open fill. Network is touched only with --fetch.

    python studies/301-triple-rsi/examples/verify.py            # cache-only
    python studies/301-triple-rsi/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from triple_rsi import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "DIA"]


def _line(tag, s):
    print(
        f"{tag:22s} n={s['n_trades']:>4} win={s['win_rate']*100:5.1f}% "
        f"mean={s['mean_bps']:+7.1f}bps skew={s['skew']:+5.2f} t={s['tstat']:+6.2f}"
    )


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, start="1993-01-01", fetch=True)
            print(f"{t:5s}  {b.index[0].date()}..{b.index[-1].date()}  "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    spy = data.fetch_daily("SPY", fetch=False)
    print(f"SPY tape: {spy.index[0].date()} to {spy.index[-1].date()}, {len(spy)} days")
    print(f"Fingerprint: {data.fingerprint(spy)}\n")

    # --- canonical recipe, close fill, gross ---
    sig = st.triple_rsi_entries(spy)
    led = st.run_trades(spy, sig, rsi_exit=50.0, cost_bps=0.0, entry="close")
    print("=== SPY Triple-RSI (canonical), close fill, gross ===")
    _line("ALL", st.summarize(led, "ret_gross"))

    pre, post = st.split_ledger(led, split_year=2010)
    _line("PRE-2010", st.summarize(pre, "ret_gross"))
    _line("POST-2010", st.summarize(post, "ret_gross"))

    # --- random-direction control on the same entries ---
    rand = st.run_trades(
        spy, sig, rsi_exit=50.0, cost_bps=0.0, entry="close",
        random_dirs=st.random_directions(int(sig.sum()), seed=301),
    )
    print("\n=== Random-direction control (same entry bars) ===")
    _line("RANDOM", st.summarize(rand, "ret_gross"))

    # --- next-open fill (look-ahead-free robustness) ---
    led_no = st.run_trades(spy, sig, rsi_exit=50.0, cost_bps=0.0, entry="next_open")
    print("\n=== SPY Triple-RSI, conservative NEXT-OPEN fill, gross ===")
    _line("ALL (next_open)", st.summarize(led_no, "ret_gross"))

    # --- no trend filter ---
    sig_nf = st.triple_rsi_entries(spy, trend_sma=None)
    led_nf = st.run_trades(spy, sig_nf, rsi_exit=50.0, cost_bps=0.0, entry="close")
    print("\n=== SPY Triple-RSI WITHOUT the 200-SMA filter, gross ===")
    _line("ALL (no filter)", st.summarize(led_nf, "ret_gross"))

    # --- cost sweep ---
    print("\n=== Cost sweep (SPY, canonical, close fill, net) ===")
    for c in (0.0, 2.0, 5.0, 10.0):
        sc = st.summarize(st.run_trades(spy, sig, rsi_exit=50.0, cost_bps=c, entry="close"), "ret_net")
        tpy = sc["n_trades"] / ((spy.index[-1] - spy.index[0]).days / 365.25)
        print(f"cost={c:5.1f}bps  net mean={sc['mean_bps']:+7.1f}bps  t={sc['tstat']:+6.2f}  "
              f"win={sc['win_rate']*100:5.1f}%  ~{sc['mean_bps']*tpy/100:+.2f}%/yr")

    # --- buy & hold benchmark ---
    yrs = (spy.index[-1] - spy.index[0]).days / 365.25
    bh = (spy["close"].iloc[-1] / spy["close"].iloc[0]) ** (1 / yrs) - 1
    n_trades = int(sig.sum())
    print(f"\nSPY buy&hold: {bh*100:+.1f}%/yr over {yrs:.1f}y   "
          f"Triple-RSI: {n_trades} trades ({n_trades/yrs:.1f}/yr), "
          f"avg hold {led['days_held'].mean():.1f}d, "
          f"~{led['days_held'].sum()/len(spy)*100:.1f}% time in market")

    # --- pooled basket ---
    print(f"\n=== Pooled basket ({', '.join(TICKERS)}), close fill, gross ===")
    frames, rand_frames = [], []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        s_ = st.triple_rsi_entries(b)
        frames.append(st.run_trades(b, s_, rsi_exit=50.0, cost_bps=0.0, entry="close"))
        rand_frames.append(st.run_trades(
            b, s_, rsi_exit=50.0, cost_bps=0.0, entry="close",
            random_dirs=st.random_directions(int(s_.sum()), seed=301)))
    pooled = pd.concat(frames, ignore_index=True)
    pooled_rand = pd.concat(rand_frames, ignore_index=True)
    _line("POOLED rule", st.summarize(pooled, "ret_gross"))
    _line("POOLED random", st.summarize(pooled_rand, "ret_gross"))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
