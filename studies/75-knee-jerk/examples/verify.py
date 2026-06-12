"""Real-tape verification — Study 75 (Knee-Jerk). Regenerates docs/results.md numbers.

Fetches (or reads from cache) SPY daily bars, runs the Connors RSI(2) mean-reversion
rule with and without the 200-SMA trend filter, splits pre/post publication (2009),
pins against a random-direction control, sweeps costs, and shows the publication-decay
picture. Network is touched only with --fetch.

    python studies/75-knee-jerk/examples/verify.py            # cache-only
    python studies/75-knee-jerk/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knee_jerk import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "JPM"]


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, start="1993-01-01", fetch=True)
            print(f"{t:5s}  {b.index[0].date()}..{b.index[-1].date()}  fp={data.fingerprint(b)}")
        print()

    # --- SPY main analysis ---
    spy = data.fetch_daily("SPY", fetch=False)
    print(f"SPY tape: {spy.index[0].date()} to {spy.index[-1].date()}, {len(spy)} days")
    print(f"Fingerprint: {data.fingerprint(spy)}\n")

    # Without trend filter
    sig = st.rsi2_entries(spy, rsi_enter=10.0, trend_sma=None)
    led = st.run_trades(spy, sig, rsi_exit=60.0, max_hold=10, cost_bps=0.0)
    print(f"=== SPY RSI(2)<10, no trend filter, gross ===")
    s = st.summarize(led, "ret_gross")
    print(f"ALL  n={s['n_trades']} win={s['win_rate']:.3f} mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    pre, post = st.split_ledger(led, split_year=2009)
    sp = st.summarize(pre, "ret_gross")
    so = st.summarize(post, "ret_gross")
    print(f"PRE-2009  n={sp['n_trades']} win={sp['win_rate']:.3f} mean={sp['mean_bps']:+.2f}bps t={sp['tstat']:+.2f}")
    print(f"POST-2009 n={so['n_trades']} win={so['win_rate']:.3f} mean={so['mean_bps']:+.2f}bps t={so['tstat']:+.2f}")

    # With trend filter
    sig_f = st.rsi2_entries(spy, rsi_enter=10.0, trend_sma=200)
    led_f = st.run_trades(spy, sig_f, rsi_exit=60.0, max_hold=10, cost_bps=0.0)
    print(f"\n=== SPY RSI(2)<10 + above 200-SMA filter, gross ===")
    sf = st.summarize(led_f, "ret_gross")
    print(f"ALL  n={sf['n_trades']} win={sf['win_rate']:.3f} mean={sf['mean_bps']:+.2f}bps t={sf['tstat']:+.2f}")

    # Random control
    rand_led = st.run_trades(
        spy, sig, rsi_exit=60.0, max_hold=10, cost_bps=0.0,
        random_dirs=st.random_directions(sig.sum(), seed=75),
    )
    sr = st.summarize(rand_led, "ret_gross")
    print(f"\n=== Random-direction control ===")
    print(f"n={sr['n_trades']} win={sr['win_rate']:.3f} mean={sr['mean_bps']:+.2f}bps t={sr['tstat']:+.2f}")

    print(f"\n=== Cost sweep (SPY, no trend filter, net) ===")
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        led_c = st.run_trades(spy, sig, rsi_exit=60.0, max_hold=10, cost_bps=c)
        sc = st.summarize(led_c, "ret_net")
        print(f"cost={c:5.1f}bps  net mean={sc['mean_bps']:+.2f}bps  t={sc['tstat']:+.2f}")

    # Pooled across basket
    print(f"\n=== Pooled basket ({', '.join(TICKERS)}) — no filter, gross ===")
    all_frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        s_ = st.rsi2_entries(b, rsi_enter=10.0, trend_sma=None)
        all_frames.append(st.run_trades(b, s_, rsi_exit=60.0, max_hold=10, cost_bps=0.0))
    pooled = pd.concat(all_frames, ignore_index=True)
    sp2 = st.summarize(pooled, "ret_gross")
    print(f"n={sp2['n_trades']} win={sp2['win_rate']:.3f} mean={sp2['mean_bps']:+.2f}bps t={sp2['tstat']:+.2f}")

    # Post-2009 pooled
    pre_p = pooled[pooled["entry_date"] < pd.Timestamp("2009-01-01")]
    post_p = pooled[pooled["entry_date"] >= pd.Timestamp("2009-01-01")]
    pre_s = st.summarize(pre_p, "ret_gross")
    post_s = st.summarize(post_p, "ret_gross")
    print(f"PRE-2009  n={pre_s['n_trades']} mean={pre_s['mean_bps']:+.2f}bps t={pre_s['tstat']:+.2f}")
    print(f"POST-2009 n={post_s['n_trades']} mean={post_s['mean_bps']:+.2f}bps t={post_s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
