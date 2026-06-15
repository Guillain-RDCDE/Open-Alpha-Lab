"""Real-tape verification — Study 190 (NR7).  Regenerates docs/results.md numbers.

Fetches (or reads from cache) the six daily tapes, runs the NR7 signal
detection, the range-expansion test, and the breakout trade against a
random-day control.  Network is touched only with --fetch.

    python studies/190-nr7/examples/verify.py            # cache-only
    python studies/190-nr7/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nr7 import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]


def _pool_led(fetch: bool, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        sig  = st.nr7_signals(bars)
        if seed is not None:
            frames.append(st.random_day_control(bars, sig.sum(), cost_bps=cost, seed=seed))
        else:
            frames.append(st.breakout_returns(bars, sig, cost_bps=cost))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    print("Study 190 — NR7 — real-tape verification\n")

    if fetch:
        print("=== Fetching / refreshing tapes ===")
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"  {t:5s}  {b.index[0].date()}..{b.index[-1].date()}  "
                  f"n={len(b)}  fp={data.fingerprint(b)}")
        print()

    # ---- 1. Per-ticker fingerprints & NR7 count --------------------------
    print("=== Data fingerprints & NR7 count ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        sig  = st.nr7_signals(bars)
        print(f"  {t:5s}  {bars.index[0].date()}..{bars.index[-1].date()}  "
              f"nr7={sig.sum():4d}  fp={data.fingerprint(bars)}")

    print()

    # ---- 2. Range-expansion test -----------------------------------------
    print("=== Range-expansion test (next-day range / rolling-baseline) ===")
    all_exp = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        sig  = st.nr7_signals(bars)
        exp  = st.range_expansion_ratio(bars, sig)
        s    = st.summarize_expansion(exp)
        all_exp.append(exp)
        print(f"  {t:5s}  n={s['n']:4d}  mean_ratio={s['mean_ratio']:.4f}  "
              f"frac>1={s['frac_above_1']:.3f}  t={s['tstat']:+.2f}")
    exp_pool = pd.concat(all_exp)
    sp       = st.summarize_expansion(exp_pool)
    print(f"  POOL   n={sp['n']:4d}  mean_ratio={sp['mean_ratio']:.4f}  "
          f"frac>1={sp['frac_above_1']:.3f}  t={sp['tstat']:+.2f}")
    print("  (ratio < 1 means NR7 is followed by a NARROWER range — opposite of the claim)")

    print()

    # ---- 3. Breakout returns vs random-day control ----------------------
    print("=== Breakout returns vs random-day control (gross, cost=0) ===")
    led_pool  = _pool_led(False, 0.0)
    ctrl_pool = _pool_led(False, 0.0, seed=42)
    s_led  = st.summarize(led_pool,  "ret_gross")
    s_ctrl = st.summarize(ctrl_pool, "ret_gross")
    print(f"  NR7 breakout: n={s_led['n_trades']:4d}  mean={s_led['mean_bps']:+.2f}bps  "
          f"win={s_led['win_rate']:.3f}  skew={s_led['skew']:+.2f}  t={s_led['tstat']:+.2f}")
    print(f"  Random ctrl:  n={s_ctrl['n_trades']:4d}  mean={s_ctrl['mean_bps']:+.2f}bps  "
          f"win={s_ctrl['win_rate']:.3f}  t={s_ctrl['tstat']:+.2f}")
    print(f"  NR7 − ctrl gap = {s_led['mean_bps'] - s_ctrl['mean_bps']:+.2f}bps")

    print()

    # ---- 4. Cost sweep ---------------------------------------------------
    print("=== Cost sweep (pooled NR7 breakout, net) ===")
    for cost in (0.0, 2.0, 5.0, 10.0, 20.0):
        led = _pool_led(False, cost)
        s   = st.summarize(led, "ret_net")
        print(f"  cost={cost:4.0f} bps:  net mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    print()

    # ---- 5. SPY alone (cleaner baseline) ---------------------------------
    print("=== SPY stand-alone (the tradable index proxy) ===")
    bars_spy = data.fetch_daily("SPY", fetch=False)
    sig_spy  = st.nr7_signals(bars_spy)
    for cost in (0.0, 2.0, 5.0, 10.0):
        led = st.breakout_returns(bars_spy, sig_spy, cost_bps=cost)
        s   = st.summarize(led, "ret_net")
        print(f"  SPY cost={cost:4.0f} bps:  net mean={s['mean_bps']:+.2f}bps  "
              f"t={s['tstat']:+.2f}  n={s['n_trades']}")

    print()
    print("See docs/results.md for the full narrative and verdict.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
