"""Real-tape verification — Study 128 (Keltner-Channel). Regenerates docs/results.md numbers.

Tests BOTH Keltner folk framings — breakout (close above upper band = momentum long) and
reversion (close below lower band = buy the dip) — on a basket of daily tapes, each arm
pinned against a random-entry control matched in trade count so the drift of the tape is
correctly priced-in.  Network is touched only with --fetch.

    python studies/128-keltner-channel/examples/verify.py            # cache-only
    python studies/128-keltner-channel/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keltner_channel import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "EEM"]
HOLD = 20
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def _pool(fetch: bool, side: str, cost: float, seed: int | None = None):
    """Pool trades across the basket for one arm."""
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch, cache_dir=CACHE)
        if seed is not None:
            ent = st.random_entries(bars, n=len(st.channel_entries(bars, side="breakout")), seed=seed)
        else:
            ent = st.channel_entries(bars, side=side)
        frames.append(st.run_trades(bars, ent, hold_days=HOLD, cost_bps=cost))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=CACHE)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== BREAKOUT (close > upper Keltner band) — vs random-entry control ===")
    print(f"{'ticker':>6} | {'n':>5} | {'mean_bps':>9} | {'win%':>6} | {'t':>7} | {'rand_bps':>9} | {'delta':>8}")
    print("-" * 75)
    bo_frames, rand_frames_bo = [], []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
        ent = st.channel_entries(bars, side="breakout")
        n = len(ent)
        s = st.summarize(st.run_trades(bars, ent, hold_days=HOLD, cost_bps=0), "ret_gross")
        rnd_ent = st.random_entries(bars, n=n, seed=128)
        r = st.summarize(st.run_trades(bars, rnd_ent, hold_days=HOLD, cost_bps=0), "ret_gross")
        bo_frames.append(st.run_trades(bars, ent, hold_days=HOLD, cost_bps=0))
        rand_frames_bo.append(st.run_trades(bars, rnd_ent, hold_days=HOLD, cost_bps=0))
        delta = s["mean_bps"] - r["mean_bps"]
        print(f"{t:>6} | {n:>5} | {s['mean_bps']:>+9.2f} | {s['win_rate']*100:>6.1f} | "
              f"{s['tstat']:>+7.2f} | {r['mean_bps']:>+9.2f} | {delta:>+8.2f}")
    pbo = pd.concat(bo_frames, ignore_index=True)
    prand_bo = pd.concat(rand_frames_bo, ignore_index=True)
    sbo = st.summarize(pbo, "ret_gross")
    rbo = st.summarize(prand_bo, "ret_gross")
    delta_bo = sbo["mean_bps"] - rbo["mean_bps"]
    print(f"{'POOLED':>6} | {sbo['n_trades']:>5} | {sbo['mean_bps']:>+9.2f} | {sbo['win_rate']*100:>6.1f} | "
          f"{sbo['tstat']:>+7.2f} | {rbo['mean_bps']:>+9.2f} | {delta_bo:>+8.2f}")

    print()
    print("=== REVERSION (close < lower Keltner band) — vs random-entry control ===")
    print(f"{'ticker':>6} | {'n':>5} | {'mean_bps':>9} | {'win%':>6} | {'t':>7} | {'rand_bps':>9} | {'delta':>8}")
    print("-" * 75)
    rv_frames, rand_frames_rv = [], []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
        ent = st.channel_entries(bars, side="reversion")
        n = len(ent)
        s = st.summarize(st.run_trades(bars, ent, hold_days=HOLD, cost_bps=0), "ret_gross")
        rnd_ent = st.random_entries(bars, n=n, seed=128)
        r = st.summarize(st.run_trades(bars, rnd_ent, hold_days=HOLD, cost_bps=0), "ret_gross")
        rv_frames.append(st.run_trades(bars, ent, hold_days=HOLD, cost_bps=0))
        rand_frames_rv.append(st.run_trades(bars, rnd_ent, hold_days=HOLD, cost_bps=0))
        delta = s["mean_bps"] - r["mean_bps"]
        print(f"{t:>6} | {n:>5} | {s['mean_bps']:>+9.2f} | {s['win_rate']*100:>6.1f} | "
              f"{s['tstat']:>+7.2f} | {r['mean_bps']:>+9.2f} | {delta:>+8.2f}")
    prv = pd.concat(rv_frames, ignore_index=True)
    prand_rv = pd.concat(rand_frames_rv, ignore_index=True)
    srv = st.summarize(prv, "ret_gross")
    rrv = st.summarize(prand_rv, "ret_gross")
    delta_rv = srv["mean_bps"] - rrv["mean_bps"]
    print(f"{'POOLED':>6} | {srv['n_trades']:>5} | {srv['mean_bps']:>+9.2f} | {srv['win_rate']*100:>6.1f} | "
          f"{srv['tstat']:>+7.2f} | {rrv['mean_bps']:>+9.2f} | {delta_rv:>+8.2f}")

    print()
    print("=== COST SWEEP (breakout, pooled, net) ===")
    for c in [0.0, 1.0, 2.0, 5.0]:
        frames = []
        for t in TICKERS:
            bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
            ent = st.channel_entries(bars, side="breakout")
            frames.append(st.run_trades(bars, ent, hold_days=HOLD, cost_bps=c))
        led = pd.concat(frames, ignore_index=True)
        s = st.summarize(led, "ret_net")
        tpy = int(round(s["n_trades"] / len(TICKERS) / 21 * 252))
        print(f"cost={c:4.1f} bps | net mean={s['mean_bps']:>+7.2f} bps | t={s['tstat']:>+6.2f} | ~{tpy} trades/yr")

    print()
    print("=== COST SWEEP (reversion, pooled, net) ===")
    for c in [0.0, 1.0, 2.0, 5.0]:
        frames = []
        for t in TICKERS:
            bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
            ent = st.channel_entries(bars, side="reversion")
            frames.append(st.run_trades(bars, ent, hold_days=HOLD, cost_bps=c))
        led = pd.concat(frames, ignore_index=True)
        s = st.summarize(led, "ret_net")
        tpy = int(round(s["n_trades"] / len(TICKERS) / 21 * 252))
        print(f"cost={c:4.1f} bps | net mean={s['mean_bps']:>+7.2f} bps | t={s['tstat']:>+6.2f} | ~{tpy} trades/yr")

    print()
    print("=== BUY-AND-HOLD BASELINE (unconditional 20-day mean forward return) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
        bh = st.buy_and_hold_mean_bps(bars, hold_days=HOLD)
        print(f"{t}: {bh:+.2f} bps over {HOLD}-day hold")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
