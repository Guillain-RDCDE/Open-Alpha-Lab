"""Real-tape verification — Study 137 (Mansfield-RS).  Regenerates docs/results.md numbers.

Loads (or fetches) the 16-ticker weekly basket, runs Stage-2 entry signals (price >
rising 30-week SMA AND Mansfield RS > 0), measures 13-week forward excess returns vs SPY,
and compares against a random-entry control on the same stocks.  Network only with --fetch.

    python studies/137-mansfield-rs/examples/verify.py            # cache-only
    python studies/137-mansfield-rs/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mansfield_rs import data, strategy as st  # noqa: E402

TICKERS = data.DEFAULT_TICKERS
STOCKS = [t for t in TICKERS if t != "SPY"]
HOLD_WEEKS = 13


def _pool(tapes, cost, seed=None):
    bm = tapes["SPY"]
    frames = []
    for tk in STOCKS:
        stock = tapes[tk]
        entries = st.stage2_entries(stock, bm, sma_n=30)
        if entries.empty:
            continue
        if seed is not None:
            entries = st.random_entries(stock, n_entries=len(entries), seed=seed)
        led = st.run_forward_returns(stock, bm, entries, hold_weeks=HOLD_WEEKS, cost_bps=cost)
        frames.append(led)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(fetch: bool) -> None:
    tapes = {t: data.fetch_weekly(t, fetch=fetch, cache_dir=data.DEFAULT_CACHE) for t in TICKERS}

    if fetch:
        print("=== Data stamps ===")
        for t in TICKERS:
            b = tapes[t]
            print(f"  {t:6s} {b.index[0].date()} .. {b.index[-1].date()}  n={len(b)}  fp={data.fingerprint(b)}")
        print()

    pooled_s2 = _pool(tapes, cost=0.0, seed=None)
    pooled_rnd = _pool(tapes, cost=0.0, seed=137)

    s2 = st.summarize(pooled_s2, "ret_gross")
    rnd = st.summarize(pooled_rnd, "ret_gross")

    print("=== Stage-2 entry vs random-entry control (13-week forward excess return, gross) ===")
    print(f"Stage-2  n={s2['n_trades']}  win={s2['win_rate']*100:.1f}%  "
          f"mean={s2['mean_bps']:+.1f}bps  t={s2['tstat']:+.2f}  skew={s2['skew']:+.2f}")
    print(f"Random   n={rnd['n_trades']}  win={rnd['win_rate']*100:.1f}%  "
          f"mean={rnd['mean_bps']:+.1f}bps  t={rnd['tstat']:+.2f}")
    print(f"Delta (Stage-2 minus random): {s2['mean_bps'] - rnd['mean_bps']:+.1f} bps")

    print("\n=== Per-stock breakdown (Stage-2 entries, gross) ===")
    bm = tapes["SPY"]
    for tk in STOCKS:
        stock = tapes[tk]
        entries = st.stage2_entries(stock, bm, sma_n=30)
        led = st.run_forward_returns(stock, bm, entries, hold_weeks=HOLD_WEEKS, cost_bps=0)
        s = st.summarize(led, "ret_gross")
        print(f"  {tk:5s}  n={s['n_trades']:3d}  win={s['win_rate']*100:.0f}%  "
              f"mean={s['mean_bps']:+.0f}bps  t={s['tstat']:+.2f}")

    print("\n=== In Stage-2 holding vs out-of-Stage-2 (next-week excess return) ===")
    in_s2, out_s2 = [], []
    for tk in STOCKS:
        stock = tapes[tk]
        sc = stock["close"]
        bc = bm["close"].reindex(sc.index, method="ffill")
        excess = sc.pct_change() - bc.pct_change()
        flag = st.stage2_signal(stock, bm, sma_n=30)
        for i in range(len(flag) - 1):
            if np.isfinite(excess.iloc[i + 1]):
                if flag.iloc[i]:
                    in_s2.append(excess.iloc[i + 1])
                else:
                    out_s2.append(excess.iloc[i + 1])
    s_in = st.summarize(pd.DataFrame({"ret_net": in_s2}))
    s_out = st.summarize(pd.DataFrame({"ret_net": out_s2}))
    print(f"  In Stage-2  n={s_in['n_trades']}  mean={s_in['mean_bps']:+.1f}bps  t={s_in['tstat']:+.2f}")
    print(f"  Out Stage-2 n={s_out['n_trades']}  mean={s_out['mean_bps']:+.1f}bps  t={s_out['tstat']:+.2f}")

    print("\n=== Cost sweep (Stage-2 entries, net) ===")
    for c in (0.0, 1.0, 5.0, 10.0):
        led = _pool(tapes, cost=c, seed=None)
        s = st.summarize(led, "ret_net")
        print(f"  cost={c:4.1f}bps  net mean={s['mean_bps']:+.1f}bps  t={s['tstat']:+.2f}")

    print("\n=== Different hold periods (Stage-2 gross) ===")
    for hw in (4, 8, 13, 26):
        frames, frames_r = [], []
        for tk in STOCKS:
            stock = tapes[tk]
            entries = st.stage2_entries(stock, bm, sma_n=30)
            if entries.empty:
                continue
            led = st.run_forward_returns(stock, bm, entries, hold_weeks=hw, cost_bps=0)
            rnd_e = st.random_entries(stock, n_entries=len(entries), seed=137)
            led_r = st.run_forward_returns(stock, bm, rnd_e, hold_weeks=hw, cost_bps=0)
            frames.append(led)
            frames_r.append(led_r)
        p = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        pr = pd.concat(frames_r, ignore_index=True) if frames_r else pd.DataFrame()
        ss = st.summarize(p, "ret_gross")
        sr = st.summarize(pr, "ret_gross")
        print(f"  hold={hw:2d}w: s2={ss['mean_bps']:+.0f}bps(t={ss['tstat']:+.2f})  "
              f"rnd={sr['mean_bps']:+.0f}bps(t={sr['tstat']:+.2f})  "
              f"delta={ss['mean_bps']-sr['mean_bps']:+.0f}bps")

    print("\n=== Data fingerprints ===")
    for t in TICKERS:
        b = tapes[t]
        print(f"  {t:6s}  {b.index[0].date()}..{b.index[-1].date()}  fp={data.fingerprint(b)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
