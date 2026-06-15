"""Real-tape verification — Study 185 (Chande-Momentum). Regenerates docs/results.md numbers.

Fetches (or reads from cache) daily tapes for a basket of liquid instruments, runs the CMO
overbought/oversold and zero-cross framings against a random-direction control, sweeps hold
periods and costs, and prints the numbers that populate docs/results.md.

Network is touched only with --fetch.

    python studies/185-chande-momentum/examples/verify.py            # cache-only
    python studies/185-chande-momentum/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chande_momentum import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
CMO_PERIOD = 14
THRESHOLD = 50.0
HOLD_DAYS = 5


def _pool_ob_os(fetch: bool, hold: int, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        c = st.cmo(bars["close"], period=CMO_PERIOD)
        ent = st.ob_os_entries(c, threshold=THRESHOLD)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=hold, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def _pool_zc(fetch: bool, hold: int, cost: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        c = st.cmo(bars["close"], period=CMO_PERIOD)
        ent = st.zero_cross_entries(c)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=hold, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        print("=== Fetching / refreshing tapes ===")
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"  {t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint(b)}")
        print()

    print("=== OB/OS framing — CMO(14) |CMO| > 50, 5-day hold, gross ===")
    sig = st.summarize(_pool_ob_os(fetch=False, hold=HOLD_DAYS, cost=0.0), "ret_gross")
    rnd = st.summarize(_pool_ob_os(fetch=False, hold=HOLD_DAYS, cost=0.0, seed=185), "ret_gross")
    print(f"  SIGNAL  n={sig['n_trades']} win={sig['win_rate']:.3f} "
          f"mean={sig['mean_bps']:+.2f}bps t={sig['tstat']:+.2f}")
    print(f"  RANDOM  n={rnd['n_trades']} win={rnd['win_rate']:.3f} "
          f"mean={rnd['mean_bps']:+.2f}bps t={rnd['tstat']:+.2f}")
    print(f"  delta = {sig['mean_bps'] - rnd['mean_bps']:+.2f} bps/trade")

    print("\n=== Zero-cross framing — CMO(14) crosses zero, 5-day hold, gross ===")
    sigz = st.summarize(_pool_zc(fetch=False, hold=HOLD_DAYS, cost=0.0), "ret_gross")
    rndz = st.summarize(_pool_zc(fetch=False, hold=HOLD_DAYS, cost=0.0, seed=185), "ret_gross")
    print(f"  SIGNAL  n={sigz['n_trades']} win={sigz['win_rate']:.3f} "
          f"mean={sigz['mean_bps']:+.2f}bps t={sigz['tstat']:+.2f}")
    print(f"  RANDOM  n={rndz['n_trades']} win={rndz['win_rate']:.3f} "
          f"mean={rndz['mean_bps']:+.2f}bps t={rndz['tstat']:+.2f}")
    print(f"  delta = {sigz['mean_bps'] - rndz['mean_bps']:+.2f} bps/trade")

    print("\n=== OB/OS hold-period sweep (gross, vs random control) ===")
    for hold in (1, 3, 5, 10, 20):
        s = st.summarize(_pool_ob_os(fetch=False, hold=hold, cost=0.0), "ret_gross")
        r = st.summarize(_pool_ob_os(fetch=False, hold=hold, cost=0.0, seed=185), "ret_gross")
        print(f"  hold={hold:2d}d  signal={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f} "
              f"| coin={r['mean_bps']:+.2f}bps t={r['tstat']:+.2f}")

    print("\n=== Zero-cross hold-period sweep (gross, vs random control) ===")
    for hold in (1, 3, 5, 10, 20):
        s = st.summarize(_pool_zc(fetch=False, hold=hold, cost=0.0), "ret_gross")
        r = st.summarize(_pool_zc(fetch=False, hold=hold, cost=0.0, seed=185), "ret_gross")
        print(f"  hold={hold:2d}d  signal={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f} "
              f"| coin={r['mean_bps']:+.2f}bps t={r['tstat']:+.2f}")

    print("\n=== OB/OS cost sweep (hold 5d, net) ===")
    for cost in (0.0, 0.5, 1.0, 2.0):
        s = st.summarize(_pool_ob_os(fetch=False, hold=HOLD_DAYS, cost=cost), "ret_net")
        print(f"  cost={cost:.1f}bps  net={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Per-instrument OB/OS (5d hold, gross) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        c = st.cmo(bars["close"], period=CMO_PERIOD)
        ent = st.ob_os_entries(c, threshold=THRESHOLD)
        s = st.summarize(st.run_trades(bars, ent, hold_days=HOLD_DAYS, cost_bps=0), "ret_gross")
        r = st.summarize(
            st.run_trades(bars, ent, hold_days=HOLD_DAYS, cost_bps=0,
                          directions=st.random_directions(len(ent), seed=185)),
            "ret_gross",
        )
        print(f"  {t:5s}  n={s['n_trades']:4d} sig={s['mean_bps']:+.2f} t={s['tstat']:+.2f} "
              f"| coin={r['mean_bps']:+.2f} t={r['tstat']:+.2f} | fp={data.fingerprint(bars)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
