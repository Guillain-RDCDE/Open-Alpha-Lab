"""Real-tape verification — Study 127 (Williams-R). Regenerates docs/results.md numbers.

Fetches (or reads from cache) six daily tapes, runs Williams %R(14) zone-entry and
cross-back signals against a random-direction control, sweeps hold periods, and shows
the two-framing comparison. Network is touched only with --fetch.

    python studies/127-williams-r/examples/verify.py            # cache-only
    python studies/127-williams-r/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from williams_r import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def _pool(fetch: bool, hold: int, cost: float, framing: str = "zone",
          seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch, cache_dir=CACHE)
        wr = st.williams_r(bars)
        ent = st.zone_entries(wr) if framing == "zone" else st.cross_entries(wr)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=hold, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:  # warm the cache + stamp fingerprints
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=CACHE)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== Zone-entry (%R enters oversold/overbought) — gross, hold=5 ===")
    zone5 = _pool(False, 5, 0)
    zone5r = _pool(False, 5, 0, seed=127)
    z = st.summarize(zone5, "ret_gross")
    r = st.summarize(zone5r, "ret_gross")
    print(f"ZONE   n={z['n_trades']} win={z['win_rate']:.3f} "
          f"mean={z['mean_bps']:+.2f}bps t={z['tstat']:+.2f}")
    print(f"RANDOM n={r['n_trades']} win={r['win_rate']:.3f} "
          f"mean={r['mean_bps']:+.2f}bps t={r['tstat']:+.2f}")
    print(f"Delta (zone - random): {z['mean_bps'] - r['mean_bps']:+.2f} bps")

    print("\n=== Cross-back (%R exits extreme zone) — gross, hold=5 ===")
    cross5 = _pool(False, 5, 0, framing="cross")
    cross5r = _pool(False, 5, 0, framing="cross", seed=127)
    c = st.summarize(cross5, "ret_gross")
    cr = st.summarize(cross5r, "ret_gross")
    print(f"CROSS  n={c['n_trades']} win={c['win_rate']:.3f} "
          f"mean={c['mean_bps']:+.2f}bps t={c['tstat']:+.2f}")
    print(f"RANDOM n={cr['n_trades']} win={cr['win_rate']:.3f} "
          f"mean={cr['mean_bps']:+.2f}bps t={cr['tstat']:+.2f}")

    print("\n=== Hold-period sweep (zone, gross) ===")
    for hold in (1, 3, 5, 10):
        s = st.summarize(_pool(False, hold, 0), "ret_gross")
        print(f"hold={hold:2d}d  n={s['n_trades']} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Cost sweep (zone, hold=5) ===")
    for c_bps in (0.0, 0.5, 1.0, 2.0):
        s = st.summarize(_pool(False, 5, c_bps), "ret_net")
        print(f"cost={c_bps:.1f}bps  net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Per-instrument (zone, hold=5, gross) ===")
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
        wr = st.williams_r(b)
        ent = st.zone_entries(wr)
        s = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0), "ret_gross")
        print(f"{t:5s}  n={s['n_trades']:3d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
