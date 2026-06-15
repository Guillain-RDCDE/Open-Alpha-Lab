"""Real-tape verification -- Study 179 (Aroon).  Regenerates docs/results.md numbers.

Fetches (or reads from cache) the three daily tapes, runs the Aroon(25) crossover
strategy against a random-direction control, sweeps hold periods and costs, and
reports the per-instrument breakdown.  Network is touched only with --fetch.

    python studies/179-aroon/examples/verify.py            # cache-only
    python studies/179-aroon/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aroon import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM"]
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")


def _pool(hold: int = 5, cost: float = 0.0, seed: int | None = None) -> pd.DataFrame:
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, cache_dir=CACHE)
        ar = st.aroon(bars, period=25)
        ent = st.crossover_entries(ar)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=hold, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, period="15y", fetch=True, cache_dir=CACHE)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} n={len(b)} fp={data.fingerprint(b)}")
        print()

    # Per-instrument breakdown
    print("=== Per-instrument, Aroon(25) crossover, hold=5, gross ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, cache_dir=CACHE)
        ar = st.aroon(bars, period=25)
        ent = st.crossover_entries(ar)
        s = st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross")
        print(f"{t:4s}  n={s['n_trades']:3d}  win={s['win_rate']:.3f}  "
              f"mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    # Pooled crossover vs random
    cross = st.summarize(_pool(5, 0.0), "ret_gross")
    rand  = st.summarize(_pool(5, 0.0, seed=179), "ret_gross")
    print()
    print("=== Pooled Aroon(25) crossover vs random-direction control, hold=5, gross ===")
    print(f"CROSS  n={cross['n_trades']:3d}  win={cross['win_rate']:.3f}  "
          f"mean={cross['mean_bps']:+.2f}bps  t={cross['tstat']:+.2f}  skew={cross['skew']:+.2f}")
    print(f"RANDOM n={rand['n_trades']:3d}  win={rand['win_rate']:.3f}  "
          f"mean={rand['mean_bps']:+.2f}bps  t={rand['tstat']:+.2f}")
    print(f"delta: {cross['mean_bps'] - rand['mean_bps']:+.2f} bps")

    # Hold-period sweep
    print()
    print("=== Hold-period sweep (crossover, gross) ===")
    for hold in (1, 3, 5, 10, 20):
        s = st.summarize(_pool(hold, 0.0), "ret_gross")
        print(f"hold={hold:2d}  mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    # Cost sweep
    print()
    print("=== Cost sweep (crossover, hold=5, net) ===")
    for cost in (0.0, 0.5, 1.0, 2.0):
        s = st.summarize(_pool(5, cost), "ret_net")
        print(f"cost={cost:.1f}bps  mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    # Long vs short breakdown
    print()
    print("=== Long vs short directional breakdown (crossover, hold=5, gross, pooled) ===")
    all_led = _pool(5, 0.0)
    longs  = all_led[all_led["dir"] == 1]
    shorts = all_led[all_led["dir"] == -1]
    sl = st.summarize(longs, "ret_gross")
    ss = st.summarize(shorts, "ret_gross")
    print(f"Longs:  n={sl['n_trades']:3d}  mean={sl['mean_bps']:+.2f}bps  t={sl['tstat']:+.2f}")
    print(f"Shorts: n={ss['n_trades']:3d}  mean={ss['mean_bps']:+.2f}bps  t={ss['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
