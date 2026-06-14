"""Real-tape verification — Study 129 (Heikin-Ashi). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the four daily tapes, runs the HA colour-flip signal
with symmetric ±1 ATR barriers against a random-direction control AND the raw-candle
equivalent, sweeps costs, and confirms the smoothing adds no directional signal.
Network is touched only with --fetch.

    python studies/129-heikin-ashi/examples/verify.py            # cache-only
    python studies/129-heikin-ashi/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from heikin_ashi import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL"]


def _pool(
    fetch: bool,
    tp: float,
    sl: float,
    cost: float,
    seed: int | None = None,
    use_ha: bool = True,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    if cache_dir is None:
        cache_dir = data.DEFAULT_CACHE
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch, cache_dir=cache_dir)
        ha = st.heikin_ashi(bars)
        col = st.ha_colour(ha) if use_ha else st.raw_colour(bars)
        ent = st.colour_flip_entries(col, warmup=20)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    cache_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "_cache")
    )

    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True, cache_dir=cache_dir)
            print(
                f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                f"fp={data.fingerprint(b)}"
            )
        print()

    cross = st.summarize(_pool(False, 1, 1, 0, cache_dir=cache_dir), "ret_gross")
    rand = st.summarize(_pool(False, 1, 1, 0, seed=129, cache_dir=cache_dir), "ret_gross")
    raw = st.summarize(_pool(False, 1, 1, 0, use_ha=False, cache_dir=cache_dir), "ret_gross")

    print("=== honest symmetric ±1 ATR (gross) ===")
    print(
        f"HA-CROSS n={cross['n_trades']} win={cross['win_rate']:.3f} "
        f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}"
    )
    print(
        f"RANDOM   n={rand['n_trades']}  win={rand['win_rate']:.3f} "
        f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}"
    )
    print(
        f"RAW-CANDLE n={raw['n_trades']} win={raw['win_rate']:.3f} "
        f"mean={raw['mean_bps']:+.2f}bps t={raw['tstat']:+.2f}"
    )
    print(f"Delta HA - Random = {cross['mean_bps'] - rand['mean_bps']:+.2f} bps")

    print("\n=== cost sweep (HA, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, 1, 1, c, cache_dir=cache_dir), "ret_net")
        print(f"cost={c:4.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== per-instrument breakdown ===")
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False, cache_dir=cache_dir)
        ha = st.heikin_ashi(b)
        col = st.ha_colour(ha)
        ent = st.colour_flip_entries(col, warmup=20)
        led = st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0)
        s = st.summarize(led, "ret_gross")
        ha_flips = col.ne(col.shift(1)).sum()
        raw_col = st.raw_colour(b)
        raw_flips = raw_col.ne(raw_col.shift(1)).sum()
        print(
            f"{t:5s} n={s['n_trades']:4d} win={s['win_rate']:.3f} "
            f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f} "
            f"| HA_flips={ha_flips} raw_flips={raw_flips}"
        )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
