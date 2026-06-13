"""Real-tape verification — Study 107 (Stochastic-Oscillator). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the six daily tapes, runs the %K/%D oversold/overbought
crossover and the crossover-only variant, pins each against a random-direction control,
sweeps hold periods and costs, and shows the zone-filter vs no-filter comparison.
Network is touched only with --fetch.

    python studies/107-stochastic-oscillator/examples/verify.py            # cache-only
    python studies/107-stochastic-oscillator/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stochastic_oscillator import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]


def _pool(fetch: bool, hold: int, cost: float, zone: bool, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        osc = st.stochastic(bars)
        sigs = st.crossover_signals(osc["pct_k"], osc["pct_d"], zone_filter=zone)
        dirs = st.random_directions(len(sigs), seed=seed) if seed is not None else None
        frames.append(
            st.run_trades(bars, sigs, hold_days=hold, cost_bps=cost, directions=dirs)
        )
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== zone-filtered crossover (5d hold, gross) vs random control ===")
    cross = st.summarize(_pool(False, 5, 0, True), "ret_gross")
    rand = st.summarize(_pool(False, 5, 0, True, seed=107), "ret_gross")
    print(
        f"ZONE-CROSS  n={cross['n_trades']} win={cross['win_rate']:.3f} "
        f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}"
    )
    print(
        f"RANDOM      n={rand['n_trades']} win={rand['win_rate']:.3f} "
        f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}"
    )
    print(f"delta cross-rand: {cross['mean_bps'] - rand['mean_bps']:+.2f} bps")

    print("\n=== per-instrument breakdown (zone-filtered, 5d hold, gross) ===")
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        osc = st.stochastic(b)
        sigs = st.crossover_signals(osc["pct_k"], osc["pct_d"], zone_filter=True)
        s = st.summarize(st.run_trades(b, sigs, hold_days=5, cost_bps=0), "ret_gross")
        print(f"{t:5s} n={s['n_trades']:4d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== hold-period sweep (zone-filtered, gross) ===")
    for hold in (1, 3, 5, 10):
        p = _pool(False, hold, 0, True)
        s = st.summarize(p, "ret_gross")
        print(f"hold={hold:2d}d: n={s['n_trades']} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== cost sweep (zone-filtered, 5d hold, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0):
        s = st.summarize(_pool(False, 5, c, True), "ret_net")
        print(f"cost={c:.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== crossover-only (no zone filter, 5d hold, gross) ===")
    p_nz = _pool(False, 5, 0, False)
    s_nz = st.summarize(p_nz, "ret_gross")
    print(
        f"NOZONE-CROSS n={s_nz['n_trades']} win={s_nz['win_rate']:.3f} "
        f"mean={s_nz['mean_bps']:+.2f}bps t={s_nz['tstat']:+.2f}"
    )


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
