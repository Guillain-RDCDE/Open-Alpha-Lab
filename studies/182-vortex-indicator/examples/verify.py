"""Real-tape verification — Study 182 (Vortex-Indicator). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the three daily tapes (SPY, QQQ, IWM), computes the
Vortex Indicator VI+(14)/VI-(14) crossover signal against a random-direction control,
sweeps hold periods, and applies a Bonferroni multiple-comparisons correction.
Network is touched only with --fetch.

    python studies/182-vortex-indicator/examples/verify.py            # cache-only
    python studies/182-vortex-indicator/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vortex_indicator import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM"]


def _pool(fetch: bool, hold_days: int, cost_bps: float, seed: int | None = None):
    frames = []
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=fetch)
        vi = st.vortex(bars, period=14)
        ent = st.crossover_entries(vi)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=hold_days,
                                    cost_bps=cost_bps, directions=dirs))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== Per-instrument breakdown (hold=14d, cost=0, gross) ===")
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        vi = st.vortex(bars, period=14)
        ent = st.crossover_entries(vi)
        led = st.run_trades(bars, ent, hold_days=14, cost_bps=0)
        s = st.summarize(led, "ret_gross")
        n_years = (bars.index[-1] - bars.index[0]).days / 365.25
        print(f"  {t:5s} n={s['n_trades']:3d} ({s['n_trades']/n_years:.1f}/yr) "
              f"win={s['win_rate']:.3f} mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print()
    print("=== Pooled: crossover vs random direction (hold=14d, cost=0) ===")
    cross = st.summarize(_pool(False, 14, 0), "ret_gross")
    rand = st.summarize(_pool(False, 14, 0, seed=182), "ret_gross")
    print(f"CROSS  n={cross['n_trades']} win={cross['win_rate']:.3f} "
          f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}")
    print(f"RANDOM n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")

    print()
    print("=== Hold-period sweep — gross, pooled (Bonferroni k=5, threshold |t|=2.58) ===")
    best_t, best_hd = 0.0, 14
    for hd in (5, 10, 14, 20, 30):
        s = st.summarize(_pool(False, hd, 0), "ret_gross")
        flag = " <<< BEST" if abs(s["tstat"]) > best_t else ""
        if abs(s["tstat"]) > best_t:
            best_t = abs(s["tstat"])
            best_hd = hd
        print(f"  hold={hd:2d}d  n={s['n_trades']}  mean={s['mean_bps']:+.2f}bps  "
              f"t={s['tstat']:+.2f}{flag}")
    print(f"  Best: hold={best_hd}d |t|={best_t:.2f}  (Bonferroni threshold 2.58 -> "
          f"{'PASSES' if best_t >= 2.58 else 'FAILS'})")

    print()
    print(f"=== Cost sweep at hold={best_hd}d ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, best_hd, c), "ret_net")
        print(f"  cost={c:.1f}bps  net mean={s['mean_bps']:+.2f}bps  t={s['tstat']:+.2f}")

    print()
    print("=== Coin comparison at multiple seeds (hold=14d, gross) ===")
    rand_means = [
        st.summarize(_pool(False, 14, 0, seed=s), "ret_gross")["mean_bps"]
        for s in range(20)
    ]
    cross_mean = st.summarize(_pool(False, 14, 0), "ret_gross")["mean_bps"]
    pct_below = np.mean([r < cross_mean for r in rand_means]) * 100
    print(f"  Cross: {cross_mean:+.2f}bps")
    print(f"  Coin range (20 seeds): [{min(rand_means):+.2f}, {max(rand_means):+.2f}]")
    print(f"  % of coin seeds below cross: {pct_below:.0f}%")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
