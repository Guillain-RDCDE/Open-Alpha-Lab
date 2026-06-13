"""Real-tape verification — Study 116 (Power-Hour). Regenerates docs/results.md numbers.

Fetches (or reads from cache) 1-hour bars for SPY, QQQ, IWM, converts them to the
per-session (morning_ret, last_ret) panel, tests the follow-morning signal against a
random-direction control, and sweeps costs. Network is touched only with --fetch.

    python studies/116-power-hour/examples/verify.py            # cache-only
    python studies/116-power-hour/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from power_hour import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM"]


def _pool(fetch: bool, arm: str, cost: float, seed: int = 0) -> pd.DataFrame:
    frames = []
    for t in TICKERS:
        panel = data.fetch_daily_panel(t, fetch=fetch)
        frames.append(st.build_ledger(panel, arm=arm, cost_bps=cost, seed=seed))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            p = data.fetch_daily_panel(t, fetch=True)
            print(f"{t:4s}  n_sessions={len(p)}  fp={data.fingerprint(p)}")
        print()

    # Correlation diagnostic — per ticker
    print("=== morning -> last-hour Pearson correlation ===")
    for t in TICKERS:
        panel = data.fetch_daily_panel(t, fetch=False)
        c = st.morning_last_corr(panel)
        print(f"{t:4s}  n={c['n']:4d}  corr={c['corr']:+.4f}  t={c['tstat']:+.2f}")

    # Pooled correlation
    all_panels = []
    for t in TICKERS:
        all_panels.append(data.fetch_daily_panel(t, fetch=False))
    pool_panel = pd.concat(all_panels, ignore_index=True)
    pc = st.morning_last_corr(pool_panel)
    print(f"POOL  n={pc['n']:4d}  corr={pc['corr']:+.4f}  t={pc['tstat']:+.2f}")

    # Main test: follow vs random
    print("\n=== follow-morning vs random-direction (gross, pooled) ===")
    follow = st.summarize(_pool(False, "follow", 0.0), "ret_gross")
    rand = st.summarize(_pool(False, "random", 0.0, seed=116), "ret_gross")
    print(f"FOLLOW n={follow['n']} win={follow['win_rate']:.3f} "
          f"mean={follow['mean_bps']:+.2f}bps t={follow['tstat']:+.2f}")
    print(f"RANDOM n={rand['n']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")

    # Fade arm
    fade = st.summarize(_pool(False, "fade", 0.0), "ret_gross")
    print(f"FADE   n={fade['n']} win={fade['win_rate']:.3f} "
          f"mean={fade['mean_bps']:+.2f}bps t={fade['tstat']:+.2f}")

    # Cost sweep
    print("\n=== cost sweep (follow arm, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, "follow", c), "ret_net")
        print(f"cost={c:4.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    # Per-instrument
    print("\n=== per-instrument (follow arm, gross) ===")
    for t in TICKERS:
        panel = data.fetch_daily_panel(t, fetch=False)
        s = st.summarize(st.build_ledger(panel, arm="follow", cost_bps=0), "ret_gross")
        print(f"{t:4s} n={s['n']} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
