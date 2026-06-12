"""Real-tape verification — Study 73 (First-Light). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the four 5-minute tapes, runs the 5-minute and 15-minute
ORB against a random-direction control, sweeps costs, and shows the TQQQ leverage drag.
Network is touched only with --fetch.

    python studies/73-first-light/examples/verify.py            # cache-only
    python studies/73-first-light/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from first_light import data, strategy as st  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "TQQQ"]


def _pool(fetch: bool, range_bars: int, sl: float, tp, cost: float,
          seed: int | None = None, financing: float = 0.0):
    frames = []
    for t in TICKERS:
        bars = data.fetch_5m(t, fetch=fetch)
        ent = st.orb_entries(bars, range_bars=range_bars)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, sl_multiplier=sl, tp_multiplier=tp,
                                    cost_bps=cost, directions=dirs,
                                    financing_bps_per_day=financing))
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_5m(t, fetch=True)
            print(f"{t:6s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== 5-min ORB — honest exit-at-close (no TP), SL = 1x range, gross ===")
    cross = st.summarize(_pool(False, 1, 1.0, None, 0), "ret_gross")
    rand  = st.summarize(_pool(False, 1, 1.0, None, 0, seed=73), "ret_gross")
    print(f"CROSS  n={cross['n_trades']} win={cross['win_rate']:.3f} "
          f"mean={cross['mean_bps']:+.2f}bps t={cross['tstat']:+.2f}")
    print(f"RANDOM n={rand['n_trades']} win={rand['win_rate']:.3f} "
          f"mean={rand['mean_bps']:+.2f}bps t={rand['tstat']:+.2f}")
    print(f"delta = {cross['mean_bps'] - rand['mean_bps']:+.2f} bps")

    print("\n=== 15-min ORB (range_bars=3) ===")
    c15 = st.summarize(_pool(False, 3, 1.0, None, 0), "ret_gross")
    r15 = st.summarize(_pool(False, 3, 1.0, None, 0, seed=73), "ret_gross")
    print(f"CROSS  n={c15['n_trades']} win={c15['win_rate']:.3f} "
          f"mean={c15['mean_bps']:+.2f}bps t={c15['tstat']:+.2f}")
    print(f"RANDOM n={r15['n_trades']} win={r15['win_rate']:.3f} "
          f"mean={r15['mean_bps']:+.2f}bps t={r15['tstat']:+.2f}")

    print("\n=== Per-ticker breakdown (5-min ORB, gross) ===")
    for t in TICKERS:
        b = data.fetch_5m(t, fetch=False)
        ent = st.orb_entries(b, range_bars=1)
        s = st.summarize(
            st.run_trades(b, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=0),
            "ret_gross",
        )
        print(f"{t:6s} n={s['n_trades']:4d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Cost sweep (5-min ORB, net) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(False, 1, 1.0, None, c), "ret_net")
        print(f"cost={c:4.1f}bps net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== TQQQ leverage drag (5-min ORB, 1bp cost) ===")
    b = data.fetch_5m("TQQQ", fetch=False)
    ent_tqqq = st.orb_entries(b, range_bars=1)
    for fin in (0, 25, 50):
        led = st.run_trades(b, ent_tqqq, sl_multiplier=1.0, tp_multiplier=None,
                            cost_bps=1.0, financing_bps_per_day=fin)
        s = st.summarize(led, "ret_net")
        print(f"fin={fin:2d}bps/d net={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f} win={s['win_rate']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
