"""Real-tape verification — Study 446 (Wyckoff Method).

Loads (cache-first) the daily basket, runs the volume-confirmed Spring/Upthrust events
across horizons, splits by leg, sweeps the volume filter / ZigZag threshold / costs, and
prints the synthetic positive control — reproducing every number in docs/results.md.
Network is touched only with --fetch (then the parquet cache is populated).

    python studies/446-wyckoff-method/examples/verify.py            # cache-only
    python studies/446-wyckoff-method/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wyckoff_method import data, strategy as st  # noqa: E402


def _pooled_ledger(basket, horizon, cost_bps=0.0, vol_confirm=True, pct=0.05,
                   coin=False, seed=446):
    leds = []
    for _tk, b in basket.items():
        ev = st.find_events(b, pct=pct, vol_confirm=vol_confirm)
        if coin:
            dirs = st.random_directions(len(ev), seed=seed)
            leds.append(st.run_trades(b, ev, horizon, cost_bps=cost_bps, directions=dirs))
        else:
            leds.append(st.run_trades(b, ev, horizon, cost_bps=cost_bps))
    return pd.concat(leds, ignore_index=True) if leds else pd.DataFrame()


def _pooled_placebo(basket, horizon, n_draws=20000, seed=446):
    led = _pooled_ledger(basket, horizon, 0.0, True)
    if len(led) == 0:
        return float("nan"), float("nan")
    obs = float(led["ret_gross"].mean())
    mags = led["ret_gross"].to_numpy() * led["dir"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.array([float((rng.choice([-1, 1], size=len(mags)) * mags).mean())
                      for _ in range(n_draws)])
    return obs, float((draws >= obs).mean())


def main(fetch: bool) -> None:
    basket = data.load_basket(fetch=fetch)
    print("=== data stamp ===")
    for tk, b in basket.items():
        print(f"{tk:5s} {b.index[0].date()}..{b.index[-1].date()} "
              f"n={len(b)} fp={data.fingerprint(b)}")

    print("\n=== combined Spring+Upthrust (vol-confirmed), gross, by horizon ===")
    for h in st.HORIZONS:
        s = st.summarize(_pooled_ledger(basket, h, 0.0, True), "ret_gross")
        coin = st.summarize(_pooled_ledger(basket, h, 0.0, True, coin=True), "ret_gross")
        obs, p = _pooled_placebo(basket, h)
        print(f"h={h:3d} n={s['n_trades']:4d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+8.2f}bps t={s['t']:+.2f} hac_t={s['hac_t']:+.2f} | "
              f"coin mean={coin['mean_bps']:+.2f} t={coin['t']:+.2f} | placebo_p={p:.4f}")

    print("\n=== split by leg (gross, 20-day hold) ===")
    led = _pooled_ledger(basket, 20, 0.0, True)
    for k in ("spring", "upthrust"):
        s = st.summarize(led[led["kind"] == k], "ret_gross")
        print(f"{k:9s} n={s['n_trades']:4d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+8.2f}bps t={s['t']:+.2f} hac_t={s['hac_t']:+.2f}")

    print("\n=== robustness: volume filter on/off (gross, 20-day) ===")
    for vc, lbl in ((True, "vol-confirmed"), (False, "no-vol-filter")):
        s = st.summarize(_pooled_ledger(basket, 20, 0.0, vc), "ret_gross")
        print(f"{lbl:13s} n={s['n_trades']:4d} mean={s['mean_bps']:+8.2f}bps "
              f"t={s['t']:+.2f} hac_t={s['hac_t']:+.2f}")

    print("\n=== robustness: ZigZag threshold (gross, 20-day, vol-confirmed) ===")
    for p in (0.03, 0.05, 0.08):
        s = st.summarize(_pooled_ledger(basket, 20, 0.0, True, pct=p), "ret_gross")
        print(f"pct={p:.2f} n={s['n_trades']:4d} mean={s['mean_bps']:+8.2f}bps "
              f"t={s['t']:+.2f} hac_t={s['hac_t']:+.2f}")

    print("\n=== cost sweep (combined, 20-day hold, net) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        s = st.summarize(_pooled_ledger(basket, 20, c, True), "ret_net")
        print(f"cost={c:4.1f}bps net mean={s['mean_bps']:+8.2f}bps t={s['t']:+.2f}")

    print("\n=== synthetic positive control (planted markup edge, 20-day) ===")
    for e in (0.0, 0.06, 0.12, 0.20):
        sb, truth = data.synthetic_panel(edge=e)
        ev = st.find_events(sb, vol_confirm=True)
        s = st.summarize(st.run_trades(sb, ev, 20, cost_bps=0.0), "ret_gross")
        print(f"edge={e:.2f} n={s['n_trades']:4d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+8.2f}bps t={s['t']:+.2f} hac_t={s['hac_t']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
