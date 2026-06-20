"""Real-tape verification — Study 305 (Gold-Oil-Ratio). Regenerates docs/results.md numbers.

Reads (or with --fetch refreshes) GLD/USO/SPY/^IRX daily bars, builds the gold/oil
risk-on/risk-off timing switch, and races it — on an excess-of-cash basis — against
buy-and-hold SPY and a random-timing control. Network is touched only with --fetch.

    python studies/305-gold-oil-ratio/examples/verify.py            # cache-only
    python studies/305-gold-oil-ratio/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gold_oil_ratio import data, strategy as st  # noqa: E402


def _drop_partial(df: pd.DataFrame) -> pd.DataFrame:
    cap = df.index.max().replace(day=1) - pd.Timedelta(days=1)
    return df[df.index <= cap]


def _show(tag, s):
    print(
        f"{tag:18s} ann={s['ann_return']*100:+6.2f}%  sharpe_ex={s['sharpe_excess']:+.3f}  "
        f"maxDD={s['max_dd']*100:6.1f}%  long={s['pct_long']*100:5.1f}%  t_ex={s['t_excess']:+.2f}"
    )


def main(fetch: bool) -> None:
    if fetch:
        for t in data.TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s}  {b.index[0].date()}..{b.index[-1].date()}  n={len(b)}")
        print()

    df = _drop_partial(data.load_real(fetch=False))
    print(f"Tape: {df.index[0].date()} → {df.index[-1].date()}  n={len(df)}")
    print(f"Fingerprints  gold={data.fingerprint(df,'gold')}  oil={data.fingerprint(df,'oil')}  "
          f"equity={data.fingerprint(df,'equity')}\n")

    r = st.race(df, lookback=60, enter_z=1.0, cost_bps=5.0)
    print("=== Canonical switch (lookback 60, z>+1, 5 bps, net) ===")
    _show("Gold/oil switch", r["switch"])
    _show("Buy-and-hold SPY", r["buy_hold"])
    _show("Random-timing", r["random"])
    print(f"\nt(switch − B&H)     = {r['t_vs_bh']:+.2f}   CI {tuple(round(x*1e4,2) for x in r['ci_vs_bh'])} bps/day")
    print(f"t(switch − random)  = {r['t_vs_random']:+.2f}   CI {tuple(round(x*1e4,2) for x in r['ci_vs_random'])} bps/day")
    print(f"switches: {r['n_switches']} over {len(df)/252:.1f}y")

    # Seed stability of the random-control t (the headline fragility check).
    ts = [st.race(df, 60, 1.0, 5.0, rand_seed=s)["t_vs_random"] for s in range(20)]
    print(f"\nt(switch − random) over 20 seeds: mean={np.mean(ts):+.2f}  "
          f"min={min(ts):+.2f}  max={max(ts):+.2f}")

    print("\n=== Robustness grid (t_vs_bh / t_vs_random) ===")
    for lb in (40, 60, 120, 200):
        cells = []
        for ez in (1.0, 1.5, 2.0):
            rr = st.race(df, lb, ez, 5.0)
            cells.append(f"{rr['t_vs_bh']:+.2f}/{rr['t_vs_random']:+.2f}")
        print(f"  lb={lb:3d}: " + "   ".join(cells))

    print("\n=== Cost sweep (canonical) ===")
    for c in (0, 2, 5, 10, 20):
        rr = st.race(df, 60, 1.0, c)
        print(f"  {c:2d}bps  ann={rr['switch']['ann_return']*100:+.2f}%  "
              f"sharpe_ex={rr['switch']['sharpe_excess']:+.3f}  t_vs_bh={rr['t_vs_bh']:+.2f}")

    print("\n=== Pre/post-2015 split (canonical) ===")
    for lo, hi, lab in (("2006", "2015", "PRE-2015"), ("2015", "2027", "POST-2015")):
        sub = df[(df.index.year >= int(lo)) & (df.index.year < int(hi))]
        rr = st.race(sub, 60, 1.0, 5.0)
        print(f"  {lab:10s} sw_ann={rr['switch']['ann_return']*100:+.1f}%  "
              f"bh_ann={rr['buy_hold']['ann_return']*100:+.1f}%  "
              f"t_vs_bh={rr['t_vs_bh']:+.2f}  t_vs_rand={rr['t_vs_random']:+.2f}")

    print("\n=== Synthetic positive control ===")
    for pr in (0.0, -0.005, -0.01):
        d, _ = data.synthetic_daily(n_days=4000, pred_r=pr, seed=305)
        rr = st.race(d, 60, 1.0, 5.0)
        print(f"  pred_r={pr:+.3f}  sw_sharpe={rr['switch']['sharpe_excess']:+.2f}  "
              f"bh_sharpe={rr['buy_hold']['sharpe_excess']:+.2f}  t_vs_bh={rr['t_vs_bh']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
