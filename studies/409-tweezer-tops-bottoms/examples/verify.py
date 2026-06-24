"""Real-tape verification — Study 409 (Tweezers). Regenerates docs/results.md numbers.

Loads (cache-first) the 31-name basket, detects tweezer tops & bottoms, and reproduces:
the base-rate confound table, the excess-over-base-rate horizon grid, the date-shuffle
placebos, the trend-filter myth-check, the cost sweep, and the synthetic positive control.
Network is touched only with --fetch (and only on a cache miss).

    python studies/409-tweezer-tops-bottoms/examples/verify.py            # cache-only
    python studies/409-tweezer-tops-bottoms/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tweezer_tops_bottoms import data, strategy as st  # noqa: E402
from tweezer_tops_bottoms.strategy import _hac_t  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def excess(panel, h, kind, require_trend=True):
    e = []
    for _, bars in panel.items():
        ev = st.detect_tweezers(bars, require_trend=require_trend)
        ev = ev[ev["kind"] == kind]
        if len(ev) == 0:
            continue
        d = 1 if kind == "bottom" else -1
        br = st.base_rate(bars, hold_days=h, direction=d)["mean_bps"] / 1e4
        led = st.forward_returns(bars, ev, hold_days=h)
        e.extend((led["ret_gross"] - br).tolist())
    r = np.array(e)
    return dict(n=r.size, mean_bps=r.mean() * 1e4, t=_hac_t(r), win=(r > 0).mean())


def signed_date_placebo(panel, n_bot, n_top, hold=5, n_iter=5000, seed=409):
    rng = np.random.default_rng(seed)
    tapes = [(b["open"].to_numpy(float), b["close"].to_numpy(float))
             for b in panel.values()]
    n_tot = n_bot + n_top
    dir_mix = np.array([1] * n_bot + [-1] * n_top)
    means = np.empty(n_iter)
    for it in range(n_iter):
        dm = rng.permutation(dir_mix)
        rr = np.empty(n_tot)
        for j in range(n_tot):
            o, c = tapes[rng.integers(len(tapes))]
            e = rng.integers(1, len(o) - 1)
            xi = min(e + hold - 1, len(o) - 1)
            rr[j] = dm[j] * (c[xi] - o[e]) / o[e]
        means[it] = rr.mean()
    return means * 1e4


def long_date_placebo(panel, n, hold=5, n_iter=5000, seed=77):
    rng = np.random.default_rng(seed)
    tapes = [(b["open"].to_numpy(float), b["close"].to_numpy(float))
             for b in panel.values()]
    means = np.empty(n_iter)
    for it in range(n_iter):
        rr = np.empty(n)
        for j in range(n):
            o, c = tapes[rng.integers(len(tapes))]
            e = rng.integers(1, len(o) - 1)
            xi = min(e + hold - 1, len(o) - 1)
            rr[j] = (c[xi] - o[e]) / o[e]
        means[it] = rr.mean()
    return means * 1e4


def main(fetch: bool) -> None:
    if fetch:
        for t in data.BASKET:
            b = data.load_real(t, fetch=True, cache_dir=CACHE)
            print(f"{t:6s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    panel = data.load_basket(fetch=False, cache_dir=CACHE)
    print(f"panel: {len(panel)} names  fingerprint={data.panel_fingerprint(panel)}")
    asof = max(b.index[-1] for b in panel.values()).date()
    start = min(b.index[0] for b in panel.values()).date()
    print(f"window: {start} -> {asof}\n")

    print("=== Base-rate confound (raw, gross, hold=5) ===")
    _, bs = st.run_experiment(panel, hold_days=5, kind="bottom")
    _, ts = st.run_experiment(panel, hold_days=5, kind="top")
    print(f"bottom n={bs['n_trades']} win={bs['win_rate']:.3f} mean={bs['mean_bps']:+.2f}bps "
          f"t={bs['tstat']:+.2f} base_long={bs['base_long_bps']:+.2f}")
    print(f"top    n={ts['n_trades']} win={ts['win_rate']:.3f} mean={ts['mean_bps']:+.2f}bps "
          f"t={ts['tstat']:+.2f} base_short={ts['base_short_bps']:+.2f}")

    print("\n=== Excess over base rate (gross), by horizon ===")
    for kind in ("bottom", "top"):
        for h in (1, 3, 5, 10):
            x = excess(panel, h, kind)
            print(f"{kind:6s} h={h:2d} n={x['n']} excess={x['mean_bps']:+.2f}bps "
                  f"t={x['t']:+.2f} win={x['win']:.3f}")

    print("\n=== Date-shuffle placebos (hold=5) ===")
    n_bot = sum((st.detect_tweezers(b)["kind"] == "bottom").sum() for b in panel.values())
    n_top = sum((st.detect_tweezers(b)["kind"] == "top").sum() for b in panel.values())
    _, sc = st.run_experiment(panel, hold_days=5)
    plc = signed_date_placebo(panel, n_bot, n_top, 5, 5000)
    print(f"signed real={sc['mean_bps']:+.2f}bps t={sc['tstat']:+.2f}; "
          f"placebo mean={plc.mean():+.2f} std={plc.std():.2f} "
          f"p(>=real)={(plc >= sc['mean_bps']).mean():.3f}")
    plb = long_date_placebo(panel, n_bot, 5, 5000)
    print(f"bottom raw={bs['mean_bps']:+.2f}bps; random-LONG placebo mean={plb.mean():+.2f} "
          f"std={plb.std():.2f} p(>=real)={(plb >= bs['mean_bps']).mean():.3f}")

    print("\n=== Myth-check: trend filter on/off (excess, hold=5) ===")
    for kind in ("bottom", "top"):
        for rt in (True, False):
            x = excess(panel, 5, kind, require_trend=rt)
            tag = "trend" if rt else "nofilt"
            print(f"{kind:6s} {tag:6s} n={x['n']} excess={x['mean_bps']:+.2f}bps t={x['t']:+.2f}")

    print("\n=== Cost sweep (signed market-neutral, hold=5, borrow 50bps shorts) ===")
    for c in (0.0, 1.0, 2.0, 5.0):
        _, s = st.run_experiment(panel, hold_days=5, cost_bps=c, borrow_bps=50.0)
        print(f"cost={c:.1f}bps net={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== Synthetic positive control (signed, hold=5, 20 names, 5 seeds) ===")
    for edge in (0.0, 0.02):
        ts_list, m_list = [], []
        for seed in (1, 2, 3, 4, 5):
            sp, _ = data.synthetic_panel(n_names=20, n_days=2000, edge=edge, seed=seed)
            _, s = st.run_experiment(sp, hold_days=5)
            ts_list.append(s["tstat"])
            m_list.append(s["mean_bps"])
        print(f"edge={edge:.2f}: mean t={np.mean(ts_list):+.2f} mean bps={np.mean(m_list):+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
