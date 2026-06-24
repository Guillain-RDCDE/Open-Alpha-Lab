"""Real-tape verification — Study 404 (Shooting Star). Regenerates docs/results.md numbers.

Loads (cache-first) the 26-name basket + SPY, detects the shooting-star geometry, splits by
prior trend, trades the star short, and reports the per-horizon edge vs the short base rate,
the pooled-geometry "busted" number, the strong-rally filter sweep, the cost landscape, and
the synthetic positive control. Network is touched only with --fetch.

    python studies/404-shooting-star/examples/verify.py            # cache-only
    python studies/404-shooting-star/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shooting_star import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def main(fetch: bool) -> None:
    panel = data.load_real(cache_dir=CACHE, fetch=fetch)
    if fetch:
        for t, b in panel.items():
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} "
                  f"n={len(b)} fp={data.fingerprint({t: b})}")
        print()
    print("basket fingerprint:", data.fingerprint(panel))
    last = max(b.index[-1] for b in panel.values()).date()
    print("as-of (latest complete bar):", last)
    print(f"basket: {len(panel)} names; star-shaped bars: "
          f"{sum(int(st.is_star_shape(b).sum()) for b in panel.values())}\n")

    print("=== SHOOTING STAR (prior uptrend), traded SHORT, gross edge vs short base ===")
    res = st.run_experiment(panel, side="star", n_draws=5000, cost_bps=5.0, borrow_bps=2.0)
    for h, v in res.items():
        print(f"H={h:2d} n={v['n']:5d} cond={v['cond_mean']*1e4:+7.1f} "
              f"base={v['base_mean']*1e4:+7.1f} edge={v['edge_mean']*1e4:+6.1f}bps "
              f"win={v['win']*100:.1f}% t={v['t']:+.2f} p={v['p_placebo']:.3f} "
              f"net={v['net_edge']*1e4:+.1f}bps")

    print("\n=== ANY star geometry, traded SHORT — the 'calls the top?' axis ===")
    anyr = st.run_experiment(panel, side="any", horizons=(1, 3, 5), n_draws=5000)
    for h, v in anyr.items():
        print(f"H={h:2d} n={v['n']:5d} edge={v['edge_mean']*1e4:+6.1f}bps "
              f"win={v['win']*100:.1f}% t={v['t']:+.2f} p={v['p_placebo']:.3f}")

    print("\n=== Myth-check: strong-rally filter (star, short, H=5) ===")
    for ms in (0.0, 0.05, 0.10, 0.15):
        cr = st.conditional_returns(panel, 5, side="star", min_strength=ms)
        print(f"min_runup={ms:.2f}  n={cr['n']:5d} edge={cr['edge_mean']*1e4:+6.1f}bps "
              f"win={cr['win']*100:.1f}% t={st.hac_t(cr['edge']):+.2f}")

    print("\n=== Cost sweep (star short, H=5; borrow 2bps/day x 5 = 10bps) ===")
    cr5 = st.conditional_returns(panel, 5, side="star")
    for c in (0.0, 1.0, 5.0, 10.0):
        net = st.net_of_costs(cr5["edge_mean"], cost_bps=c, borrow_bps=2.0 * 5)
        print(f"cost={c:4.1f}bps  net={net*1e4:+.1f}bps")

    print("\n=== Per-name star short edge (H=5) — count of t>2 ===")
    npos = 0
    for tk, bars in panel.items():
        cr = st.conditional_returns({tk: bars}, 5, side="star")
        t = st.hac_t(cr["edge"])
        if t > 2:
            npos += 1
        print(f"{tk:5s} n={cr['n']:4d} edge={cr['edge_mean']*1e4:+7.1f}bps t={t:+.2f}")
    print(f"names with HAC t > 2: {npos} of {len(panel)} (chance ~1 at 5%)")

    print("\n=== Synthetic positive control (H=1, planted horizon) ===")
    for ed in (0.0, 0.01):
        p, _ = data.synthetic_panel(edge=ed, seed=404)
        v = st.run_experiment(p, side="star", horizons=(1,), n_draws=1000)[1]
        tag = "null" if ed == 0 else "planted decline"
        print(f"{tag:16s} edge={v['edge_mean']*1e4:+7.1f}bps t={v['t']:+.2f} "
              f"p={v['p_placebo']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
