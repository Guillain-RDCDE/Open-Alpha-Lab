"""Real-tape verification — Study 684 (Inverted Hammer). Regenerates docs/results.md numbers.

Loads (cache-first) the 26-name basket + SPY, detects the inverted-hammer geometry (long
upper wick, small body, little lower wick), splits by prior trend, trades the bullish claim
LONG (buy after a downtrend), and reports the per-horizon edge vs the unconditional base
rate, a HAC t + label-shuffle placebo + Bonferroni correction across the four horizons, the
"any-trend" and "wrong-side" (post-uptrend) contrasts, a myth-check filter sweep, the cost
landscape, and the synthetic positive control. Network is touched only with --fetch.

    python studies/684-inverted-hammer/examples/verify.py            # cache-only
    python studies/684-inverted-hammer/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from inverted_hammer import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
ASOF = "2026-06-30"          # pinned as-of: the last COMPLETE calendar month (never in the future)


def _load(fetch: bool) -> dict:
    panel = data.load_real(cache_dir=CACHE, fetch=fetch or not data.have_real(cache_dir=CACHE))
    return {t: b[b.index <= ASOF] for t, b in panel.items()}


def main(fetch: bool) -> None:
    print("# Inverted Hammer — does a long upper wick after a downtrend mark a floor?")
    panel = _load(fetch)
    fp = data.fingerprint(panel)
    span0 = min(b.index[0] for b in panel.values()).date()
    span1 = max(b.index[-1] for b in panel.values()).date()
    tot = sum(len(b) for b in panel.values())
    n_shape = sum(int(st.is_invhammer_shape(b).sum()) for b in panel.values())
    print(f"basket={len(panel)} names  span {span0}..{span1}  bars={tot:,}  "
          f"inverted-hammer-shaped={n_shape:,}  panel fingerprint={fp}  as-of {ASOF}")
    print(data_stamp("SPY OHLC (representative single-name stamp)", panel["SPY"],
                     cols=["open", "high", "low", "close"], asof=ASOF))

    print("\n# THE HEADLINE — INVERTED HAMMER (shape after a downtrend), traded LONG, "
          "gross edge vs unconditional base rate")
    res = st.run_experiment(panel, side="invhammer", n_draws=5000, cost_bps=5.0, seed=684)
    print(" H |     n | cond%  base%  edge%  | win%  | HAC t | placebo p | bonf p | net%")
    for h in st.HORIZONS:
        r = res[h]
        print(f"{h:2d} | {r['n']:5d} | {r['cond_mean']*100:+.3f} {r['base_mean']*100:+.3f} "
              f"{r['edge_mean']*100:+.3f} | {r['win']*100:4.1f} | {r['t']:+5.2f} | "
              f"{r['p_placebo']:8.3f} | {r['p_bonferroni']:6.3f} | {r['net_edge']*100:+.3f}")

    print("\n=== The bearish look-alike — SAME geometry after an UPTREND (the shooting-star "
          "side, sibling study 404), traded long here as a myth-check: does the direction "
          "matter at all? ===")
    star = st.run_experiment(panel, side="star", horizons=(1, 3, 5, 10), n_draws=5000, seed=684)
    for h, v in star.items():
        print(f"H={h:2d} n={v['n']:5d} edge={v['edge_mean']*1e4:+6.1f}bps "
              f"win={v['win']*100:.1f}% t={v['t']:+.2f} p={v['p_placebo']:.3f}")

    print("\n=== ANY inverted-hammer geometry, traded long — ignoring the trend split ===")
    anyr = st.run_experiment(panel, side="any", horizons=(1, 3, 5, 10), n_draws=5000, seed=684)
    for h, v in anyr.items():
        print(f"H={h:2d} n={v['n']:5d} edge={v['edge_mean']*1e4:+6.1f}bps "
              f"win={v['win']*100:.1f}% t={v['t']:+.2f} p={v['p_placebo']:.3f}")

    print("\n=== Myth-check: does a deeper-washout filter rescue the inverted hammer? (H=3) ===")
    base = st.run_experiment(panel, side="invhammer", horizons=(3,), n_draws=3000, seed=684)[3]
    print(f"plain inverted hammer (lookback=10):  edge={base['edge_mean']*100:+.3f}%  "
          f"t={base['t']:+.2f}  p={base['p_placebo']:.3f}  n={base['n']}")
    for lb in (5, 20):
        r = st.run_experiment(panel, side="invhammer", horizons=(3,), lookback=lb,
                              n_draws=3000, seed=684)[3]
        print(f"trend lookback={lb:2d}:                 edge={r['edge_mean']*100:+.3f}%  "
              f"t={r['t']:+.2f}  p={r['p_placebo']:.3f}  n={r['n']}")
    for ms in (0.05, 0.10):
        r = st.run_experiment(panel, side="invhammer", horizons=(3,), min_strength=ms,
                              n_draws=3000, seed=684)[3]
        print(f"min washout={ms:.2f}:                  edge={r['edge_mean']*100:+.3f}%  "
              f"t={r['t']:+.2f}  p={r['p_placebo']:.3f}  n={r['n']}")
    for wm in (3.0, 4.0):
        r = st.run_experiment(panel, side="invhammer", horizons=(3,), n_draws=3000,
                              seed=684, wick_mult=wm)[3]
        print(f"wick_mult={wm}:                   edge={r['edge_mean']*100:+.3f}%  "
              f"t={r['t']:+.2f}  p={r['p_placebo']:.3f}  n={r['n']}")

    print("\n=== Cost sweep (invhammer long, H=3) ===")
    cr3 = st.conditional_returns(panel, 3, side="invhammer")
    for c in (0.0, 1.0, 5.0, 10.0):
        net = st.net_of_costs(cr3["edge_mean"], cost_bps=c)
        print(f"cost={c:4.1f}bps  net={net*1e4:+.1f}bps")

    print("\n=== Per-name inverted-hammer long edge (H=3) — count of |t|>2 ===")
    npos = 0
    for tk, bars in panel.items():
        cr = st.conditional_returns({tk: bars}, 3, side="invhammer")
        t = st.hac_t(cr["edge"])
        if abs(t) > 2:
            npos += 1
        print(f"{tk:5s} n={cr['n']:4d} edge={cr['edge_mean']*1e4:+7.1f}bps t={t:+.2f}")
    print(f"names with |HAC t| > 2: {npos} of {len(panel)} (chance ~1-2 at 5% two-sided)")

    print("\n=== Synthetic positive control (H=1, side=any) ===")
    print("planted | events | cond%  edge%  | HAC t | placebo p | win%")
    for edge in (0.0, 0.005):
        sp, _ = data.synthetic_panel(edge=edge, seed=684)
        r = st.run_experiment(sp, side="any", horizons=(1,), n_draws=2000, seed=684)[1]
        print(f"{edge:6.3f}  | {r['n']:6d} | {r['cond_mean']*100:+.3f} {r['edge_mean']*100:+.3f} "
              f"| {r['t']:+5.2f} | {r['p_placebo']:8.3f} | {r['win']*100:4.1f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
