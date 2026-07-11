"""Real-tape verification — Study 691 (Homing Pigeon). Regenerates docs/results.md numbers.

Loads (cache-first) the 26-name basket + SPY, detects the homing-pigeon geometry (a
smaller down day fully inside a larger prior down day), splits by prior trend, trades the
bullish claim LONG (buy after a downtrend), and reports the per-horizon edge vs the
unconditional base rate, a HAC t + label-shuffle placebo + Bonferroni correction across
the four horizons, the "any-trend" and "wrong-side" (post-uptrend) contrasts, a
myth-check filter sweep, the cost landscape, and the synthetic positive control. Network
is touched only with --fetch.

    python studies/691-homing-pigeon/examples/verify.py            # cache-only
    python studies/691-homing-pigeon/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from homing_pigeon import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
ASOF = "2026-06-30"          # pinned as-of: the last COMPLETE calendar month (never in the future)


def _load(fetch: bool) -> dict:
    panel = data.load_real(cache_dir=CACHE, fetch=fetch or not data.have_real(cache_dir=CACHE))
    return {t: b[b.index <= ASOF] for t, b in panel.items()}


def main(fetch: bool) -> None:
    print("# Homing Pigeon — does a shrinking down day inside a bigger down day, in a "
          "downtrend, mark a floor?")
    panel = _load(fetch)
    fp = data.fingerprint(panel)
    span0 = min(b.index[0] for b in panel.values()).date()
    span1 = max(b.index[-1] for b in panel.values()).date()
    tot = sum(len(b) for b in panel.values())
    n_shape = sum(int(st.is_homingpigeon_shape(b).sum()) for b in panel.values())
    print(f"basket={len(panel)} names  span {span0}..{span1}  bars={tot:,}  "
          f"homing-pigeon-shaped={n_shape:,}  panel fingerprint={fp}  as-of {ASOF}")
    print(data_stamp("SPY OHLC (representative single-name stamp)", panel["SPY"],
                     cols=["open", "high", "low", "close"], asof=ASOF))

    print("\n# THE HEADLINE — HOMING PIGEON (shape after a downtrend), traded LONG, "
          "gross edge vs unconditional base rate")
    res = st.run_experiment(panel, side="pigeon", n_draws=5000, cost_bps=5.0, seed=691)
    print(" H |     n | cond%  base%  edge%  | win%  | HAC t | placebo p | bonf p | net%")
    for h in st.HORIZONS:
        r = res[h]
        print(f"{h:2d} | {r['n']:5d} | {r['cond_mean']*100:+.3f} {r['base_mean']*100:+.3f} "
              f"{r['edge_mean']*100:+.3f} | {r['win']*100:4.1f} | {r['t']:+5.2f} | "
              f"{r['p_placebo']:8.3f} | {r['p_bonferroni']:6.3f} | {r['net_edge']*100:+.3f}")

    print("\n=== Alpha vs beta — does the SHAPE beat plain 'buy any dip in a downtrend' "
          "(no pattern required)? ===")
    for h in st.HORIZONS:
        cr = st.conditional_returns(panel, h, side="pigeon")
        dtp = st.downtrend_pool(panel, h)
        ex_t = st.welch_t(cr["cond"], dtp)
        print(f"H={h:2d}  pigeon={cr['cond_mean']*100:+.3f}%  any-downtrend-dip="
              f"{dtp.mean()*100:+.3f}%  (n={len(dtp)})  excess Welch t={ex_t:+.2f}")

    print("\n=== The 'wrong side' — SAME geometry after an UPTREND, traded long as a "
          "myth-check: does the trend split discriminate a floor from generic drift? ===")
    wrong = st.run_experiment(panel, side="wrongside", horizons=(1, 3, 5, 10), n_draws=5000,
                              seed=691)
    for h, v in wrong.items():
        print(f"H={h:2d} n={v['n']:5d} edge={v['edge_mean']*1e4:+6.1f}bps "
              f"win={v['win']*100:.1f}% t={v['t']:+.2f} p={v['p_placebo']:.3f}")

    print("\n=== ANY homing-pigeon geometry, traded long — ignoring the trend split ===")
    anyr = st.run_experiment(panel, side="any", horizons=(1, 3, 5, 10), n_draws=5000, seed=691)
    for h, v in anyr.items():
        print(f"H={h:2d} n={v['n']:5d} edge={v['edge_mean']*1e4:+6.1f}bps "
              f"win={v['win']*100:.1f}% t={v['t']:+.2f} p={v['p_placebo']:.3f}")

    print("\n=== Myth-check: does a deeper-washout or shorter-lookback filter rescue the "
          "floor? (H=3) ===")
    base = st.run_experiment(panel, side="pigeon", horizons=(3,), n_draws=3000, seed=691)[3]
    print(f"plain homing pigeon (lookback=10):    edge={base['edge_mean']*100:+.3f}%  "
          f"t={base['t']:+.2f}  p={base['p_placebo']:.3f}  n={base['n']}")
    for lb in (5, 20):
        r = st.run_experiment(panel, side="pigeon", horizons=(3,), lookback=lb,
                              n_draws=3000, seed=691)[3]
        print(f"trend lookback={lb:2d}:                 edge={r['edge_mean']*100:+.3f}%  "
              f"t={r['t']:+.2f}  p={r['p_placebo']:.3f}  n={r['n']}")
    for ms in (0.05, 0.10):
        r = st.run_experiment(panel, side="pigeon", horizons=(3,), min_strength=ms,
                              n_draws=3000, seed=691)[3]
        print(f"min washout={ms:.2f}:                  edge={r['edge_mean']*100:+.3f}%  "
              f"t={r['t']:+.2f}  p={r['p_placebo']:.3f}  n={r['n']}")

    print("\n=== Cost sweep (homing pigeon long, H=3) ===")
    cr3 = st.conditional_returns(panel, 3, side="pigeon")
    for c in (0.0, 1.0, 5.0, 10.0):
        net = st.net_of_costs(cr3["edge_mean"], cost_bps=c)
        print(f"cost={c:4.1f}bps  net={net*1e4:+.1f}bps")

    print("\n=== Event clustering — are the pigeon events a handful of shared crash "
          "weeks, or broad-based? ===")
    ec = st.event_clustering(panel, side="pigeon")
    print(f"n={ec['n']}  distinct ISO weeks with >=1 event={ec['n_weeks']}  "
          f"share of events in the busiest 10 weeks={ec['top10_week_share']*100:.1f}%")

    print("\n=== Per-name homing-pigeon long edge (H=3) — count of |t|>2 ===")
    npos = 0
    nnamed = 0
    for tk, bars in panel.items():
        cr = st.conditional_returns({tk: bars}, 3, side="pigeon")
        if cr["n"] == 0:
            print(f"{tk:5s} n=   0 (no occurrences)")
            continue
        nnamed += 1
        t = st.hac_t(cr["edge"])
        if abs(t) > 2:
            npos += 1
        print(f"{tk:5s} n={cr['n']:4d} edge={cr['edge_mean']*1e4:+7.1f}bps t={t:+.2f}")
    print(f"names with |HAC t| > 2: {npos} of {nnamed} with occurrences "
          f"(chance ~{0.05*nnamed:.1f} at 5% two-sided)")

    print("\n=== Synthetic positive control (H=1, side=any) ===")
    print("planted | events | cond%  edge%  | HAC t | placebo p | win%")
    for edge in (0.0, 0.006):
        sp, truth = data.synthetic_panel(edge=edge, seed=691)
        r = st.run_experiment(sp, side="any", horizons=(1,), n_draws=2000, seed=691)[1]
        print(f"{edge:6.3f}  | {r['n']:6d} | {r['cond_mean']*100:+.3f} {r['edge_mean']*100:+.3f} "
              f"| {r['t']:+5.2f} | {r['p_placebo']:8.3f} | {r['win']*100:4.1f}  "
              f"(planted days: {truth['n_planted_days']})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
