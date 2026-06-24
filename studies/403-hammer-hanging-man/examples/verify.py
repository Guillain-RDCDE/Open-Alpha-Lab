"""Real-tape verification — Study 403 (Hammer & Hanging Man).

Reproduces every number in ``docs/results.md``: detects the hammer / hanging-man candle
geometry across a fixed basket of ~26 liquid US large-caps + SPY (yfinance daily OHLC,
cache-first), measures forward 1/3/5/10-day returns vs each name's unconditional base rate,
runs a HAC one-sample t and a label-shuffle placebo, charges costs, splits the same shape
by prior trend (hammer vs hanging man), runs a filter myth-check, and prints the synthetic
positive control. Network is touched only on a cache miss (or with --fetch).

    python studies/403-hammer-hanging-man/examples/verify.py            # cache-only
    python studies/403-hammer-hanging-man/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hammer_hanging_man import data, strategy as st  # noqa: E402

ASOF = "2026-06-23"          # pinned as-of (never in the future); drop later bars


def _load(fetch: bool) -> dict:
    panel = data.load_real(fetch=fetch or not data.have_real())
    return {t: b[b.index <= ASOF] for t, b in panel.items()}


def main(fetch: bool) -> None:
    panel = _load(fetch)
    fp = data.fingerprint(panel)
    span0 = min(b.index[0] for b in panel.values()).date()
    span1 = max(b.index[-1] for b in panel.values()).date()
    tot = sum(len(b) for b in panel.values())
    n_shape = sum(int(st.is_hammer_shape(b).sum()) for b in panel.values())
    print(f"basket={len(panel)} names  span {span0}..{span1}  bars={tot:,}  "
          f"hammer-shaped={n_shape:,}  fingerprint={fp}  as-of {ASOF}")

    for side, label in (("hammer", "HAMMER (shape after a downtrend — bullish claim)"),
                        ("hangingman", "HANGING MAN (shape after an uptrend — bearish claim)"),
                        ("any", "ANY (the shape, ignoring trend)")):
        print(f"\n=== {label} ===")
        res = st.run_experiment(panel, side=side, n_draws=3000, seed=403)
        print(" H |     n | cond%  base%  edge%  | win%  | HAC t | placebo p | net%")
        for h in st.HORIZONS:
            r = res[h]
            print(f"{h:2d} | {r['n']:5d} | {r['cond_mean']*100:+.3f} {r['base_mean']*100:+.3f} "
                  f"{r['edge_mean']*100:+.3f} | {r['win']*100:4.1f} | {r['t']:+5.2f} | "
                  f"{r['p_placebo']:8.3f} | {r['net_edge']*100:+.3f}")

    print("\n=== Myth-check: does a trend-strength / wick filter rescue the hammer? (H=3) ===")
    base = st.run_experiment(panel, side="hammer", horizons=(3,), n_draws=3000, seed=403)[3]
    print(f"plain hammer (lookback=10):  edge={base['edge_mean']*100:+.3f}%  "
          f"t={base['t']:+.2f}  p={base['p_placebo']:.3f}  n={base['n']}")
    for lb in (5, 20):
        r = st.run_experiment(panel, side="hammer", horizons=(3,), lookback=lb,
                              n_draws=3000, seed=403)[3]
        print(f"trend lookback={lb:2d}:           edge={r['edge_mean']*100:+.3f}%  "
              f"t={r['t']:+.2f}  p={r['p_placebo']:.3f}  n={r['n']}")
    for wm in (3.0, 4.0):
        r = st.run_experiment(panel, side="hammer", horizons=(3,), n_draws=3000,
                              seed=403, wick_mult=wm)[3]
        print(f"wick_mult={wm}:             edge={r['edge_mean']*100:+.3f}%  "
              f"t={r['t']:+.2f}  p={r['p_placebo']:.3f}  n={r['n']}")

    print("\n=== Synthetic positive control (H=1, side=any) ===")
    print("planted | events | cond%  edge%  | HAC t | placebo p | win%")
    for edge in (0.0, 0.005):
        sp, _ = data.synthetic_panel(edge=edge, seed=403)
        r = st.run_experiment(sp, side="any", horizons=(1,), n_draws=2000, seed=403)[1]
        print(f"{edge:6.3f}  | {r['n']:6d} | {r['cond_mean']*100:+.3f} {r['edge_mean']*100:+.3f} "
              f"| {r['t']:+5.2f} | {r['p_placebo']:8.3f} | {r['win']*100:4.1f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
