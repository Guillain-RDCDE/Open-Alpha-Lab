"""Real-tape verification — Study 306 (Crack-Spread). Regenerates docs/results.md numbers.

Builds the 3-2-1 crack spread from RB=F/HO=F/CL=F and an equal-weight total-return refiner
basket (VLO/MPC/PSX), then asks the only question that matters for a *tradable* claim: does
*yesterday's* crack change forecast *today's* basket return? Reports the predictive HAC *t*
(the inference-bar number) against the coincident relation, and races a crack-timer vs
always-long buy-and-hold on the active return. Network is touched only with --fetch.

    python studies/306-crack-spread/examples/verify.py            # cache-only
    python studies/306-crack-spread/examples/verify.py --fetch    # refresh the tapes

The pinned run drops the partial current month (as-of 2026-06-17 → data through May 2026).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crack_spread import data, strategy as st  # noqa: E402

AS_OF_CUTOFF = "2026-06-01"  # drop the partial June 2026 month


def main(fetch: bool) -> None:
    f = data.load_real(fetch=fetch)
    f = f[f.index < AS_OF_CUTOFF]
    print(f"Crack/refiner tape: {f.index[0].date()} to {f.index[-1].date()}, "
          f"{len(f)} days   fp={data.fingerprint(f)}")
    print(f"Crack spread ($/bbl): mean {f['crack'].mean():.1f}  "
          f"min {f['crack'].min():.1f}  max {f['crack'].max():.1f}\n")

    pr = st.predictive_regression(f)
    co = st.contemporaneous_regression(f)
    print("=== Does the crack PREDICT refiner stocks? ===")
    print(f"  Predictive (lag-1)  slope={pr['slope']:+.5f}  HAC t={pr['tstat']:+.3f}  "
          f"R2={pr['r2']:.5f}  n={pr['n']}")
    print(f"  Coincident (lag-0)  slope={co['slope']:+.5f}  HAC t={co['tstat']:+.3f}  "
          f"R2={co['r2']:.5f}  n={co['n']}\n")

    print("=== Crack-timer vs always-long buy-and-hold (active = timer - B&H) ===")
    for name, pos in (("level (>trailing median)", st.level_signal(f["crack"])),
                      ("momentum (20-day)", st.momentum_signal(f["crack"], 20)),
                      ("momentum (1-day)", st.momentum_signal(f["crack"], 1))):
        s = st.summarize_timer(f, pos, cost_bps=0.0)
        print(f"  {name:26s} timer {s['timer_ann']:+5.1f}%/yr (Sh {s['timer_sharpe']:.2f})  "
              f"B&H {s['bh_ann']:+5.1f}%/yr (Sh {s['bh_sharpe']:.2f})  "
              f"active {s['active_ann']:+5.2f}%/yr  t={s['active_t']:+.2f}  "
              f"pim {s['pct_in_market']:.0f}%  sw {s['n_switches']}")

    print("\n=== Cost sweep — 20-day crack-momentum timer, active return ===")
    pos = st.momentum_signal(f["crack"], 20)
    for c in (0.0, 2.0, 5.0, 10.0):
        s = st.summarize_timer(f, pos, cost_bps=c)
        print(f"  {c:5.1f} bps  active {s['active_ann']:+6.2f}%/yr  t={s['active_t']:+.2f}")

    lo, hi = st.block_bootstrap_ci(st.book_returns(pos, f["stock_ret"]))
    print(f"\n  20-day timer Sharpe 95% block-bootstrap CI: [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
