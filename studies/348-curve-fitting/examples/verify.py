"""Real-tape verification — Study 348 (Curve-Fitting). Regenerates docs/results.md numbers.

Runs the optimise-then-validate experiment on the real SPY tape: sweep the MA-crossover
grid on the in-sample half, crown the IS winner, run it out of sample, and strip the market
beta out of any apparent OOS edge. Also prints the synthetic null and positive control so
the machinery proof travels with the real result.

    python studies/348-curve-fitting/examples/verify.py           # cache-only
    python studies/348-curve-fitting/examples/verify.py --fetch   # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from curve_fitting import data, strategy as st  # noqa: E402

TICKER = "SPY"


def main(fetch: bool) -> None:
    px = data.load_real(ticker=TICKER, fetch=fetch)
    # Drop the in-progress final month (as-of is never in the future / a partial bar).
    asof = px.index.max().to_period("M").to_timestamp()
    px = px[px.index < asof]
    is_px, oos_px = st.split(px, 0.5)
    print(f"{TICKER} tape: {px.index[0].date()} .. {px.index[-1].date()}  "
          f"({len(px)} days)   fingerprint: {data.fingerprint(px)}")
    print(f"  IS  {is_px.index[0].date()} .. {is_px.index[-1].date()}")
    print(f"  OOS {oos_px.index[0].date()} .. {oos_px.index[-1].date()}\n")

    res = st.curve_fit_experiment(px, frac_is=0.5, cost_bps=0.0)
    print("=== Optimise-then-validate (real SPY) ===")
    print(f"IS-best crossover : fast={res['winner'][0]}, slow={res['winner'][1]}")
    print(f"IS  Sharpe        : {res['is_sharpe']:.2f}")
    print(f"OOS Sharpe        : {res['oos_sharpe']:.2f}  "
          f"(HAC t={res['oos_test']['tstat']:+.2f}, "
          f"95% CI [{res['oos_ci'][0]:.2f}, {res['oos_ci'][1]:.2f}])")
    print(f"shrinkage (IS-OOS): {res['shrinkage']:.2f}")
    print(f"grid OOS median Sharpe: {res['grid_oos_median_sharpe']:.2f}  (grid={res['n_grid']} pairs)")

    a = st.alpha_vs_buyhold(oos_px, *res["winner"])
    print("\n=== Strip the beta out (OOS) ===")
    print(f"winner OOS Sharpe {a['strat_sharpe']:.2f}  vs buy-hold OOS Sharpe {a['buyhold_sharpe']:.2f}")
    print(f"avg exposure {a['avg_exposure']:.2f}  | "
          f"alpha vs buy-hold {a['alpha_ann']*100:+.2f}%/yr  HAC t={a['alpha_test']['tstat']:+.2f}")

    print("\n=== The mirage grows with the search (real SPY) ===")
    print(st.grid_size_sweep(px).to_string(index=False))

    print("\n=== Synthetic null (random walk) — the mirage in a bottle ===")
    pn, _ = data.synthetic_series(n_days=5000, trend_strength=0.0, trend_halflife=80, seed=348)
    rn = st.curve_fit_experiment(pn, frac_is=0.5)
    print(f"winner {rn['winner']}  IS {rn['is_sharpe']:.2f} -> OOS {rn['oos_sharpe']:.2f}  "
          f"(t={rn['oos_test']['tstat']:+.2f})  shrinkage {rn['shrinkage']:.2f}")

    print("\n=== Synthetic positive control (planted trend) — the harness banks real signal ===")
    pc, _ = data.synthetic_series(n_days=5000, trend_strength=0.0015, trend_halflife=80, seed=348)
    rc = st.curve_fit_experiment(pc, frac_is=0.5)
    print(f"winner {rc['winner']}  IS {rc['is_sharpe']:.2f} -> OOS {rc['oos_sharpe']:.2f}  "
          f"(t={rc['oos_test']['tstat']:+.2f})  shrinkage {rc['shrinkage']:.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
