"""Real-tape verification — Study 350 (Dartboard-Portfolio). Regenerates docs/results.md.

Loads (or fetches) the mega-cap return panel + cap snapshot, runs the dartboard vs
cap-weighted index vs concentrated-expert race, reports the dartboard win-rate and the
HAC test of the median monkey minus the index — and the decisive control, the same test
minus the *equal-weight* index (which removes the size tilt the monkey exploits).

    python studies/350-dartboard-portfolio/examples/verify.py           # cache-only
    python studies/350-dartboard-portfolio/examples/verify.py --fetch   # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dartboard_portfolio import data, strategy as st  # noqa: E402

K = 10
N_DARTS = 500
REBAL = 12


def main(fetch: bool) -> None:
    rets, caps = data.load_real(fetch=fetch, allow_survivorship_bias=True)
    print(f"panel: {rets.index[0].date()} .. {rets.index[-1].date()}  "
          f"({len(rets)} months, {rets.shape[1]} names)")
    print(f"fingerprint: {data.fingerprint(rets)}\n")

    res = st.race(rets, caps, k=K, n_darts=N_DARTS, rebalance_every=REBAL, cost_bps=0.0)

    idx_s = st.summarize(res["index"])
    eq_s = st.summarize(res["equal_index"])
    exp_s = st.summarize(res["expert"])
    md_s = st.summarize(res["median_dart"])

    print("=== Race (gross, annual rebalance) ===")
    for name, s in [("Cap index", idx_s), ("Equal index", eq_s),
                    ("Expert (k largest)", exp_s), ("Median dartboard", md_s)]:
        print(f"{name:20s} CAGR={s['cagr']*100:+6.2f}%  Sharpe={s['sharpe']:5.2f}  "
              f"maxDD={s['max_dd']*100:6.1f}%")

    print(f"\nDartboard win-rate vs cap index : {res['win_rate_vs_index']*100:5.1f}%  "
          f"({N_DARTS} monkeys)")
    print(f"Dartboard win-rate vs expert    : {res['win_rate_vs_expert']*100:5.1f}%")

    t_idx = res["test_vs_index"]
    t_eq = res["test_vs_equal_index"]
    print(f"\nMedian dart − CAP index   : {t_idx['mean_diff_ann']*100:+.2f}%/yr  "
          f"HAC t={t_idx['tstat']:+.2f}  (n={t_idx['n']})")
    print(f"Median dart − EQUAL index : {t_eq['mean_diff_ann']*100:+.2f}%/yr  "
          f"HAC t={t_eq['tstat']:+.2f}  (n={t_eq['n']})  <- the size-tilt control")

    lo, hi = st.block_bootstrap_ci(res["median_dart"] - res["index"])
    print(f"\nBlock-bootstrap 95% CI on monthly (dart − cap index): "
          f"[{lo*1e4:+.1f}, {hi*1e4:+.1f}] bps")

    # Cost sweep — the dartboard's full re-draw is high turnover.
    print("\n=== Cost sweep: dartboard win-rate vs cap index (net) ===")
    for c in (0.0, 10.0, 25.0, 50.0):
        rc = st.race(rets, caps, k=K, n_darts=N_DARTS, rebalance_every=REBAL,
                     cost_bps=c, seed=350)
        print(f"cost={c:5.1f} bps/leg  dartboard win-rate={rc['win_rate_vs_index']*100:5.1f}%  "
              f"median dart−cap t={rc['test_vs_index']['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
