"""Real-tape verification — Study 954 (High Yield in Disguise). Feeds docs/results.md.

Reads cached daily total-return closes for HYG/JNK/USHY (high yield), SPY (equity leg),
IEF (duration leg) and BIL (cash), fits the held-out ``w * SPY + (1 - w) * IEF``
replication of each high-yield fund, and prints the replication quality, the
excess-of-cash Sharpe race, the vol-matched HAC *t*, the bootstrap CIs, the era cut, the
crisis table, the cost sweep and the estimation-window sweep. Network only on ``--fetch``.

    python studies/954-hy-as-levered-equity/examples/verify.py            # cache-only
    python studies/954-hy-as-levered-equity/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hy_replication import data, strategy as st  # noqa: E402

HY = "HYG"
EQ, DUR, CASH = "SPY", "IEF", "BIL"
DUR_LEGS = ("SHY", "IEI", "IEF", "TLT")   # 1-3y, 3-7y, 7-10y (headline), 20y+
WINDOW = 252
COST_BPS = 2.0


def legs(px, hy_tk):
    d = px[[hy_tk, EQ, DUR, CASH]].dropna()
    return d[hy_tk], d[EQ], d[DUR], d[CASH]


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    hy, eq, dur, cash = legs(px, HY)

    print(f"{HY} vs {EQ}+{DUR} (cash {CASH}): raw overlap {hy.index[0].date()} -> {hy.index[-1].date()}")
    print(f"as-of {data.AS_OF}  fp={data.fingerprint(px[[HY, EQ, DUR, CASH]].dropna())}")

    cmp = st.compare(hy, eq, dur, cash, window=WINDOW, cost_bps=COST_BPS)
    print(f"\nheld-out record: {cmp['start'].date()} -> {cmp['end'].date()}  n={cmp['hy']['n_days']}")
    print(f"fitted equity share w: mean {cmp['w_mean']:.3f}  range [{cmp['w_min']:.3f}, {cmp['w_max']:.3f}]"
          f"  max short notional {cmp['short_notional_max']:.3f}  turnover {cmp['turnover_per_year']:.2f}/yr")

    print("\n=== is HY a costume? replication quality ===")
    print(f"  daily corr {cmp['corr']:.4f}   R^2 {cmp['r2']:.4f}   tracking error {cmp['tracking_error_ann']:.2%}/yr")
    print(f"  residual (HY - replication): {cmp['residual_ann']:+.2%}/yr  HAC t = {cmp['t_residual']:+.2f}")
    print(st.replication_r2_by_horizon(cmp["bt"]).round(4).to_string())

    print("\n=== does it pay? excess-of-cash race ===")

    def fmt(s):
        return (f"exSharpe={s['sharpe']:+.3f}  exCAGR={s['cagr']:+.2%}  Vol={s['vol_ann']:.2%}  "
                f"HAC t={s['tstat']:+.2f}")

    print(f"  {HY:12s}: {fmt(cmp['hy'])}  MaxDD(abs)={cmp['dd_hy_abs']:+.2%}")
    print(f"  replication : {fmt(cmp['repl'])}  MaxDD(abs)={cmp['dd_repl_abs']:+.2%}")
    print(f"  lived (absolute, total-return) CAGR: {HY} {cmp['cagr_hy_abs']:+.2%}  "
          f"replication {cmp['cagr_repl_abs']:+.2%}  cash({CASH}) {cmp['cagr_cash_abs']:+.2%}")
    print(f"  excess-Sharpe gap (HY - replication): {cmp['excess_sharpe_gap']:+.3f}  "
          f"vol-matched HAC t = {cmp['t_gap']:+.2f}  (vol-match is ex-post: a test "
          f"statistic, not a tradable path)")

    print("\n=== bootstrap CIs (2000 draws, 21-day blocks) ===")
    for tag, s in [(HY, cmp["e_hy"]), ("replication", cmp["e_repl"])]:
        ci = st.bootstrap_sharpe_ci(s, seed=954)
        print(f"  {tag:12s}: Sharpe {ci['sharpe']:+.3f}  95% CI [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]"
              f"  share<0 {ci['frac_negative']:.3f}")
    d = st.vol_matched_diff(cmp["e_hy"], cmp["e_repl"])
    cid = st.bootstrap_sharpe_ci(d, seed=954)
    print(f"  vol-matched gap: {cid['sharpe']:+.3f}  95% CI [{cid['ci_low']:+.3f}, {cid['ci_high']:+.3f}]"
          f"  share<0 {cid['frac_negative']:.3f}")

    print("\n=== era cut (split 2017-01-01) ===")
    for tag, e in st.era_cut(cmp, split="2017-01-01").items():
        if e is None:
            continue
        print(f"  {tag:5s} (n={e['n_days']}): HY {e['sharpe_hy']:+.3f} / repl {e['sharpe_repl']:+.3f}"
              f"  gap {e['excess_sharpe_gap']:+.3f} (t={e['t_gap']:+.2f})"
              f"  residual {e['residual_ann']:+.2%}/yr (t={e['t_residual']:+.2f})")

    print("\n=== crisis table (absolute, lived) ===")
    print(st.crisis_table(cmp).round(4).to_string())

    print("\n=== cost sweep (one-way bps on the replication's rebalance) ===")
    print(st.cost_sweep(hy, eq, dur, cash, window=WINDOW).round(4).to_string())

    print("\n=== estimation-window sweep ===")
    print(st.window_sweep(hy, eq, dur, cash).round(4).to_string())

    print("\n=== duration-leg sweep (which Treasury maturity is 'the' duration leg?) ===")
    d_all = px[[HY, EQ, CASH] + list(DUR_LEGS)].dropna()
    print(st.leg_sweep(d_all[HY], d_all[EQ], {k: d_all[k] for k in DUR_LEGS},
                       d_all[CASH], window=WINDOW, cost_bps=COST_BPS).round(4).to_string())

    print("\n=== cross-checks: other high-yield funds ===")
    for tk in ("JNK", "USHY"):
        h2, e2, d2, c2 = legs(px, tk)
        c = st.compare(h2, e2, d2, c2, window=WINDOW, cost_bps=COST_BPS)
        print(f"  {tk:5s} {c['start'].date()}->{c['end'].date()} n={c['hy']['n_days']}: "
              f"w {c['w_mean']:.3f}  R^2 {c['r2']:.3f}  gap {c['excess_sharpe_gap']:+.3f} (t={c['t_gap']:+.2f})"
              f"  residual {c['residual_ann']:+.2%}/yr (t={c['t_residual']:+.2f})")

    print("\n=== synthetic control (machinery proof, never supports the stamp) ===")
    for ss in (1.0, 0.0):
        p, truth = data.synthetic_panel(signal_strength=ss, seed=954)
        s = st.synthetic_detect(p)
        print(f"  signal_strength={ss:.0f} (planted drag {truth['planted_drag_ann']:.2%}/yr, "
              f"w_true {truth['w_true']:.2f}): w_hat {s['w_mean']:.3f}  R^2 {s['r2']:.3f}  "
              f"residual {s['residual_ann']:+.2%}/yr (t={s['t_residual']:+.2f})  "
              f"gap {s['excess_sharpe_gap']:+.3f} (t={s['t_gap']:+.2f})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
