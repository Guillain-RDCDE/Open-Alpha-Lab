"""Real-tape verification — Study 914 (Securities-Lending Offset).

Regenerates every number in docs/results.md. Reads cached daily total-return closes for
five same-asset-class ETF pairs, measures each pair's fee-adjusted relative tracking
residual (monthly log returns, HAC *t*, block-bootstrap CI, noise floor), cuts it by era,
sweeps the assumed expense ratios, pools the tight pairs, prices the long/short
fee-harvest trade against a borrow sweep, and runs the offline synthetic control.

    python studies/914-sec-lending-offset/examples/verify.py            # cache-only
    python studies/914-sec-lending-offset/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lending_offset import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    ER = data.EXPENSE_RATIOS

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"frame {px.index[0].date()} -> {px.index[-1].date()}  n={len(px)}")

    print("\n=== fee-adjusted relative residual, monthly log total returns ===")
    print("residual = annualised drift(a - b) + (ER_a - ER_b);  0 = fees explain the gap")
    tbl = st.residual_table(px, data.PAIRS, ER, freq="M")
    for _, r in tbl.iterrows():
        print(f"  {r['a']:>4s}-{r['b']:<4s} [{r['asset_class']:<30s}] "
              f"{r['start']}->{r['end']} n={r['n_obs']:3d} "
              f"drift {r['drift_bp']:+7.1f}  dER {r['delta_er_bp']:+6.1f}  "
              f"resid {r['residual_bp']:+7.1f}bp  t={r['t_hac']:+5.2f}  "
              f"CI[{r['ci_low_bp']:+7.1f},{r['ci_high_bp']:+7.1f}]  "
              f"TE {r['te_pct']:.2f}%  floor {r['noise_floor_bp']:.0f}bp  beta {r['beta']:.3f}")

    print("\n=== same, DAILY log returns (robustness — dividend-timing noise inflates TE) ===")
    tbl_d = st.residual_table(px, data.PAIRS, ER, freq="D", n_boot=1500)
    for _, r in tbl_d.iterrows():
        print(f"  {r['a']:>4s}-{r['b']:<4s} resid {r['residual_bp']:+7.1f}bp  "
              f"t={r['t_hac']:+5.2f}  TE {r['te_pct']:.2f}%")

    print("\n=== era cut (split 2020-01-01, monthly) ===")
    for a, b, klass, _n in data.PAIRS:
        eras = st.era_cut(px, a, b, ER[a], ER[b], split="2020-01-01")
        parts = []
        for tag in ("early", "late"):
            e = eras[tag]
            parts.append("n/a" if e is None else
                         f"{tag} n={e['n_obs']:3d} resid {e['residual_bp']:+7.1f} t={e['t_hac']:+5.2f}")
        print(f"  {a:>4s}-{b:<4s}  " + "  |  ".join(parts))

    print("\n=== expense-ratio assumption sweep, DEAR leg (the non-tape input) ===")
    for a, b in (("EEM", "IEMG"), ("IWM", "IJR"), ("EFA", "IEFA")):
        for row in st.er_sweep(px, a, b, data.ER_ALTERNATIVES.get(a, (ER[a],)), ER[b]):
            print(f"  {a}-{b}  ER_{a}={row['er_a']:.4f}%  dER {row['delta_er_bp']:+6.1f}bp  "
                  f"resid {row['residual_bp']:+7.1f}bp  t={row['t_hac']:+5.2f}")

    print("\n=== expense-ratio sweep, CHEAP leg — THE DECISIVE ONE ===")
    print("every cheap leg was CUT during the sample; pinning it at today's fee")
    print("overstates the fee gap and pushes the residual UP. Grid = (early, mid, today).")
    for a, b, _k, _n in data.PAIRS:
        grid = data.ER_HISTORY_BAND[b]
        for row in st.er_sweep_b(px, a, b, ER[a], grid):
            tag = {grid[0]: "early ", grid[1]: "mid   ", grid[2]: "today "}[row["er_b"]]
            print(f"  {a:>4s}-{b:<4s} [{tag}] ER_{b}={row['er_b']:.4f}%  "
                  f"dER {row['delta_er_bp']:+6.1f}bp  resid {row['residual_bp']:+7.1f}bp  "
                  f"t={row['t_hac']:+5.2f}")

    print("\n=== the whole table re-run under three fee-history scenarios ===")
    for tag, ers in (("today (headline)", ER),
                     ("time-avg        ", data.EXPENSE_RATIOS_TIMEAVG),
                     ("early-era       ", data.EXPENSE_RATIOS_EARLY)):
        t2 = st.residual_table(px, data.PAIRS, ers, freq="M", n_boot=1)
        cells = "  ".join(f"{r['a']}-{r['b']} {r['residual_bp']:+6.1f}" for _, r in t2.iterrows())
        print(f"  {tag}: {cells}")

    print("\n=== pooled residual ===")
    tight = ("US large cap", "Emerging markets", "Developed ex-US")
    for tag, classes, ers in (
        ("tight pairs, today's fees ", tight, ER),
        ("tight pairs, early fees   ", tight, data.EXPENSE_RATIOS_EARLY),
        ("ALL FIVE pairs, today     ", None, ER),
    ):
        pooled = st.pooled_residual(px, data.PAIRS, ers, classes=classes)
        print(f"  {tag} {pooled['pairs']}")
        print(f"    n={pooled['n_obs']}  residual {pooled['residual_bp']:+.1f}bp/yr  "
              f"t={pooled['t_hac']:+.2f}  "
              f"CI[{pooled['ci_low_bp']:+.1f},{pooled['ci_high_bp']:+.1f}]  "
              f"TE {pooled['te_pct']:.2f}%  floor {pooled['noise_floor_bp']:.0f}bp")

    print("\n=== the tradable leg: long the cheap fund, short the dear one ===")
    print("static pair (no signal, so no lag is claimed); monthly equal-notional restore;")
    print("friction charged annually on FULL notional; short pays borrow")
    print("  cost conservatism check (true drift-rebalancing turnover vs what is charged):")
    for L, S in (("IEMG", "EEM"), ("IEFA", "EFA"), ("IVV", "SPY"), ("IJR", "IWM")):
        t = st.rebalance_turnover(px, L, S)
        print(f"    {t['pair']:>10s} turnover {t['turnover_ann_pct']:5.2f}%/yr -> true cost "
              f"{t['true_cost_bp_yr']:.2f}bp/yr  vs charged {t['charged_bp_yr']:.1f}bp/yr")
    for L, S in (("IEMG", "EEM"), ("IEFA", "EFA"), ("IVV", "SPY"), ("IJR", "IWM")):
        for row in st.borrow_sweep(px, L, S):
            print(f"  {L:>4s}/{S:<4s} borrow {row['borrow_bps']:5.0f}bp  "
                  f"ann {row['ann_bp']:+7.1f}bp  vol {row['vol_pct']:.2f}%  "
                  f"Sharpe {row['sharpe']:+5.2f}  t={row['t_hac']:+5.2f}  "
                  f"maxDD {row['max_drawdown']:+.2%}")

    print("\n=== each leg's own excess-of-cash Sharpe (context, vs BIL) ===")
    for tk in ("EEM", "IEMG", "IWM", "IJR", "SPY", "IVV"):
        s = st.excess_of_cash_sharpe(px, tk)
        print(f"  {tk:>4s}: excess Sharpe {s['excess_sharpe']:+.2f}  "
              f"ann excess {s['ann_excess_pct']:+.2f}%  n={s['n_months']}")

    print("\n=== synthetic control (offline, never supports the stamp) ===")
    pl_px, pl_tr = data.synthetic_daily(signal_strength=1.0, seed=914)
    pl = st.synthetic_detect(pl_px, pl_tr)
    print(f"  planted {pl['planted_bp']:+.0f}bp -> recovered {pl['residual_bp']:+.1f}bp "
          f"(err {pl['error_bp']:+.1f}) t={pl['t_hac']:+.2f}  TE {pl['te_pct']:.2f}%")
    nulls = []
    for s in range(8):
        nx, nt = data.synthetic_daily(signal_strength=0.0, seed=914 + s)
        nulls.append(st.synthetic_detect(nx, nt))
    nb = np.array([n["residual_bp"] for n in nulls])
    ntt = np.array([n["t_hac"] for n in nulls])
    print(f"  null x8: residual mean {nb.mean():+.1f}bp (sd {nb.std(ddof=1):.1f}), "
          f"|t|>=2 in {(np.abs(ntt) >= 2).sum()}/8")
    frames, ptr = data.synthetic_panel(signal_strength=1.0, seed=914)
    pool = st.synthetic_panel_detect(frames, ptr)
    print(f"  pooled panel ({pool['n_pairs']} pairs): planted {pool['planted_bp']:+.0f}bp -> "
          f"{pool['residual_bp']:+.1f}bp  t={pool['t_hac']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
