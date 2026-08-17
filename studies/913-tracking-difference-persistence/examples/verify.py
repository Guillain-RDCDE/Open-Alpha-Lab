"""Real-tape verification — Study 913 (Tracking-Difference Persistence).

Regenerates every number in ``docs/results.md``. Reads cached daily total-return closes
for two index-fund families, computes each fund's complete-calendar-year tracking
difference against a common family proxy, tests year-over-year rank persistence
(Spearman + a year-label permutation null), then races the annually-rebalanced rules —
``winner`` (last year's best TD), ``loser``, ``cheapest`` (lowest published expense ratio,
never switches fund), ``leader`` (the flagship, likewise) and ``eqw`` — with exactly one
execution lag, a one-way switching cost, an era cut, a cost sweep and the capital-gains
break-even assumption sweep. Because ``cheapest`` reads TODAY's fee sheet, a hindsight-free
control is printed alongside it: every non-flagship fund, and their equal-weight blend,
raced against the flagship with no fee information at all. Network only on ``--fetch``.

    python studies/913-tracking-difference-persistence/examples/verify.py
    python studies/913-tracking-difference-persistence/examples/verify.py --fetch
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from td_persist import data, strategy as st  # noqa: E402

COST_BPS = 1.0        # one-way x NAV on realised turnover; no short leg, so no borrow
FAMILIES = [
    ("S&P 500 ETF trio", list(data.SP500_ETFS)),
    ("S&P 500 full ladder", list(data.SP500_FUNDS)),
    ("Nasdaq-100 pair", list(data.NDX_ETFS)),
]


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    er = data.EXPENSE_RATIO_BPS

    print(f"as-of {data.AS_OF}   unavailable from source: {', '.join(data.UNAVAILABLE)}")
    print(data.price_only_note())

    for name, tk in FAMILIES:
        p = px[tk].dropna()
        ann = st.annual_returns(p)
        td = st.tracking_difference(ann, proxy="mean")
        fp = data.fingerprint(p)
        print("\n" + "=" * 78)
        print(f"{name}: {' '.join(tk)}")
        print(f"  window {p.index[0].date()} -> {p.index[-1].date()}  n={len(p)}  fp={fp}")
        print(f"  complete calendar years: {int(td.index.min())}-{int(td.index.max())} "
              f"({len(td)} years)   expense ratios (bp): "
              + ", ".join(f"{t}={er[t]:g}" for t in tk))

        print("\n  --- annual tracking difference vs the family mean (bp) ---")
        print(td.round(1).to_string())
        # np, not DataFrame.stack(): identical value, stable across pandas 2.x and 3.x.
        print(f"  noise floor: sd of relative TD = "
              f"{np.nanstd(td.to_numpy(dtype=float), ddof=1):.1f} bp")

        pers = st.persistence(td, n_perm=5000, seed=913)
        print("\n  --- persistence of the TD rank ---")
        print(f"  mean year-over-year Spearman {pers['mean_spearman']:+.3f} "
              f"(t={pers['t_spearman']:+.2f}, n={pers['n_pairs']} pairs, "
              f"permutation p={pers['p_permutation']:.3f})")
        print(f"  residual (fund-demeaned) Spearman {pers['mean_spearman_residual']:+.3f} "
              f"(t={pers['t_spearman_residual']:+.2f})")
        corr = st._pairwise_rank_corr_matrix(td)
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        print(f"  mean rank corr over ALL year pairs {np.nanmean(off):+.3f} "
              f"vs consecutive pairs {pers['mean_spearman']:+.3f} "
              "(a level, not a memory, when these match)")
        if pers["degenerate_two_fund"]:
            print("  NOTE: two funds -> Spearman degenerates to a sign-agreement indicator.")

        print("\n  --- the rules (annual rebalance, one-day lag, "
              f"{COST_BPS:g} bp one-way) ---")
        for a, b in [("winner", "cheapest"), ("cheapest", "leader"),
                     ("winner", "leader"), ("loser", "cheapest")]:
            g = st.gap_stats(p, er, a, b, cost_bps=COST_BPS)
            print(f"  {a:8s}-{b:8s}: annual {g['annual_gap_bp_yr']:+7.2f} bp/yr "
                  f"t={g['t_annual']:+5.2f} (HAC {g['t_annual_hac']:+5.2f}) "
                  f"({g['years_positive']}/{g['n_years']} yrs +) "
                  f"CI[{g['ci_low']:+.1f},{g['ci_high']:+.1f}] | "
                  f"daily {g['daily_gap_bp_yr']:+7.2f} bp/yr t_HAC={g['t_daily_hac']:+5.2f}")
        print(f"  winner-rule fund switches: {st.switch_count(td, er, 'winner')} "
              f"over {len(td) - 1} live years")

        print("\n  --- HINDSIGHT-FREE check: every fund vs the flagship, no fee sheet ---")
        print("  (the `cheapest` arm is picked with TODAY's expense ratios; these rows are "
              "not)")
        print(st.per_fund_gap_vs_leader(p).round(2).to_string())

        print("\n  --- excess-of-cash Sharpe race (vs BIL; both arms excess) ---")
        print(st.excess_of_cash_race(p, px[data.CASH], er, cost_bps=COST_BPS)
              .round(4).to_string())

        print("\n  --- era cut (cheapest - leader) ---")
        e = st.era_cut(p, er, "cheapest", "leader", cost_bps=COST_BPS)
        for tag in ("early", "late"):
            d = e[tag]
            print(f"  {tag:5s} {d['years']:>10s} (n={d['n_years']:2d}): "
                  f"{d['annual_gap_bp_yr']:+6.2f} bp/yr  t={d['t_annual']:+5.2f}  "
                  f"{d['years_positive']}/{d['n_years']} yrs +")

        print("\n  --- cost sweep (winner - cheapest) ---")
        print(st.cost_sweep(p, er, "winner", "cheapest").round(2).to_string())

    # ------------------------------------------------------------------ #
    # The price-only proxy, shown rather than asserted.
    # ------------------------------------------------------------------ #
    p = px[list(data.SP500_FUNDS) + [data.INDEX_PROXY]].dropna()
    td_gspc = st.tracking_difference(st.annual_returns(p), proxy=data.INDEX_PROXY)
    print("\n" + "=" * 78)
    print("TD vs ^GSPC (PRICE-ONLY proxy), mean bp/yr — this is the dividend yield, "
          "not a tracking difference:")
    print(td_gspc.mean().round(0).to_string())

    # ------------------------------------------------------------------ #
    # The tax break-even — an ASSUMPTION sweep, not a measurement.
    # ------------------------------------------------------------------ #
    gap = data.EXPENSE_RATIO_BPS["SPY"] - data.EXPENSE_RATIO_BPS["VOO"]
    print(f"\nCapital-gains break-even for a SPY -> VOO switch at the published "
          f"{gap:.2f} bp/yr fee gap (years to repay the realised-gain tax; "
          "ASSUMPTION grid, not tape):")
    print(st.tax_breakeven_years(gap).round(1).to_string())

    # ------------------------------------------------------------------ #
    # Synthetic control — machinery proof only, never supports the stamp.
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("Synthetic control (machinery proof; never supports a real-tape stamp)")
    for ss in (1.0, 0.0):
        rows = []
        for s in range(8):
            sp, truth = data.synthetic_panel(signal_strength=ss, seed=913 + s)
            rows.append(st.synthetic_detect(sp, truth))
        f = pd.DataFrame(rows)
        tag = "planted fee ladder" if ss == 1.0 else "flat-fee null    "
        print(f"  {tag} (8 seeds): mean Spearman {f['mean_spearman'].mean():+.3f} "
              f"(min {f['mean_spearman'].min():+.3f}, max {f['mean_spearman'].max():+.3f}) | "
              f"cheapest-leader {f['cheapest_minus_leader_bp'].mean():+.2f} bp "
              f"(t={f['t_cheapest_minus_leader'].mean():+.2f})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
