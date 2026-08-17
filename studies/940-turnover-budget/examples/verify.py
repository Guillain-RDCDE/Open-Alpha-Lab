"""Real-tape verification — Study 940 (The Turnover Budget). Regenerates docs/results.md.

Reads the eleven cached Select Sector SPDRs and BIL (the cash leg), runs one
cross-sectional 12-1 momentum sleeve at four rebalance speeds — daily, weekly, monthly,
quarterly — and prints the frequency table (gross vs net excess-of-cash Sharpe, annual
traded notional, break-even cost per unit turnover), the cost surface, the borrow sweep,
the era cut, the bootstrap CIs, the head-to-head races, the long-only cross-check raced
against equal weight, and the synthetic control. Network only on ``--fetch``.

    python studies/940-turnover-budget/examples/verify.py            # cache-only
    python studies/940-turnover-budget/examples/verify.py --fetch    # refresh the tapes

Runtime is roughly 15-25 minutes: every arm is a day-by-day drifting-weight simulation
over ~6,900 bars, and the cost/borrow sweeps alone run 40 arms.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab import repro  # noqa: E402
from turnover_budget import data, strategy as st  # noqa: E402

COST_BPS = 5.0
BORROW_BPS = 40.0


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    sec = px[list(data.SECTORS)].dropna(how="all")
    cash = px[data.CASH].dropna()
    bil_window = sec.index.intersection(cash.index)

    pd.set_option("display.width", 250)
    print(f"sector panel : {sec.index[0].date()} -> {sec.index[-1].date()}  n={len(sec)}")
    print(f"BIL cash leg : {cash.index[0].date()} -> {cash.index[-1].date()}  n={len(cash)}")
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px[list(data.SECTORS) + [data.CASH]])}")
    print(f"quantlab.repro fingerprint {repro.fingerprint(px)}")
    print("\nDollar-neutral book: the long-minus-short spread IS the excess-of-cash return "
          "(collateral earns the bill), so BIL gates only the long-only cross-check.")

    print("\n=== frequency table (cost 5 bps/unit turnover, borrow 40 bps/yr) ===")
    tbl = st.frequency_table(sec, cash, cost_bps=COST_BPS, borrow_bps=BORROW_BPS)
    print(tbl[["label", "n_days", "n_rebalances", "ann_turnover", "turnover_per_rebalance",
               "churn_share", "sharpe_gross", "sharpe_net", "ann_return_gross",
               "ann_return_net", "vol_ann", "max_drawdown", "t_gross", "t_net",
               "breakeven_bps"]].round(4).to_string())

    print("\n=== cost surface: net excess-of-cash Sharpe by (speed, cost per unit turnover) ===")
    cs = st.cost_sweep(sec, cash, borrow_bps=BORROW_BPS)
    print(cs.pivot(index="cost_bps", columns="label", values="sharpe_net").round(3).to_string())

    print("\n=== borrow sweep (net excess Sharpe, cost 5 bps) — the second assumption ===")
    bs = st.borrow_sweep(sec, cash, cost_bps=COST_BPS)
    print(bs.pivot(index="borrow_bps", columns="label", values="sharpe_net").round(3).to_string())

    print("\n=== parameter robustness (gross excess Sharpe / HAC t, monthly clock) ===")
    print("  the sort is fixed at the Jegadeesh-Titman convention, not tuned; here is the")
    print("  neighbourhood around it, so nobody has to take that on trust.")
    for tk, lb, sk in [(2, 252, 21), (3, 252, 21), (4, 252, 21), (5, 252, 21),
                       (3, 126, 21), (3, 252, 0), (3, 63, 5)]:
        b = st.run_backtest(sec, None, freq="M", cost_bps=0.0, borrow_bps=0.0,
                            top_k=tk, lookback=lb, skip=sk)
        s = st.summary(b["excess_gross"])
        print(f"  top_k={tk} lookback={lb:3d} skip={sk:2d}: "
              f"gross Sharpe {s['sharpe']:+.3f}  HAC t {s['tstat']:+.2f}  "
              f"ann {s['ann_return']*100:+.2f}%")

    print("\n=== era cut (split 2013-01-01) ===")
    print("  NB: the signal is rebuilt inside each half, so each half spends its first 12")
    print("  months warming up the 12-1 window (2013 is therefore in neither table).")
    for tag, tab in st.era_cut(sec, cash, split="2013-01-01",
                               cost_bps=COST_BPS, borrow_bps=BORROW_BPS).items():
        print(f"-- {tag}")
        print(tab[["label", "n_days", "ann_turnover", "sharpe_gross", "sharpe_net",
                   "ann_return_gross", "t_gross", "breakeven_bps"]].round(4).to_string())

    print("\n=== bootstrap Sharpe CIs (2,000 draws, 21-day blocks) ===")
    for f in st.FREQS:
        g = st.run_backtest(sec, None, freq=f, cost_bps=0.0, borrow_bps=0.0)
        n = st.run_backtest(sec, None, freq=f, cost_bps=COST_BPS, borrow_bps=BORROW_BPS)
        cg = st.bootstrap_sharpe_ci(g["excess_gross"], seed=940)
        cn = st.bootstrap_sharpe_ci(n["excess_net"], seed=940)
        print(f"  {st.FREQ_LABEL[f]:10s} gross {cg['sharpe']:+.3f} "
              f"CI [{cg['ci_low']:+.3f},{cg['ci_high']:+.3f}] | "
              f"net {cn['sharpe']:+.3f} CI [{cn['ci_low']:+.3f},{cn['ci_high']:+.3f}] "
              f"(share<0 {cn['frac_negative']:.3f})")

    print("\n=== head-to-head speed races ===")
    for a, b in [("D", "M"), ("D", "Q"), ("M", "Q")]:
        gr = st.frequency_race(sec, cash, fast=a, slow=b, cost_bps=0.0, borrow_bps=0.0)
        ne = st.frequency_race(sec, cash, fast=a, slow=b, cost_bps=COST_BPS, borrow_bps=BORROW_BPS)
        print(f"  {a} vs {b}: gross gap {gr['sharpe_gap']:+.3f} (t={gr['t_diff']:+.2f})  "
              f"net gap {ne['sharpe_gap']:+.3f} (t={ne['t_diff']:+.2f})")

    print("\n=== long-only cross-check (top-3, excess of BIL, raced against equal weight) ===")
    print("  the EW control runs through the SAME engine, on the SAME clock, with the same")
    print("  one-day lag, and pays the same 5 bps on its OWN traded notional. Both the net")
    print("  and the cost-free alpha are shown so friction and selection stay separable.")
    lo = st.long_only_vs_equal_weight(sec.loc[bil_window], cash.loc[bil_window],
                                      cost_bps=COST_BPS)
    print(lo.round(4).to_string())
    print("  the excess-of-cash t clears 2 for every speed — that is equity beta, not "
          "selection; the alpha-vs-equal-weight columns are the honest read.")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    p1, c1, _ = data.synthetic_panel(signal_strength=1.0, seed=940)
    d1 = st.synthetic_detect(p1, c1)
    print("  planted gross Sharpe:", {k: round(v, 2) for k, v in d1["sharpe_gross"].items()})
    print("  planted gross t     :", {k: round(v, 1) for k, v in d1["t_gross"].items()})
    print(f"  planted daily-minus-quarterly gross return: "
          f"{d1['daily_minus_quarterly_gross']*100:+.2f} pp/yr")
    p0, c0, _ = data.synthetic_panel(signal_strength=0.0, seed=940)
    d0 = st.synthetic_detect(p0, c0)
    print("  null    gross Sharpe:", {k: round(v, 3) for k, v in d0["sharpe_gross"].items()})
    nulls = np.array([
        st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=940 + s)[:2])
        ["sharpe_gross"]["M"] for s in range(6)
    ])
    print(f"  null x6 monthly gross Sharpe: mean {nulls.mean():+.3f} "
          f"sd {nulls.std(ddof=1):.3f}  max|.| {np.abs(nulls).max():.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
