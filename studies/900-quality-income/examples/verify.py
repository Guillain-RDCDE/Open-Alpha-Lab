"""Reproducible headline run for Study 900 — Quality-Income.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the study's cached yfinance tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quality_income import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

try:
    from quantlab.repro import fingerprint
except Exception:
    fingerprint = None


def main():
    print("# Quality-Income — does screening dividends for QUALITY beat CHASING YIELD?")
    print("# quality sleeve = SCHD+NOBL  |  yield sleeve = SPHD+VYM  |  vs SPY  |  cash = BIL")
    print("# (Dedup: 206 dividend-aristocrats / 233 shareholder-yield / 57 yield-trap grade the")
    print("#  academic signals; 601 audits factor wrappers. Here we RACE two live dividend sleeves.)")

    if not data.have_real():
        print("(cache miss — fetching the six-ticker tape once)")
        data.fetch()

    prices = data.load_prices()
    mret = data.monthly_total_returns(prices)
    cash = st.cash_returns(mret, data.CASH)
    spy = mret[data.BENCH].dropna()

    quality = st.sleeve_returns(mret, data.QUALITY)
    yld = st.sleeve_returns(mret, data.YIELD)

    print("\n# Data stamp")
    print(f"tape        : {prices.shape[0]} days x {prices.shape[1]} tickers  "
          f"{prices.index.min().date()} -> {prices.index.max().date()} "
          f"(yfinance auto-adjusted = TOTAL RETURN)")
    print(f"as-of       : monthly stats sliced to {data.AS_OF} (last complete month)")
    for name, mem in [("quality", data.QUALITY), ("yield", data.YIELD)]:
        s = st.sleeve_returns(mret, mem)
        print(f"  {name:8s} {'+'.join(mem):12s}: {s.index.min().date()} -> "
              f"{s.index.max().date()}  ({len(s)} complete months)")
    if fingerprint is not None:
        panel = pd.concat([quality.rename("Q"), yld.rename("Y"),
                           spy.rename("SPY"), cash.rename("cash")], axis=1).dropna()
        print(f"fingerprint : common-window panel = {fingerprint(panel)}")

    # Common window (NOBL-bound) race.
    common = pd.concat([quality, yld, spy, cash], axis=1).dropna().index
    q_c, y_c, spy_c, cash_c = (quality.loc[common], yld.loc[common],
                               spy.loc[common], cash.loc[common])
    print(f"\n# THE RACE — common window {common.min().date()} -> {common.max().date()} "
          f"({len(common)} months), all excess of cash (minus BIL)")
    for name, s in [("QUALITY (SCHD+NOBL)", q_c), ("YIELD  (SPHD+VYM) ", y_c),
                    ("SPY                ", spy_c)]:
        a = st.ann_stats(s, cash_c)
        print(f"  {name}: CAGR {a['cagr']*100:5.2f}%  vol {a['vol']*100:4.1f}%  "
              f"exSharpe {a['sharpe']:.3f}  maxDD {a['maxdd']*100:6.1f}%  $1->{a['wealth']:.2f}")

    print("\n# HEADLINE — quality vs yield Sharpe gap + HAC t on the monthly difference")
    g = st.sharpe_gap_test(q_c, y_c, cash_c, lags=6)
    print(f"  excess Sharpe: quality {g['sharpe_q']:.3f}  vs yield {g['sharpe_y']:.3f}  "
          f"-> GAP {g['sharpe_gap']:+.3f}")
    print(f"  quality-minus-yield spread: {g['diff_mean_bps']:+.1f} bps/mo "
          f"({g['diff_ann_pct']:+.2f}%/yr)  one-sample t {g['t_1s']:+.2f}  NW t {g['t_nw']:+.2f}")

    print("\n# Paired block-bootstrap CI on the Sharpe gap (4,000 draws, block 6)")
    bs = st.sharpe_gap_bootstrap(q_c, y_c, cash_c, seed=900)
    print(f"  gap {bs['obs']:+.3f}  95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}]  "
          f"P(gap<0) = {bs['frac_negative']:.3f}")

    print("\n# ERA CUT (split 2020-01-01)")
    eras = st.era_cut(q_c, y_c, cash_c, split="2020-01-01", lags=6)
    for lbl in ("early", "late"):
        e = eras[lbl]
        if "sharpe_gap" in e:
            print(f"  {lbl:5s} (n={e['n_months']:3d}): gap {e['sharpe_gap']:+.3f}  "
                  f"diff {e['diff_ann_pct']:+.2f}%/yr  NW t {e['t_nw']:+.2f}")

    print("\n# vs SPY — do the sleeves beat the plain benchmark? (excess Sharpe race)")
    gq_spy = st.sharpe_gap_test(q_c, spy_c, cash_c, lags=6)
    gy_spy = st.sharpe_gap_test(y_c, spy_c, cash_c, lags=6)
    print(f"  quality vs SPY: Sharpe {gq_spy['sharpe_q']:.3f} vs {gq_spy['sharpe_y']:.3f} "
          f"(gap {gq_spy['sharpe_gap']:+.3f})  diff {gq_spy['diff_ann_pct']:+.2f}%/yr  "
          f"NW t {gq_spy['t_nw']:+.2f}")
    print(f"  yield   vs SPY: Sharpe {gy_spy['sharpe_q']:.3f} vs {gy_spy['sharpe_y']:.3f} "
          f"(gap {gy_spy['sharpe_gap']:+.3f})  diff {gy_spy['diff_ann_pct']:+.2f}%/yr  "
          f"NW t {gy_spy['t_nw']:+.2f}")

    print("\n# COSTED — monthly rebalance turnover x one-way spread (long-only, no borrow)")
    mret_c = mret.loc[common]  # common window so the costed Sharpes match the race
    for cb in (3.0, 10.0):
        cq = st.costed_sleeve(mret_c, data.QUALITY, cash_c, one_way_bps=cb)
        cy = st.costed_sleeve(mret_c, data.YIELD, cash_c, one_way_bps=cb)
        print(f"  {cb:>4.0f} bps/side: quality gross Sharpe {cq['gross_sharpe']:.3f} -> "
              f"net {cq['net_sharpe']:.3f} (drag {cq['cost_drag_bps_yr']:.1f} bps/yr, "
              f"turn {cq['avg_turnover_pct']:.1f}%/mo);  yield {cy['gross_sharpe']:.3f} -> "
              f"{cy['net_sharpe']:.3f}")

    print("\n# CALENDAR YEAR total returns (%)")
    cy_tab = st.calendar_year_table({"QUALITY": q_c, "YIELD": y_c, "SPY": spy_c})
    with pd.option_context("display.float_format", lambda v: f"{v*100:6.1f}"):
        print(cy_tab.to_string())

    print("\n# SYNTHETIC CONTROL — machinery proof only, never market evidence")
    null_t = []
    for s_ in range(20):
        w = data.synthetic_world(n_months=150, edge=0.0, seed=900 + s_)
        null_t.append(st.synthetic_detect(w)["t_nw"])
    null_t = np.asarray(null_t)
    print(f"  null (edge=0), 20 seeds: gap NW t mean {null_t.mean():+.2f} "
          f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20")
    planted = st.synthetic_detect(data.synthetic_world(n_months=150, edge=0.03, seed=900))
    print(f"  planted (edge=+3%/yr, seed 900): gap {planted['sharpe_gap']:+.3f}  "
          f"NW t {planted['t_nw']:+.2f}  diff {planted['diff_ann_pct']:+.2f}%/yr")


if __name__ == "__main__":
    main()
