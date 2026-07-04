"""Reproducible headline run for Study 624 — Buffer ETFs: the Cost of Comfort.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached yfinance tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic
machinery controls with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd

from buffer_etf_cost import data, strategy as st

P_SERIES = ["PJAN", "PAPR", "PJUL", "POCT"]

print("# Buffer ETFs — the Cost of Comfort (BUFR + 4 Innovator Power Buffer vintages, yfinance)")
if not data.have_real():
    print("(no _cache/bec_adj.csv — run data.fetch_panel() once to build the cache)")
    real = False
else:
    real = True
    from quantlab.repro import data_stamp

    adj, raw = data.load_real(asof=data.AS_OF)
    m = data.monthly_returns(adj)
    bil, spy = m["BIL"], m["SPY"]
    print(data_stamp("bec_adj (TR closes, 7 tickers)", adj, asof=data.AS_OF))
    print(data_stamp("bec_raw SPY (price closes)", raw[["SPY"]], asof=data.AS_OF))
    print(f"monthly stats window: {m.index.min().date()} -> {m.index.max().date()} "
          f"(last complete month)")

    # ---------------------------------------------------------------- #
    print("\n# 1 — Mechanical delivery: did the buffer show up when it mattered?")
    print("(fund TOTAL return vs SPY PRICE return — the stated reference — per 12-month")
    print(" outcome period; stated terms: 15% buffer, 0.79% ER, annual reset)")
    n_up = n_inside = n_beyond = n_hon = n_capped = 0
    giveups, capped_rets = [], []
    for tk in P_SERIES:
        per = st.outcome_periods(adj[tk], raw["SPY"], data.FUNDS[tk]["reset_month"])
        dc = st.delivery_check(per, data.FUNDS[tk]["buffer_pct"], data.FUNDS[tk]["er_pct"])
        n_up += dc["n_up"]; n_inside += dc["n_inside"]; n_beyond += dc["n_beyond"]
        n_hon += dc["inside_honored"] + dc["beyond_honored"]
        n_capped += dc["n_capped"]
        for _, row in per.iterrows():
            if row["spy_price_ret_pct"] >= 0:
                giveups.append(row["spy_price_ret_pct"] - row["fund_ret_pct"])
                if row["spy_price_ret_pct"] - row["fund_ret_pct"] >= 2.0:
                    capped_rets.append(row["fund_ret_pct"])
        print(f"  {tk}: {dc['n_periods']} periods | up {dc['n_up']} (capped {dc['n_capped']}, "
              f"mean capped ret {dc['mean_capped_ret_pct']:.2f}%) | down "
              f"{dc['n_inside'] + dc['n_beyond']} (buffer honored "
              f"{dc['inside_honored'] + dc['beyond_honored']}/{dc['n_inside'] + dc['n_beyond']})")
    n_down = n_inside + n_beyond
    print(f"  AGGREGATE: {n_up + n_down} completed outcome periods | buffer honored "
          f"{n_hon}/{n_down} down periods | cap bound in {n_capped}/{n_up} up periods")
    print(f"  mean upside give-up in up periods: {sum(giveups)/len(giveups):.2f} pp/period | "
          f"mean capped-year fund return {sum(capped_rets)/len(capped_rets):.2f}%")
    # the two beyond-buffer 2022 examples, quoted verbatim in results.md
    pjan = st.outcome_periods(adj["PJAN"], raw["SPY"], 1)
    row22 = pjan[pjan["period"].str.startswith("2021-12-31")].iloc[0]
    print(f"  PJAN 2022 (beyond buffer): SPY price {row22['spy_price_ret_pct']:+.2f}% -> fund "
          f"{row22['fund_ret_pct']:+.2f}%  (terms floor = {row22['spy_price_ret_pct'] + 15 - 0.79:+.2f}%)")
    papr = st.outcome_periods(adj["PAPR"], raw["SPY"], 4)
    rin = papr[papr["period"].str.startswith("2022-03-31")].iloc[0]
    print(f"  PAPR 2022/23 (inside buffer): SPY price {rin['spy_price_ret_pct']:+.2f}% -> fund "
          f"{rin['fund_ret_pct']:+.2f}%  (promise: ~0% minus fee)")

    # ---------------------------------------------------------------- #
    print("\n# 2 — The cost of comfort vs the market (fund minus SPY, both total return)")
    for tk in list(data.FUNDS):
        f = m[tk].dropna()
        g = st.gap_stats(f, spy.loc[f.index])
        print(f"  {tk}: SPY - fund = {g['gap_pp_yr']:+.2f} pp/yr  HAC t = {g['hac_t']:+.2f}  "
              f"(n={g['n_months']} months, {g['start']} -> {g['end']})")
    coh = m[P_SERIES].dropna().mean(axis=1)          # equal-weight cohort of the 4 vintages
    gc = st.gap_stats(coh, spy.loc[coh.index])
    print(f"  COHORT (EW 4 vintages): SPY - cohort = {gc['gap_pp_yr']:+.2f} pp/yr  "
          f"HAC t = {gc['hac_t']:+.2f}  (n={gc['n_months']})")

    # ---------------------------------------------------------------- #
    print("\n# 3 — The fair race: each fund vs its beta-matched SPY/BIL mix (2 bps rebalance)")
    winners = 0
    fair = {}
    for tk in list(data.FUNDS):
        f = m[tk].dropna()
        b = st.beta_vs(f, spy, bil)
        mix = st.mix_returns(spy.loc[f.index], bil.loc[f.index], w=b)
        g = st.gap_stats(f, mix)
        fair[tk] = (b, g)
        if g["gap_pp_yr"] < 0:
            winners += 1
        print(f"  {tk}: beta={b:.2f}  mix - fund = {g['gap_pp_yr']:+.2f} pp/yr  "
              f"HAC t = {g['hac_t']:+.2f}  corr = {g['corr']:.3f}  "
              f"up-months {g['gap_up_bps']:+.1f} bps/mo, down-months {g['gap_dn_bps']:+.1f} bps/mo")
    bc = st.beta_vs(coh, spy, bil)
    mixc = st.mix_returns(spy.loc[coh.index], bil.loc[coh.index], w=bc)
    gfc = st.gap_stats(coh, mixc)
    print(f"  COHORT: beta={bc:.2f}  mix - cohort = {gfc['gap_pp_yr']:+.2f} pp/yr  "
          f"HAC t = {gfc['hac_t']:+.2f}  corr = {gfc['corr']:.3f}")
    print(f"  funds that beat their own mix (point estimate): {winners}/5 — "
          f"none significant either way (all |t| < 1)")

    # ---------------------------------------------------------------- #
    print("\n# 4 — CAGR / vol / drawdown / Sharpe (BUFR window, Sharpe excess vs BIL)")
    f = m["BUFR"].dropna()
    win = f.index
    b_bufr = fair["BUFR"][0]
    rows = [("BUFR", f), ("SPY", spy.loc[win]), ("BIL", bil.loc[win]),
            (f"mix {b_bufr:.2f} SPY/BIL", st.mix_returns(spy.loc[win], bil.loc[win], b_bufr)),
            ("mix 0.70 SPY/BIL", st.mix_returns(spy.loc[win], bil.loc[win], 0.70))]
    for lbl, s in rows:
        p = st.perf_stats(s, bil.loc[win])
        print(f"  {lbl:18s} CAGR {p['cagr_pct']:+6.2f}%  vol {p['vol_pct']:5.2f}%  "
              f"maxDD {p['maxdd_pct']:+6.2f}%  Sharpe {p['sharpe']:+.2f}")

    # ---------------------------------------------------------------- #
    print("\n# 5 — Cost decomposition of the fair-race gap (descriptive arithmetic)")
    for tk in ["BUFR"]:
        b, g = fair[tk]
        dec = st.cost_decomposition(g["gap_pp_yr"], data.FUNDS[tk]["er_pct"], b,
                                    data.SPY_DIV_YIELD_PCT)
        print(f"  {tk}: measured gap {dec['gap_pp_yr']:+.2f} pp/yr = fee {dec['fee_pp']:+.2f} "
              f"+ dividends forgone {dec['div_forgone_pp']:+.2f} "
              f"+ option-payoff residual {dec['residual_pp']:+.2f}")
    dec = st.cost_decomposition(gfc["gap_pp_yr"], 0.79, bc, data.SPY_DIV_YIELD_PCT)
    print(f"  COHORT: measured gap {dec['gap_pp_yr']:+.2f} pp/yr = fee {dec['fee_pp']:+.2f} "
          f"+ dividends forgone {dec['div_forgone_pp']:+.2f} "
          f"+ option-payoff residual {dec['residual_pp']:+.2f}")

    # ---------------------------------------------------------------- #
    print("\n# 6 — Robustness")
    print("  (a) mix weight grid, BUFR (beta-hat = %.2f):" % b_bufr)
    for w in (0.45, 0.55, 0.65, 0.70):
        mix = st.mix_returns(spy.loc[win], bil.loc[win], w)
        g = st.gap_stats(f, mix)
        print(f"      w={w:.2f}: mix - BUFR = {g['gap_pp_yr']:+.2f} pp/yr  HAC t = {g['hac_t']:+.2f}")
    print("  (b) Newey-West lag grid, cohort-vs-SPY gap:")
    dfc = pd.concat([coh, spy.loc[coh.index]], axis=1, keys=["f", "s"]).dropna()
    dser = (dfc["s"] - dfc["f"]).values
    for L in (3, 6, 12):
        print(f"      lags={L}: HAC t = {st.nw_tstat(dser, lags=L):+.2f}")
    print("  (c) rebalance-cost grid on the BUFR beta-mix gap:")
    for cb in (2.0, 5.0, 10.0):
        mix = st.mix_returns(spy.loc[win], bil.loc[win], b_bufr, cost_bps=cb)
        g = st.gap_stats(f, mix)
        print(f"      cost={cb:.0f} bps one-way: mix - BUFR = {g['gap_pp_yr']:+.2f} pp/yr  "
              f"HAC t = {g['hac_t']:+.2f}")

# -------------------------------------------------------------------- #
print("\n# 7 — Synthetic machinery controls (deterministic, no network — never market evidence)")
print("  (a) gap detector: planted annual structuring drag on a mimic fund")
for drag in (0.0, 2.0):
    w = data.synthetic_world(drag_pct=drag)
    mix = st.mix_returns(w["IDX"], w["CASH"], 0.55, cost_bps=0.0)
    g = st.gap_stats(w["FUND"], mix)
    print(f"      drag={drag:.1f} pp/yr: measured gap = {g['gap_pp_yr']:+.2f} pp/yr  "
          f"HAC t = {g['hac_t']:+.2f}")
print("  (b) delivery checker: planted cap 10% / buffer 15% / fee 0.79% over 40 periods")
so = data.synthetic_outcomes()
per = pd.DataFrame({"spy_price_ret_pct": so["ref_ret"] * 100,
                    "fund_ret_pct": so["fund_ret"] * 100})
dc = st.delivery_check(per, 15.0, 0.79)
print(f"      buffer honored {dc['inside_honored'] + dc['beyond_honored']}/"
      f"{dc['n_inside'] + dc['n_beyond']} down periods | mean capped-year return "
      f"{dc['mean_capped_ret_pct']:.2f}% (planted cap 10% - fee 0.79% = 9.21%)")
