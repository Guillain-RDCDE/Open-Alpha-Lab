"""Real-tape verification — Study 1017 (Do Bull Markets Die of Old Age?). Regenerates docs/results.md.

Dates every US bull market since 1926 on the Fama-French monthly **total-return** market with the
conventional ±20% rule (plus 15%, 25% and an asymmetric filter), fits a censored Weibull duration
model to the bulls, and then asks the only question that matters: do real bulls age any faster than
the bulls of a **random walk** (and of a GARCH(1,1)-t) with the tape's own drift and volatility,
dated by the very same rule? It then tests whether the real-time age of a bull predicts the next
12 months' return or drawdown (Newey-West plus a simulated null for the persistent regressor), and
books a "de-risk old bulls" rule against buy-and-hold with one month of lag, costs and bills as
cash. The daily S&P 500 price index (1990→2022) is the out-of-sample cross-check.

    python studies/1017-bull-markets-old-age/examples/verify.py

Offline: both tapes are bundled and SHA-pinned via quantlab.bundled. Runs in ~1–2 minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from oldage import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

N_SIMS = 1000          # null paths per null model, main threshold
N_SIMS_SENS = 400      # null paths per null model, sensitivity thresholds
N_BOOT = 2000          # cycle bootstrap for the Weibull shape
COST_BPS = 10.0        # one-way, on traded NAV
SEED = 1017
TD_PER_MONTH = 21.0    # trading days per month, for the daily cross-check


def _fmt_date(d) -> str:
    return "—" if pd.isna(d) else pd.Timestamp(d).strftime("%Y-%m")


def daily_rf_from_monthly(rf_m: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Spread each month's T-bill return evenly (geometrically) over that month's trading days."""
    per = pd.Series(idx, index=idx).dt.to_period("M")
    counts = per.value_counts()
    m = rf_m.copy()
    m.index = m.index.to_period("M")
    out = np.full(len(idx), np.nan)
    for j, p in enumerate(per.to_numpy()):
        if p in m.index:
            out[j] = (1.0 + m.loc[p]) ** (1.0 / counts.loc[p]) - 1.0
    return pd.Series(out, index=idx)


def report() -> dict:
    t0 = time.time()
    ff = data.load_ff_monthly()
    sp = data.load_sp500_daily()
    lv = data.total_return_index(ff["mkt"])
    h: dict = {"as_of": data.AS_OF, "as_of_ff": data.AS_OF_FF, "as_of_sp": data.AS_OF_SP,
               "fp_ff": data.fingerprint(ff), "fp_sp": data.fingerprint(sp),
               "provenance": data.provenance(), "cost_bps": COST_BPS,
               "n_sims": N_SIMS, "n_sims_sens": N_SIMS_SENS, "n_boot": N_BOOT}
    h["n_years"] = len(ff) / 12.0
    h["ff_first"], h["ff_last"] = _fmt_date(ff.index[0]), _fmt_date(ff.index[-1])
    h["sp_first"] = sp.index[0].strftime("%Y-%m-%d")
    h["sp_last"] = sp.index[-1].strftime("%Y-%m-%d")
    print(f"FF monthly total return {h['ff_first']} → {h['ff_last']} ({len(ff)} months)  "
          f"fingerprint {h['fp_ff']}")
    print(f"S&P 500 daily price     {h['sp_first']} → {h['sp_last']} ({len(sp)} days)  "
          f"fingerprint {h['fp_sp']}")

    # ------------------------------------------------------------------ 1. dating
    print("\n=== 1. the bull markets (±20%, FF monthly total return) ===")
    cy = st.date_cycles(lv, 0.20)
    bulls = cy[cy["phase"] == "bull"]
    h["bulls"] = [{"start": _fmt_date(r.start), "peak": _fmt_date(r.end),
                   "months": int(r.duration), "gain": float(r.amplitude),
                   "confirmed": _fmt_date(r.confirmed), "censored": bool(r.censored)}
                  for r in bulls.itertuples()]
    bears = cy[(cy["phase"] == "bear") & ~cy["censored"]]
    h["bear_median_months"] = float(bears["duration"].median())
    for b in h["bulls"]:
        print(f"  {b['start']} → {b['peak']}  {b['months']:4d} months  {b['gain']:+8.1%}"
              f"{'  (still running at tape end)' if b['censored'] else ''}")
    lag = (bulls["confirmed"] - bulls["end"]).dropna().dt.days / 30.44
    h["confirm_lag_median"] = float(lag.median())
    print(f"  median lag from the peak to the bar that CONFIRMS it: "
          f"{h['confirm_lag_median']:.1f} months")

    spc = st.date_cycles(sp, 0.20)
    spb = spc[spc["phase"] == "bull"]
    h["sp_bulls"] = [{"start": r.start.strftime("%Y-%m-%d"), "peak": r.end.strftime("%Y-%m-%d"),
                      "months": float(r.duration / TD_PER_MONTH), "gain": float(r.amplitude),
                      "censored": bool(r.censored)} for r in spb.itertuples()]
    print("  S&P 500 daily price index, ±20%:")
    for b in h["sp_bulls"]:
        print(f"    {b['start']} → {b['peak']}  {b['months']:6.1f} months  {b['gain']:+7.1%}")

    # ------------------------------------------------------------------ 2-3. hazard + nulls
    print("\n=== 2-3. hazard, Weibull, and the nulls (±20%) ===")
    gp = st.fit_garch(ff["mkt"])
    h["garch"] = gp
    main = st.duration_test(ff["mkt"], 0.20, n_sims=N_SIMS, n_boot=N_BOOT, seed=SEED,
                            garch_params=gp, rf=ff["rf"])
    fit, boot = main["fit"], main["boot"]
    h.update({"k": fit["k"], "lam": fit["lam"], "weib_median": fit["median"],
              "p_k1": fit["p_k1"], "k_lo": boot["k_lo"], "k_hi": boot["k_hi"],
              "share_boot_k_gt_1": boot["share_k_gt_1"], "n_bulls": main["n_bulls"],
              "n_censored": fit["n_censored"], "median_bull": main["median_bull"]})
    rw, ga = main["nulls"]["rw"], main["nulls"]["garch"]
    h.update({"k_null_rw": rw["k_null_median"], "k_null_rw_lo": rw["k_null_lo"],
              "k_null_rw_hi": rw["k_null_hi"], "p_k_rw": rw["p_k"],
              "k_null_garch": ga["k_null_median"], "k_null_garch_lo": ga["k_null_lo"],
              "k_null_garch_hi": ga["k_null_hi"], "p_k_garch": ga["p_k"],
              "n_bulls_null_rw": rw["n_bulls_null_median"],
              "n_bulls_null_garch": ga["n_bulls_null_median"],
              "median_bull_null_rw": rw["median_bull_null"],
              "median_bull_null_garch": ga["median_bull_null"],
              "share_rw_k_gt_1": float((rw["table"]["k"] > 1).mean()),
              "share_garch_k_gt_1": float((ga["table"]["k"] > 1).mean()),
              "rw_mu": rw["params"]["mu"], "rw_sigma": rw["params"]["sigma"]})
    # how often does the random walk alone yield a "significant" k>1 by the naive LR test?
    naive = []
    for p in st.simulate_log_returns("rw", len(ff), 300, rw["params"], seed=SEED + 7):
        d_, c_ = st._bulls_from_tps(st.turning_points(np.exp(np.cumsum(p)), 0.20), len(p))
        f_ = st.fit_weibull(d_, c_)
        if f_:
            naive.append(f_["k"] > 1 and f_["p_k1"] < 0.05)
    h["rw_naive_reject"] = float(np.mean(naive))
    life = main["life"].copy()
    life["rw_hazard"] = rw["life"]["annual_hazard"]
    life["garch_hazard"] = ga["life"]["annual_hazard"]
    h["life"] = life.reset_index().to_dict("records")
    print(life[["at_risk", "ended", "censored", "annual_hazard", "rw_hazard",
                "garch_hazard"]].round(3).to_string())
    print(f"  real: {h['n_bulls']} completed bulls (+{h['n_censored']} running), "
          f"Weibull k = {h['k']:.2f}  [{h['k_lo']:.2f}, {h['k_hi']:.2f}]  "
          f"LR test of k=1 p = {h['p_k1']:.2f}")
    print(f"  random walk (μ={h['rw_mu']:.4f}, σ={h['rw_sigma']:.4f}/month): median k "
          f"{h['k_null_rw']:.2f} [{h['k_null_rw_lo']:.2f}, {h['k_null_rw_hi']:.2f}], "
          f"{h['share_rw_k_gt_1']:.0%} of paths have k>1  -> p = {h['p_k_rw']:.3f}")
    print(f"  GARCH(1,1)-t:                             median k "
          f"{h['k_null_garch']:.2f} [{h['k_null_garch_lo']:.2f}, {h['k_null_garch_hi']:.2f}]"
          f"  -> p = {h['p_k_garch']:.3f}")
    print(f"  the naive LR test calls a random walk 'dies of old age' on "
          f"{h['rw_naive_reject']:.0%} of paths")
    h["_null_k_rw"] = rw["table"]["k"].to_numpy()
    h["_null_k_garch"] = ga["table"]["k"].to_numpy()

    # ------------------------------------------------------------------ 4. sensitivity
    print("\n=== 4. sensitivity to the dating rule ===")
    sens = []
    for label, up, down in (("±15%", 0.15, 0.15), ("±20%", 0.20, 0.20), ("±25%", 0.25, 0.25),
                            ("+20% / −15% (asymmetric)", 0.20, 0.15)):
        if up == 0.20 and down == 0.20:
            res = main
        else:
            res = st.duration_test(ff["mkt"], up, down, n_sims=N_SIMS_SENS, n_boot=500,
                                   seed=SEED, with_prediction=False, garch_params=gp,
                                   rf=ff["rf"])
        f = res["fit"]
        row = {"rule": label, "n_bulls": res["n_bulls"], "median": res["median_bull"],
               "k": f.get("k", np.nan), "k_lo": res["boot"].get("k_lo", np.nan),
               "k_hi": res["boot"].get("k_hi", np.nan),
               "k_rw": res["nulls"]["rw"]["k_null_median"], "p_rw": res["nulls"]["rw"]["p_k"],
               "k_garch": res["nulls"]["garch"]["k_null_median"],
               "p_garch": res["nulls"]["garch"]["p_k"]}
        sens.append(row)
        print(f"  {label:26s} n={row['n_bulls']:2d} median {row['median']:5.0f}m  "
              f"k={row['k']:.2f} [{row['k_lo']:.2f},{row['k_hi']:.2f}]  "
              f"RW k {row['k_rw']:.2f} p={row['p_rw']:.2f}  "
              f"GARCH k {row['k_garch']:.2f} p={row['p_garch']:.2f}")
    h["sens"] = sens
    h["sens_min_p"] = float(min(min(r["p_rw"], r["p_garch"]) for r in sens))
    h["sens_txt"] = (f"Across the 15%, 20%, 25% and asymmetric rules the smallest single-null "
                     f"p-value is {h['sens_min_p']:.2f}, and the real k never clears both nulls.")

    # ------------------------------------------------------------------ 5. prediction
    print("\n=== 5. does the age of the bull predict the next 12 months? ===")
    pr = main["pred_raw"]
    h.update({"pred_slope": pr["slope"], "pred_t": pr["t"], "pred_n": pr["n"],
              "pred_t_nov": pr["t_nonoverlap"], "pred_n_nov": pr["n_nonoverlap"],
              "pred_r2": pr["r2"], "p_pred_rw": rw["p_pred"], "p_pred_garch": ga["p_pred"],
              "pred_t_null_lo_rw": rw["pred_t_null_lo"],
              "pred_t_null_med_rw": rw["pred_t_null_median"]})
    dd = st.predictive_regression(ff["mkt"], ff["rf"], target="drawdown")
    h.update({"dd_slope": dd["slope"], "dd_t": dd["t"], "dd_t_nov": dd["t_nonoverlap"]})
    print(f"  next-12m excess log return on age (years): slope {h['pred_slope']:+.4f}, "
          f"NW t {h['pred_t']:+.2f} (n={h['pred_n']}), non-overlapping t "
          f"{h['pred_t_nov']:+.2f} (n={h['pred_n_nov']})")
    print(f"  simulated p (RW) {h['p_pred_rw']:.2f}, (GARCH) {h['p_pred_garch']:.2f};  "
          f"RW null t: median {h['pred_t_null_med_rw']:+.2f}, 5th pct "
          f"{h['pred_t_null_lo_rw']:+.2f}")
    print(f"  next-12m max drawdown on age: slope {h['dd_slope']:+.4f}, NW t {h['dd_t']:+.2f}, "
          f"non-overlapping t {h['dd_t_nov']:+.2f}")
    # conditional one-month excess return: old vs young bull (real-time, causal)
    W = st.age_rule_weights(lv, 0.20)
    nxt = ff["mkt_rf"].shift(-1)
    in_bull = W["age"].notna() & W["median_age"].notna()
    old = in_bull & (W["age"] > W["median_age"])
    young = in_bull & (W["age"] <= W["median_age"])
    rest = ~in_bull & W["median_age"].notna()
    cond = []
    for name, m in (("young bull (age ≤ median)", young), ("old bull (age > median)", old),
                    ("bear / unconfirmed", rest)):
        x = nxt[m].dropna()
        cond.append({"state": name, "months": int(len(x)), "mean": float(x.mean() * 12),
                     "vol": float(x.std() * np.sqrt(12)),
                     "sharpe": float(x.mean() / x.std() * np.sqrt(12))})
    h["cond"] = cond
    diff = st.ols_hac(nxt[young | old].to_numpy(), old[young | old].astype(float).to_numpy(), 6)
    h["old_minus_young"] = diff["slope"] * 12
    h["old_minus_young_t"] = diff["t"]
    for c in cond:
        print(f"  next-month excess, {c['state']:28s} {c['months']:4d} months  mean "
              f"{c['mean']:+6.2%}/yr  vol {c['vol']:.1%}  Sharpe {c['sharpe']:.2f}")
    print(f"  old minus young: {h['old_minus_young']:+.2%}/yr, NW t {h['old_minus_young_t']:+.2f}")

    # ------------------------------------------------------------------ 6. tradability
    print("\n=== 6. could you trade it? de-risk to 50% once the bull outlives its median ===")
    start = W["median_age"].first_valid_index()
    h["rule_start"] = _fmt_date(start)
    rows = []
    for cbps in (0.0, COST_BPS, 25.0):
        bt = st.age_rule_backtest(ff["mkt"], ff["rf"], W["weight"], cbps)
        bt = bt[bt.index > start]
        s = st.summarize_backtest(bt)
        b = st.sharpe_diff_bootstrap(bt["net_x"], bt["bh_x"], block=12, n_boot=2000,
                                     seed=SEED)
        rows.append({"cost_bps": cbps, **s, "sharpe_diff": b["diff"], "sharpe_lo": b["lo"],
                     "sharpe_hi": b["hi"], "p_sharpe": b["p"]})
        if cbps == COST_BPS:
            h.update({"sharpe_bh": s["sharpe_bh"], "sharpe_gross": s["sharpe_gross"],
                      "sharpe_net": s["sharpe_net"], "tw_bh": s["tw_bh"],
                      "tw_net": s["tw_net"], "tw_gross": s["tw_gross"],
                      "cagr_bh": s["cagr_bh"], "cagr_net": s["cagr_net"],
                      "vol_bh": s["vol_bh"], "vol_net": s["vol_net"],
                      "mdd_bh": s["mdd_bh"], "mdd_net": s["mdd_net"],
                      "share_derisked": s["share_derisked"], "avg_weight": s["avg_weight"],
                      "turnover_yr": s["turnover_yr"], "sharpe_diff": b["diff"],
                      "sharpe_lo": b["lo"], "sharpe_hi": b["hi"], "p_sharpe": b["p"],
                      "bt_years": len(bt) / 12.0})
            cm = st.constant_mix(bt["bh"], bt["rf"], s["avg_weight"])
            h["sharpe_cm"] = st.sharpe(cm - bt["rf"])
            h["tw_cm"] = float((1 + cm).prod())
            h["_bt"] = bt
    h["bt_rows"] = rows
    for r in rows:
        print(f"  cost {r['cost_bps']:4.0f}bp: Sharpe rule {r['sharpe_net']:.3f} vs B&H "
              f"{r['sharpe_bh']:.3f}  diff {r['sharpe_diff']:+.3f} [{r['sharpe_lo']:+.3f}, "
              f"{r['sharpe_hi']:+.3f}] p={r['p_sharpe']:.3f}  TW ${r['tw_net']:,.0f} vs "
              f"${r['tw_bh']:,.0f}")
    print(f"  exposure-matched constant mix ({h['avg_weight']:.0%} equity): Sharpe "
          f"{h['sharpe_cm']:.3f}, TW ${h['tw_cm']:,.0f}")

    # daily S&P cross-check: median seeded by FF bulls completed before 1990
    sp_m = sp[sp.index <= pd.Timestamp(data.AS_OF_FF)]
    prior = [b["months"] * TD_PER_MONTH for b in h["bulls"]
             if not b["censored"] and pd.Timestamp(b["confirmed"]) < pd.Timestamp("1990-01-01")]
    Wd = st.age_rule_weights(sp_m, 0.20, prior_durations=prior)
    r_d = sp_m.pct_change().dropna()
    rf_d = daily_rf_from_monthly(ff["rf"], r_d.index).fillna(0.0)
    btd = st.age_rule_backtest(r_d, rf_d, Wd["weight"], COST_BPS)
    sd = st.summarize_backtest(btd, periods=252)
    bd = st.sharpe_diff_bootstrap(btd["net_x"], btd["bh_x"], periods=252, block=63,
                                  n_boot=1000, seed=SEED)
    h.update({"sp_sharpe_bh": sd["sharpe_bh"], "sp_sharpe_net": sd["sharpe_net"],
              "sp_sharpe_diff": bd["diff"], "sp_sharpe_lo": bd["lo"], "sp_sharpe_hi": bd["hi"],
              "sp_p": bd["p"], "sp_tw_bh": sd["tw_bh"], "sp_tw_net": sd["tw_net"],
              "sp_share_derisked": sd["share_derisked"],
              "sp_prior_median_months": float(np.median(prior) / TD_PER_MONTH)})
    print(f"  S&P daily (price-only, 1990-01 → {data.AS_OF_FF[:7]}, median seeded with "
          f"{h['sp_prior_median_months']:.0f}m from pre-1990 FF bulls): Sharpe "
          f"{sd['sharpe_net']:.3f} vs {sd['sharpe_bh']:.3f}, diff {bd['diff']:+.3f} "
          f"[{bd['lo']:+.3f}, {bd['hi']:+.3f}]  TW {sd['tw_net']:.2f} vs {sd['tw_bh']:.2f}")

    # ------------------------------------------------------------------ 7. synthetic control
    print("\n=== 7. machinery proof: planted ageing vs matched null (synthetic) ===")
    ctrl = []
    for s_ in (1.0, 0.0):
        dfs, tr = data.synthetic_monthly(n_months=len(ff), signal_strength=s_, seed=SEED)
        rs = st.duration_test(dfs["mkt"], 0.20, n_sims=200, n_boot=200, kinds=("rw",),
                              seed=SEED, rf=dfs["rf"])
        ctrl.append({"signal_strength": s_, "n_bulls": rs["n_bulls"], "k": rs["fit"]["k"],
                     "k_null": rs["nulls"]["rw"]["k_null_median"],
                     "p_k": rs["nulls"]["rw"]["p_k"], "pred_t": rs["pred_raw"]["t"],
                     "p_pred": rs["nulls"]["rw"]["p_pred"], "n_crash": tr["n_crash"]})
        c = ctrl[-1]
        print(f"  signal_strength={s_:.1f}: {c['n_bulls']} bulls, k {c['k']:.2f} vs null "
              f"{c['k_null']:.2f}, p {c['p_k']:.3f}; pred t {c['pred_t']:+.2f}, "
              f"p {c['p_pred']:.3f}")
    h["control"] = ctrl

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = time.time() - t0
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  runtime {h['runtime_s']:.0f}s")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    prov = "\n".join(
        f"| {p['tape']} | {p['source']} | `{p['sha256'][:16]}…` | {p['label']} | {p['as_of']} |"
        for p in h["provenance"])
    bulls = "\n".join(
        f"| {b['start']} | {b['peak']} | {b['months']} | {b['gain']:+.0%} | {b['confirmed']} |"
        f"{' running (censored)' if b['censored'] else ''} |" for b in h["bulls"])
    spb = "\n".join(
        f"| {b['start']} | {b['peak']} | {b['months']:.0f} | {b['gain']:+.0%} |"
        f"{' running at tape end' if b['censored'] else ''} |" for b in h["sp_bulls"])
    life = "\n".join(
        f"| {r['bucket']} | {r['at_risk']} | {r['ended']} | {r['censored']} | "
        f"**{r['annual_hazard']:.0%}** | {r['rw_hazard']:.0%} | {r['garch_hazard']:.0%} |"
        for r in h["life"])
    sens = "\n".join(
        f"| {r['rule']} | {r['n_bulls']} | {r['median']:.0f} | **{r['k']:.2f}** | "
        f"{r['k_lo']:.2f}–{r['k_hi']:.2f} | {r['k_rw']:.2f} | {r['p_rw']:.2f} | "
        f"{r['k_garch']:.2f} | {r['p_garch']:.2f} |" for r in h["sens"])
    cond = "\n".join(
        f"| {c['state']} | {c['months']} | {c['mean']:+.2%} | {c['vol']:.1%} | "
        f"{c['sharpe']:.2f} |" for c in h["cond"])
    bt = "\n".join(
        f"| {r['cost_bps']:.0f} bp | {r['sharpe_bh']:.3f} | {r['sharpe_gross']:.3f} | "
        f"{r['sharpe_net']:.3f} | **{r['sharpe_diff']:+.3f}** ({r['sharpe_lo']:+.3f} to "
        f"{r['sharpe_hi']:+.3f}) | {r['p_sharpe']:.3f} | ${r['tw_bh']:,.0f} | "
        f"${r['tw_net']:,.0f} |" for r in h["bt_rows"])
    ctrl = "\n".join(
        f"| {c['signal_strength']:.1f} | {c['n_bulls']} | {c['k']:.2f} | {c['k_null']:.2f} | "
        f"{c['p_k']:.3f} | {c['pred_t']:+.2f} | {c['p_pred']:.3f} |" for c in h["control"])
    g = h["garch"]
    return f"""# Results — Study 1017 (Do Bull Markets Die of Old Age?) on the real tapes

*Generated by [`examples/verify.py`](../examples/verify.py). As-of **{h['as_of']}** (FF monthly to
**{h['as_of_ff']}**, S&P daily to **{h['as_of_sp']}** — the partial December 2022 is dropped).
Fingerprints: FF `{h['fp_ff']}`, S&P `{h['fp_sp']}`. {h['n_sims']:,} null paths per null model
({h['n_sims_sens']} for the sensitivity rows), {h['n_boot']:,} cycle-bootstrap draws.*

## 0. Data and provenance

| Tape | Package · file | SHA-256 pin | Label | As-of |
|---|---|---|---|---|
{prov}

Both tapes are read offline through `quantlab.bundled`, which refuses any file whose bytes differ
from the pin. The main tape is the Fama-French value-weighted market **total return**
(`Mkt-RF + RF`, dividends in) {h['ff_first']} → {h['ff_last']}, {h['n_years']:.0f} years — the
only sample long enough to hold more than a handful of bull markets. The S&P 500 tape is a **price
index** (no dividends): an out-of-sample cross-check with daily resolution and few cycles.

## 1. The bull markets

Conventional rule: a bull ends when the index falls 20% below its high; a bear ends when it rises
20% above its low; turning points are the extremes in between. Monthly closes on the total-return
index. The first partial phase (from the July 1926 sample start) is dropped — its start is unknown.

| Trough | Peak | Months | Gain | Peak confirmed |
|---|---|--:|--:|---|
{bulls}

**{h['n_bulls']} completed bulls** in {h['n_years']:.0f} years (plus the one still running in
November 2018, right-censored). That is the whole sample — and the honest constraint on everything
below. The median completed bull ran **{h['median_bull']:.0f} months**; the median bear
{h['bear_median_months']:.0f}. A peak is only *known* once the index is 20% below it: the median
confirmation came **{h['confirm_lag_median']:.1f} months** after the top.

Daily S&P 500 price index, same rule (durations in months of 21 trading days):

| Trough | Peak | Months | Gain | |
|---|---|--:|--:|---|
{spb}

## 2. The hazard by age — real bulls against random-walk bulls

Life table: the probability that a bull which has reached a given age ends within the following
year. The real column rests on {h['n_bulls'] + h['n_censored']} spells; the null columns pool
{h['n_sims']:,} simulated tapes of the same length dated with the same rule.

| Bull age | At risk | Ended | Censored | Real: P(end within a year) | Random walk | GARCH(1,1)-t |
|---|--:|--:|--:|--:|--:|--:|
{life}

The real hazard does rise from the first year to the third — exactly the shape the believers
point to. But a dating rule that needs a 20% move to start and a 20% fall to end a bull also gives
a random walk a low first-year hazard. The real hazard after year three is *flat to falling*, not
rising.

## 3. The Weibull shape against the null

| | Shape k | 90% band / 95% CI | p (H₁: real ages faster) |
|---|--:|--:|--:|
| **Real bulls (±20%)** | **{h['k']:.2f}** | {h['k_lo']:.2f}–{h['k_hi']:.2f} (cycle bootstrap, 95%) | — |
| Naive test of k = 1 (likelihood ratio) | | | {h['p_k1']:.2f} |
| Random walk, μ = {h['rw_mu']:.4f}, σ = {h['rw_sigma']:.4f} / month | {h['k_null_rw']:.2f} | {h['k_null_rw_lo']:.2f}–{h['k_null_rw_hi']:.2f} | **{h['p_k_rw']:.2f}** |
| GARCH(1,1)-t (α = {g['alpha']:.3f}, β = {g['beta']:.3f}, ν = {g['nu']:.1f}), same drift | {h['k_null_garch']:.2f} | {h['k_null_garch_lo']:.2f}–{h['k_null_garch_hi']:.2f} | **{h['p_k_garch']:.2f}** |

Two things are true at once. **The dating rule manufactures ageing**: {h['share_rw_k_gt_1']:.0%}
of pure random walks dated this way show k > 1 (median {h['k_null_rw']:.2f}), and the naive
likelihood-ratio test would call a memoryless random walk "dies of old age" on
{h['rw_naive_reject']:.0%} of paths. And **real bulls age no faster than that**: the real k of
{h['k']:.2f} sits *below* the random-walk median, at p = {h['p_k_rw']:.2f}. Against GARCH — where
volatility clustering produces long calm bulls and a falling hazard (median k
{h['k_null_garch']:.2f}) — the real k is higher, but at p = {h['p_k_garch']:.2f}, nowhere near a
test. Real bulls are fewer and longer than random-walk bulls ({h['n_bulls']} vs a median
{h['n_bulls_null_rw']:.0f} per tape; median length {h['median_bull']:.0f} vs
{h['median_bull_null_rw']:.0f} months), closer to the GARCH world ({h['n_bulls_null_garch']:.0f}
bulls, median {h['median_bull_null_garch']:.0f} months) — long, calm expansions are a
volatility-regime fact, not an ageing fact.

## 4. Sensitivity to the dating rule

| Rule | Completed bulls | Median (months) | Real k | 95% CI | RW null k | p (RW) | GARCH null k | p (GARCH) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{sens}

The asymmetric row ends bulls on a 15% fall but needs a 20% rise to start one, a filter in the
spirit of Lunde & Timmermann (2004). No rule produces real ageing beyond both nulls.

## 5. Does the age of the bull predict the next 12 months?

Real-time age (months since the trough that started the current *confirmed* bull, using only data
to date), in years; months in a confirmed bull only; overlapping 12-month windows.

| Target | Slope per year of age | Newey-West t (18 lags) | Non-overlapping t | Simulated p (RW / GARCH) |
|---|--:|--:|--:|--:|
| Next-12m excess log return | {h['pred_slope']:+.2%} | **{h['pred_t']:+.2f}** | {h['pred_t_nov']:+.2f} (n = {h['pred_n_nov']}) | {h['p_pred_rw']:.2f} / {h['p_pred_garch']:.2f} |
| Next-12m max drawdown | {h['dd_slope']:+.2%} | {h['dd_t']:+.2f} | {h['dd_t_nov']:+.2f} | — |

The sign leans the believers' way and the size is nothing: a decade of extra age moves the
expected next-year return by about {10 * h['pred_slope']:+.0%}, with t = {h['pred_t']:+.2f}
on {h['pred_n']} overlapping months ({h['pred_n_nov']} independent ones). Age is a trending,
persistent regressor, so the HAC t is checked against the same regression on random-walk paths: a
random walk yields a t at or below the real one {h['p_pred_rw']:.0%} of the time (its median t is
{h['pred_t_null_med_rw']:+.2f}, its 5th percentile {h['pred_t_null_lo_rw']:+.2f}).

What a de-risker would actually give up — next-month excess return by real-time state (from
{h['rule_start']}, when three completed bulls first exist to set a median):

| State at month-end | Months | Mean excess (ann.) | Vol | Sharpe |
|---|--:|--:|--:|--:|
{cond}

Old bulls minus young bulls: **{h['old_minus_young']:+.2%} a year** (Newey-West t =
{h['old_minus_young_t']:+.2f}). Late-cycle months did not pay less.

## 6. Could you trade it?

Rule: hold 100% equity, cut to **50% (rest in T-bills)** once the current bull is older than the
median completed bull known at that month-end. One month of lag (weight set at the close of *t*
earns *t+1*), costs one-way × traded NAV, excess-of-cash Sharpe for both legs. Window: from
{h['rule_start']} (the rule's first live month) to {h['as_of_ff'][:7]}, {h['bt_years']:.0f} years,
FF total return.

| Cost | Sharpe B&H | Sharpe rule (gross) | Sharpe rule (net) | Δ net − B&H (95% CI) | p (rule better) | $1 B&H | $1 rule (net) |
|---|--:|--:|--:|--:|--:|--:|--:|
{bt}

The rule is de-risked in {h['share_derisked']:.0%} of months (average equity weight
{h['avg_weight']:.0%}, one-way turnover {h['turnover_yr']:.0%} of NAV a year). It cuts vol from
{h['vol_bh']:.1%} to {h['vol_net']:.1%} and trims the worst drawdown only from {h['mdd_bh']:.0%}
to {h['mdd_net']:.0%} (old bulls do not end in worse crashes than young ones), and it
**{'lowers' if h['sharpe_diff'] < 0 else 'raises'}** the Sharpe ratio: a costless constant mix with the same
average exposure earns Sharpe {h['sharpe_cm']:.3f} and ${h['tw_cm']:,.0f}, so the timing itself
subtracts value. Out of sample, on the daily S&P 500 price index 1990 → {h['as_of_ff'][:7]} (median
age seeded with the {h['sp_prior_median_months']:.0f}-month median of FF bulls confirmed before
1990, cash = daily-ised T-bill): Sharpe {h['sp_sharpe_net']:.3f} net vs {h['sp_sharpe_bh']:.3f}
buy-and-hold, Δ {h['sp_sharpe_diff']:+.3f} ({h['sp_sharpe_lo']:+.3f} to {h['sp_sharpe_hi']:+.3f});
$1 → ${h['sp_tw_net']:.2f} vs ${h['sp_tw_bh']:.2f} (price only, dividends excluded on both legs).

## 7. Machinery proof — the harness can see ageing when it is there

Synthetic tape, same length and similar drift/vol, from `data.synthetic_monthly`. At
`signal_strength = 1` bulls older than three years face a planted, rising monthly probability of a
bull-ending crash; at `0` the tape is exactly the random-walk null. Same code path as the real run
(random-walk null only, 200 paths). **Synthetic — not market evidence.**

| signal_strength | Bulls | Real-path k | Null k | p (k) | Prediction t | p (pred) |
|---|--:|--:|--:|--:|--:|--:|
{ctrl}

The planted world is flagged on both tests; the matched null is not. A null result on the real
tape is therefore a finding about markets, not a blind harness.

## Caveats

- **{h['n_bulls']} completed bull markets is the whole sample.** The Weibull CI on the real k
  ({h['k_lo']:.2f}–{h['k_hi']:.2f}) is wide enough to contain both "memoryless" and "ages
  briskly". The null comparison does not create information; it only stops the dating rule from
  being mistaken for a finding. No test on this topic can be powerful with ~12 cycles a century.
- **Monthly closes smooth intramonth extremes.** October 1987 or March 2020 bottoms fall between
  month-ends, which moves some turning points by a month and merges or splits a few short phases
  (the 1932 and 1940–42 episodes sit close to the 20% line). The 15%/25% rows bracket this.
- **The rule is the conventional newspaper one, not a business-cycle algorithm.** We did not
  implement the Pagan–Sossounov (2003) Bry–Boschan-style censoring rules (minimum phase and cycle
  lengths); they would remove the shortest phases, which mechanically raises k on real *and* null
  tapes alike.
- **The S&P cross-check is a price index** (no dividends) and contains only
  {sum(1 for b in h['sp_bulls'] if not b['censored'])} completed bulls since 1990, two of them
  short bear-market rallies; it is a robustness check of the trading rule, not a second hazard estimate.
- **Bull "age" in the predictive regression is real-time** and starts only once a bull is
  confirmed — up to several months after the trough. Ex-post dated ages would add look-ahead.
- **The nulls are monthly i.i.d. lognormal and GARCH(1,1)-t** with the tape's drift. Neither has
  time-varying expected returns (valuation mean reversion), which is the most plausible channel
  for genuine late-cycle weakness — and which shows up in neither the hazard nor the 12-month
  regression here.

## Verdict

Produced by `strategy.verdict`, thresholds fixed before the real tape was run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1017-bull-markets-old-age](../README.md). Not investment advice.*
"""


def headline(h: dict) -> dict:
    v = h["_verdict"]
    keys = ["as_of", "fp_ff", "fp_sp", "n_bulls", "k", "k_lo", "k_hi", "k_null_rw",
            "k_null_garch", "p_k_rw", "p_k_garch", "pred_slope", "pred_t", "p_pred_rw",
            "p_pred_garch", "sharpe_bh", "sharpe_net", "sharpe_diff", "p_sharpe", "tw_bh",
            "tw_net", "sp_sharpe_diff", "runtime_s"]
    out = {k: h[k] for k in keys}
    out.update({"signal": v["signal"], "trad": v["trad"], "one_sentence": v["one_sentence"]})
    return out


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    print("##HEADLINE## " + json.dumps(headline(h), default=float))


if __name__ == "__main__":
    main()
