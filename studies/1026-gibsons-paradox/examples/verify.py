"""Real-tape verification — Study 1026 (Gibson's Paradox). Regenerates docs/results.md.

Puts the claim "long-term yields follow the price level, not inflation" through the gates a
spurious correlation fails: unit roots, Engle-Granger cointegration, a first-difference horse
race with Newey-West errors, an adaptive-expectations confound (Fisher's own resolution), an
expanding-window out-of-sample forecast with Clark-West, and a duration overlay on a 20-year AAA
par bond with costs. US monthly (Moody's AAA + core CPI, 1957-2018) is the primary tape; US
quarterly headline CPI (1959-2009) and Germany (1972-98) are the checks. Ends with the synthetic
two-world control and the verdict produced by ``strategy.verdict``.

    python studies/1026-gibsons-paradox/examples/verify.py

Everything is read from tapes frozen inside ``arch`` / ``statsmodels`` (no network).
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gibson import data, strategy as st  # noqa: E402

warnings.simplefilter("ignore")
DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

COST_BPS = 5.0          # one-way, per unit of NAV traded (long-bond futures / ETF overlay)
COST_GRID = (0.0, 2.0, 5.0, 10.0, 25.0, 50.0)
EWMA_HEADLINE = 36      # months; the middle of the grid below, reported in full
EWMA_GRID = (12, 36, 60, 120)
H_MONTHLY = (3, 12)
MIN_TRAIN_M = 120
MIN_TRAIN_Q = 40
SPLITS = {"1965-81": data.GREAT_INFLATION, "1982-2018": data.DISINFLATION}


def _f(x, fmt="{:+.2f}"):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def report() -> dict:
    t0 = time.time()
    us = data.load_us_monthly()
    uq = data.load_us_quarterly()
    de = data.load_germany()
    ewma = st.adaptive_expectation(us["pi_1m"], EWMA_HEADLINE)
    us["ewma"] = ewma
    h: dict = {"as_of": data.AS_OF, "cost_bps": COST_BPS, "ewma_hl": float(EWMA_HEADLINE)}
    h["tapes"] = [
        {"tape": "US monthly (AAA + core CPI + RF)", "rows": len(us),
         "first": str(us.index[0].date()), "last": str(us.index[-1].date()),
         "fingerprint": data.fingerprint(us[["aaa", "baa", "cpi", "rf"]])},
        {"tape": "US quarterly (AAA + headline CPI, macrodata)", "rows": len(uq),
         "first": str(uq.index[0].date()), "last": str(uq.index[-1].date()),
         "fingerprint": data.fingerprint(uq[["aaa", "cpi"]])},
        {"tape": "Germany quarterly (R + Dp)", "rows": len(de),
         "first": str(de.index[0].date()), "last": str(de.index[-1].date()),
         "fingerprint": data.fingerprint(de[["R", "Dp"]])},
    ]
    h["provenance"] = data.provenance()
    print(f"as-of {data.AS_OF}")
    for r in h["tapes"]:
        print(f"  {r['tape']:45s} {r['rows']:4d} rows {r['first']} -> {r['last']}  "
              f"fp {r['fingerprint']}")

    # ---------------------------------------------------------------- 1 levels
    print("\n=== 1. levels (descriptive) ===")
    cols = ("logp", "gap_exp", "gap_rol", "pi", "ewma")
    lv = []
    for name, (a, b) in {"1957-2018": (None, None), **SPLITS}.items():
        c = st.level_correlations(us, "aaa", cols, start=a, end=b)
        lv.append({"tape": "US monthly", "period": name, **c})
    lv.append({"tape": "US quarterly (headline)", "period": "1959-2009",
               **st.level_correlations(uq, "aaa", cols[:4])})
    lv.append({"tape": "Germany", "period": "1972-98",
               **st.level_correlations(de, "R", cols[:4])})
    h["levels"] = lv
    for r in lv:
        print("  ", r)
    h["corr_logp"], h["corr_gap"] = lv[0]["logp"], lv[0]["gap_exp"]
    h["corr_pi"], h["corr_ewma"] = lv[0]["pi"], lv[0]["ewma"]
    h["gi_corr_logp"], h["dis_corr_logp"] = lv[1]["logp"], lv[2]["logp"]
    h["gi_corr_gap"], h["dis_corr_gap"] = lv[1]["gap_exp"], lv[2]["gap_exp"]

    # ---------------------------------------------------------------- 2 unit roots
    print("\n=== 2. unit roots ===")
    ur = st.unit_root_table({
        "US AAA yield": us["aaa"], "US ΔAAA": us["aaa"].diff(),
        "US log core CPI": us["logp"], "US 1m inflation (Δlog CPI)": us["pi_1m"],
        "US 12m inflation": us["pi"], "US price gap (expanding)": us["gap_exp"],
        "US price gap (rolling 10y)": us["gap_rol"], "US adaptive exp. (36m)": us["ewma"],
        "DE long rate": de["R"], "DE 4q inflation": de["pi"],
        "DE price gap (expanding)": de["gap_exp"]})
    print(ur.round(3).to_string())
    h["unit_roots"] = ur.reset_index().to_dict("records")

    # ---------------------------------------------------------------- 3 cointegration
    print("\n=== 3. Engle-Granger ===")
    eg = []
    for lab, x in (("log price level (raw)", us["logp"]), ("price gap (expanding)", us["gap_exp"]),
                   ("price gap (rolling 10y)", us["gap_rol"]), ("12m inflation", us["pi"]),
                   (f"adaptive expectation ({EWMA_HEADLINE}m)", us["ewma"])):
        r = st.engle_granger(us["aaa"], x)
        eg.append({"tape": "US monthly", "x": lab, **r})
    for lab, x in (("price gap (expanding)", de["gap_exp"]), ("4q inflation", de["pi"])):
        eg.append({"tape": "Germany", "x": lab, **st.engle_granger(de["R"], x)})
    for r in eg:
        print(f"  {r['tape']:11s} {r['x']:32s} EG p {r['eg_p']:.3f}  naive t {r['naive_t']:+.1f}"
              f"  R2 {r['r2']:.2f}")
    h["eg"] = eg
    h["eg_p_gap"], h["eg_p_pi"], h["eg_p_ewma"] = eg[1]["eg_p"], eg[3]["eg_p"], eg[4]["eg_p"]
    h["de_eg_p_gap"] = eg[5]["eg_p"]

    # ---------------------------------------------------------------- 4 differences
    print("\n=== 4. first-difference horse race (Newey-West) ===")
    hr = []
    for name, (a, b) in {"1957-2018": (None, None), **SPLITS}.items():
        r = st.diff_horse_race(us, "aaa", "gap_exp", "pi", lags=12, start=a, end=b)
        hr.append({"tape": "US monthly AAA", "period": name, **r})
    hr.append({"tape": "US monthly BAA", "period": "1957-2018",
               **st.diff_horse_race(us, "baa", "gap_exp", "pi", lags=12)})
    hr.append({"tape": "US quarterly (headline)", "period": "1959-2009",
               **st.diff_horse_race(uq, "aaa", "gap_exp", "pi", lags=4)})
    hr.append({"tape": "Germany", "period": "1972-98",
               **st.diff_horse_race(de, "R", "gap_exp", "pi", lags=4)})
    for r in hr:
        print(f"  {r['tape']:24s} {r['period']:10s} b_gap {r['b_gap']:+.3f} (t {r['t_gap']:+.2f})"
              f"  b_pi {r['b_pi']:+.3f} (t {r['t_pi']:+.2f})  n {r['n']}")
    h["horse_race"] = hr
    h["t_gap"], h["t_pi"] = hr[0]["t_gap"], hr[0]["t_pi"]
    h["de_t_gap"], h["de_t_pi"] = hr[-1]["t_gap"], hr[-1]["t_pi"]
    rd = st.regime_difference(us, "aaa", data.GREAT_INFLATION, data.DISINFLATION)
    h["regime_diff"] = rd
    print(f"  gap slope 1982-2018 minus 1965-81: {rd['diff_gap']:+.3f} (t {rd['t_diff_gap']:+.2f})")

    # ---------------------------------------------------------------- 5 long memory
    print("\n=== 5. Fisher with a long memory (adaptive expectations) ===")
    lm = []
    for hl in EWMA_GRID:
        e = st.adaptive_expectation(us["pi_1m"], hl)
        d = us.assign(ew=e)
        r = st.diff_horse_race(d, "aaa", "gap_exp", "ew", lags=12)
        egr = st.engle_granger(us["aaa"], e)
        dd = d[["gap_exp", "ew"]].diff().dropna()
        lm.append({"halflife": hl, "corr_level": float(us["aaa"].corr(e)),
                   "eg_p": egr["eg_p"], "t_gap": r["t_gap"], "t_ewma": r["t_pi"],
                   "corr_changes": float(dd["gap_exp"].corr(dd["ew"]))})
        print(f"  half-life {hl:4d}m  corr {lm[-1]['corr_level']:+.2f}  EG p {egr['eg_p']:.3f}  "
              f"Δ: t_gap {r['t_gap']:+.2f}  t_ewma {r['t_pi']:+.2f}  "
              f"corr(Δgap, Δewma) {lm[-1]['corr_changes']:.2f}")
    h["long_memory"] = lm
    h["t_gap_vs_ewma"] = [r for r in lm if r["halflife"] == EWMA_HEADLINE][0]["t_gap"]

    # ---------------------------------------------------------------- 6 OOS
    print("\n=== 6. out of sample (expanding window, CPI lagged one month) ===")
    lag_us = st.add_publication_lag(us, cols=("gap_exp", "gap_rol", "pi", "ewma"), lag=1)
    oos_rows, fcs = [], {}
    for hh in H_MONTHLY:
        tbl, fc = st.oos_table(lag_us, "aaa", hh, MIN_TRAIN_M,
                               variables=("gap_exp", "gap_rol", "pi", "ewma"))
        for v, r in tbl.iterrows():
            oos_rows.append({"tape": "US monthly", "h": f"{hh}m", "variable": v,
                             **{k: r[k] for k in ("n", "oos_r2_vs_yield_only", "cw_t", "cw_p",
                                                  "yield_only_oos_r2_vs_mean")},
                             "first": str(r["first"].date())})
        fcs[hh] = fc
    lag_de = st.add_publication_lag(de, cols=("gap_exp", "pi"), lag=1)
    for hh in (1, 4):
        tbl, _ = st.oos_table(lag_de, "R", hh, MIN_TRAIN_Q, variables=("gap_exp", "pi"))
        for v, r in tbl.iterrows():
            oos_rows.append({"tape": "Germany", "h": f"{hh}q", "variable": v,
                             **{k: r[k] for k in ("n", "oos_r2_vs_yield_only", "cw_t", "cw_p",
                                                  "yield_only_oos_r2_vs_mean")},
                             "first": str(r["first"].date())})
    for r in oos_rows:
        print(f"  {r['tape']:10s} {r['h']:4s} {r['variable']:8s} n {r['n']:4d}  "
              f"OOS R2 vs yield-only {r['oos_r2_vs_yield_only']:+.3f}  CW t {r['cw_t']:+.2f} "
              f"p {r['cw_p']:.3f}")
    h["oos"] = oos_rows

    def _cw(tape, hz, v):
        return [r for r in oos_rows if r["tape"] == tape and r["h"] == hz
                and r["variable"] == v][0]["cw_p"]
    h["cw_p_gap"], h["cw_p_pi"] = _cw("US monthly", "12m", "gap_exp"), _cw("US monthly", "12m", "pi")
    h["cw_p_gap_rol"] = _cw("US monthly", "12m", "gap_rol")

    # ---------------------------------------------------------------- legs
    print("\n=== legs (strategy.gibson_legs) ===")
    Lus = st.gibson_legs(us, "aaa", h=12, min_train=MIN_TRAIN_M, lags=12)
    Lde = st.gibson_legs(de, "R", h=4, min_train=MIN_TRAIN_Q, lags=4)
    Luq = st.gibson_legs(uq, "aaa", h=4, min_train=MIN_TRAIN_Q, lags=4)
    h["us_gibson"], h["us_fisher"] = Lus["gibson"], Lus["fisher"]
    h["de_gibson"], h["de_fisher"] = Lde["gibson"], Lde["fisher"]
    h["uq_gibson"], h["uq_fisher"] = Luq["gibson"], Luq["fisher"]
    for k in ("us", "uq", "de"):
        print(f"  {k}: Gibson {h[k + '_gibson']}  Fisher {h[k + '_fisher']}")

    # ---------------------------------------------------------------- 7 timing
    print("\n=== 7. duration overlay on a 20y AAA par bond ===")
    bond = st.bond_returns(us["aaa"], us["rf"])
    fc12 = fcs[12]
    timers = {}
    for v in ("gap_exp", "gap_rol", "pi", "ewma"):
        w = st.timing_positions(fc12[v], "f_full")
        timers[v] = st.timing_backtest(bond, w, COST_BPS)
    timers["yield_only"] = st.timing_backtest(
        bond, st.timing_positions(fc12["gap_exp"], "f_base"), COST_BPS)
    tim = []
    for v, bt in timers.items():
        s = st.timing_summary(bt, splits=SPLITS)
        tim.append({"timer": v, "first": str(s["first"].date()), "n": s["n"],
                    "gross_sharpe": s["gross"]["sharpe"], "net_sharpe": s["net"]["sharpe"],
                    "const_sharpe": s["const"]["sharpe"], "net_ann": s["net"]["ann_mean"],
                    "const_ann": s["const"]["ann_mean"], "diff_net_ann": s["diff_net_ann"],
                    "diff_net_t": s["diff_net_t"], "turnover_ann": s["turnover_ann"],
                    "mean_pos": s["mean_pos"], "alpha_ann": s["alpha_ann"],
                    "alpha_t": s["alpha_t"], "breakeven_bps": s["breakeven_bps"],
                    "subs": s["subs"]})
        print(f"  {v:10s} net-const {s['diff_net_ann']:+.2%} (t {s['diff_net_t']:+.2f})  "
              f"Sharpe net {s['net']['sharpe']:.2f} vs const {s['const']['sharpe']:.2f}  "
              f"pos {s['mean_pos']:.2f}  alpha {s['alpha_ann']:+.2%} (t {s['alpha_t']:+.2f})  "
              f"subs {s['subs']}")
    h["timers"] = tim
    g = timers["gap_exp"]
    gs = st.timing_summary(g, splits=SPLITS)
    yo = timers["yield_only"]
    common = g.index.intersection(yo.index)
    dv = st.hac_mean(g.loc[common, "net"] - yo.loc[common, "net"], 12)
    gs["diff_vs_yield_only_ann"] = dv["mean"] * 12
    gs["diff_vs_yield_only_t"] = dv["t"]
    gs["beats_yield_only"] = bool(dv["mean"] > 0)
    h["timing"] = gs
    sweep = []
    for c in COST_GRID:
        s = st.timing_summary(st.timing_backtest(bond, st.timing_positions(fc12["gap_exp"]), c))
        sweep.append({"cost_bps": c, "diff_net_ann": s["diff_net_ann"],
                      "diff_net_t": s["diff_net_t"], "net_sharpe": s["net"]["sharpe"]})
    h["cost_sweep"] = sweep
    h["bond_stats"] = st.perf(bond["bond_ex"].loc[g.index])
    h["bond_duration_mean"] = float(bond["duration"].mean())
    print(f"  gap timer vs yield-only timer: {gs['diff_vs_yield_only_ann']:+.2%} "
          f"(t {dv['t']:+.2f})")

    # ---------------------------------------------------------------- 8 synthetic control
    print("\n=== 8. synthetic control: Fisher world vs Gibson world ===")
    syn = []
    for s_ in (0.0, 0.5, 1.0):
        w, tr = data.synthetic_world(signal_strength=s_, seed=1026)
        L = st.gibson_legs(w, "aaa", h=12, min_train=MIN_TRAIN_M, lags=12)
        syn.append({"signal_strength": s_, "gibson": L["gibson"], "fisher": L["fisher"],
                    "t_gap": L["horse_race"]["t_gap"], "t_pi": L["horse_race"]["t_pi"],
                    "eg_p_gap": L["eg_gap"]["eg_p"], "eg_p_pi": L["eg_pi"]["eg_p"],
                    "stamp_if_both_tapes": st.signal_stamp(L["gibson"], L["gibson"])})
        print(f"  s={s_:.1f} Gibson {L['gibson']} Fisher {L['fisher']}  "
              f"t_gap {L['horse_race']['t_gap']:+.2f} t_pi {L['horse_race']['t_pi']:+.2f}")
    h["synthetic"] = syn

    h["_verdict"] = st.verdict(h)
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  {h['_verdict']['one_sentence']}")
    h["runtime_s"] = time.time() - t0
    return h


def _legs(d):
    return " · ".join(f"{k} {'✔' if v else '✘'}" for k, v in d.items())


def results_md(h: dict) -> str:
    v = h["_verdict"]
    tapes = "\n".join(f"| {r['tape']} | {r['rows']} | {r['first']} → {r['last']} | "
                      f"`{r['fingerprint']}` |" for r in h["tapes"])
    prov = "\n".join(f"| `{r['tape']}` | {r['package']} | `{r['file']}` | "
                     f"`{r['sha256'][:16]}…` | {r['what']} |" if len(r["sha256"]) == 64 else
                     f"| `{r['tape']}` | {r['package']} | `{r['file']}` | {r['sha256']} | "
                     f"{r['what']} |" for r in h["provenance"])
    lv = "\n".join(f"| {r['tape']} | {r['period']} | {_f(r.get('logp'))} | "
                   f"**{_f(r.get('gap_exp'))}** | {_f(r.get('gap_rol'))} | {_f(r.get('pi'))} | "
                   f"{_f(r.get('ewma'))} |" for r in h["levels"])
    ur = "\n".join(f"| {r['series']} | {r['n']} | {r['adf_p']:.3f} | {r['kpss_p']:.3f} | "
                   f"{'**I(1)-like**' if r['i1_like'] else 'not I(1)-like'} |"
                   for r in h["unit_roots"])
    eg = "\n".join(f"| {r['tape']} | {r['x']} | {r['naive_t']:+.1f} | {r['r2']:.2f} | "
                   f"**{r['eg_p']:.3f}** |" for r in h["eg"])
    hr = "\n".join(f"| {r['tape']} | {r['period']} | {r['n']} | {r['b_gap']:+.3f} | "
                   f"**{r['t_gap']:+.2f}** | {r['b_pi']:+.3f} | **{r['t_pi']:+.2f}** | "
                   f"{r['t_gap_alone']:+.2f} | {r['t_pi_alone']:+.2f} |" for r in h["horse_race"])
    lm = "\n".join(f"| {r['halflife']} | {r['corr_level']:+.2f} | **{r['eg_p']:.3f}** | "
                   f"{r['corr_changes']:.2f} | {r['t_gap']:+.2f} | {r['t_ewma']:+.2f} |"
                   for r in h["long_memory"])
    names = {"gap_exp": "price gap (expanding)", "gap_rol": "price gap (rolling 10y)",
             "pi": "12m / 4q inflation", "ewma": f"adaptive exp. ({h['ewma_hl']:.0f}m)"}
    oos = "\n".join(f"| {r['tape']} | {r['h']} | {names.get(r['variable'], r['variable'])} | "
                    f"{r['n']} | {r['first']} | {r['oos_r2_vs_yield_only']:+.3f} | "
                    f"{r['cw_t']:+.2f} | **{r['cw_p']:.3f}** |" for r in h["oos"])
    tnames = {**names, "pi": "12m inflation", "yield_only": "yield level only"}
    tim = "\n".join(f"| {tnames.get(r['timer'], r['timer'])} | {r['gross_sharpe']:.2f} | "
                    f"{r['net_sharpe']:.2f} | {r['const_sharpe']:.2f} | "
                    f"**{r['diff_net_ann']:+.2%}** | {r['diff_net_t']:+.2f} | "
                    f"{r['mean_pos']:.2f} | {r['turnover_ann']:.2f} | "
                    f"{r['alpha_ann']:+.2%} ({r['alpha_t']:+.2f}) | "
                    + " · ".join(f"{k} {s['diff_net_ann']:+.2%} (t {s['t']:+.1f})"
                                 for k, s in r["subs"].items()) + " |"
                    for r in h["timers"])
    sw = "\n".join(f"| {r['cost_bps']:.0f} | {r['diff_net_ann']:+.2%} | {r['diff_net_t']:+.2f} | "
                   f"{r['net_sharpe']:.2f} |" for r in h["cost_sweep"])
    syn = "\n".join(f"| {r['signal_strength']:.1f} | {_legs(r['gibson'])} | {_legs(r['fisher'])} | "
                    f"{r['t_gap']:+.2f} | {r['t_pi']:+.2f} | {r['eg_p_gap']:.3f} | "
                    f"{r['eg_p_pi']:.3f} |" for r in h["synthetic"])
    t = h["timing"]
    rd = h["regime_diff"]
    return f"""# Results — Study 1026 (Gibson's Paradox) on the real tapes

*Generated by [`examples/verify.py`](../examples/verify.py) in {h['runtime_s']:.0f} s. Every
series is read from tapes frozen inside the `arch` and `statsmodels` wheels through
[`quantlab/bundled.py`](../../../quantlab/bundled.py) — no network. As-of **{h['as_of']}** (the
last full month common to the AAA, core-CPI and T-bill tapes). Yields are **nominal**, percent;
prices are 100 × log index; the macro tapes are **final-vintage**.*

## 0. Data and provenance

| Tape used here | Rows | Span | Fingerprint |
|---|--:|---|---|
{tapes}

| Source tape | Package | File | SHA-256 pin | What it is |
|---|---|---|---|---|
{prov}

**The limit that frames everything.** Gibson's paradox is a gold-standard regularity. These tapes
carry Moody's yields back to 1919 but **no price index before 1957**, so the era in which the
paradox is supposed to live cannot be tested here. This is a test of the investor's version under
**fiat money**: 1957-2018 in the US, 1972-98 in Germany.

## 1. Levels — what the claim is built on (descriptive, not evidence)

Correlation of the long yield with each price variable.

| Tape | Period | Raw log price | Price gap (expanding) | Price gap (rolling 10y) | Inflation | Adaptive exp. ({h['ewma_hl']:.0f}m) |
|---|---|--:|--:|--:|--:|--:|
{lv}

The raw price level is a clock under fiat money: it correlates {h['gi_corr_logp']:+.2f} with
the AAA yield in 1965-81, when both rose, and {h['dis_corr_logp']:+.2f} in 1982-2018, when prices
kept rising and yields fell. Detrended past-only from the start of the sample, the gap tracks
yields at {h['corr_gap']:+.2f} — higher than 12-month inflation ({h['corr_pi']:+.2f}). That is the
paradox in its modern dress. Section 2 is why none of these numbers is evidence.

## 2. Unit roots — the spurious-regression trap

ADF (H0: unit root) and KPSS (H0: stationary), constant only.

| Series | n | ADF p | KPSS p | Reading |
|---|--:|--:|--:|---|
{ur}

The yield, the log price level, inflation and the expanding price gap all read as I(1)-like.
Correlating levels of such series is exactly the Granger-Newbold (1974) trap: two independent
random walks routinely correlate at ±0.5 with t-statistics in the dozens.

## 3. Cointegration — is there a long-run relation at all?

Engle-Granger (H0: no cointegration) of the yield on each variable. The "naive t" is the levels-
OLS t-statistic the trap produces; read it as a warning, not a result.

| Tape | Yield on… | Naive levels t | R² | EG p |
|---|---|--:|--:|--:|
{eg}

The yield is **not** cointegrated with the price gap (p = {h['eg_p_gap']:.2f}) nor with 12-month
inflation (p = {h['eg_p_pi']:.2f}) — despite naive t-statistics in the dozens. It **is**
cointegrated with an adaptive inflation expectation (p = {h['eg_p_ewma']:.3f}); see §5.

## 4. First differences — the horse race (Newey-West)

ΔY on Δgap and Δπ jointly; the last two columns are each alone. In differences, Gibson's claim
reads "yield changes follow the price change (= inflation)", Fisher's "yield changes follow the
change in inflation".

| Tape | Period | n | b gap | **t gap** | b infl. | **t infl.** | t gap alone | t infl. alone |
|---|---|--:|--:|--:|--:|--:|--:|--:|
{hr}

On the US tape the gap wins: t = {h['t_gap']:+.2f} against {h['t_pi']:+.2f} for inflation in the
joint regression, and it holds on headline CPI (macrodata) and for BAA yields. Germany shows
nothing either way. The regime split is not significant: gap slope 1982-2018 minus 1965-81 =
{rd['diff_gap']:+.3f} (HAC t {rd['t_diff_gap']:+.2f}).

## 5. The confound — Fisher with a long memory

Fisher (1930) explained Gibson's correlation with **slowly adapting inflation expectations**: if
expected inflation is a long distributed lag of past inflation, the yield tracks a smoothed
*cumulation* of inflation, which is what a detrended price level is. Proxy: an exponentially
weighted average of one-month inflation (past-only). The whole half-life grid is shown, not a
chosen point; the headline uses {h['ewma_hl']:.0f} months, the middle of it.

| Half-life (months) | Corr with yield (levels) | EG p | Corr(Δgap, Δexp.) | t gap (joint Δ) | t exp. (joint Δ) |
|--:|--:|--:|--:|--:|--:|
{lm}

In changes the gap and the adaptive expectation are nearly the same variable (correlation
≥ 0.9), so the difference regression cannot separate them — the gap's t drops to
{h['t_gap_vs_ewma']:+.2f}. In levels the adaptive expectation fits the yield better than the gap
*and* is cointegrated with it, which the gap is not. The US "Gibson" signal is most economically
read as **Fisher with a long memory**, which is the resolution Fisher himself proposed.

## 6. Out of sample — does the price level tell you where yields go?

Expanding window, minimum {MIN_TRAIN_M} months (Germany {MIN_TRAIN_Q} quarters) of realised
outcomes; the price variable is lagged one period for CPI publication. Benchmark = yield-level-
only model (mean reversion); Clark-West (2007) one-sided test of the improvement.

| Tape | Horizon | Added variable | n | First forecast | OOS R² vs yield-only | CW t | CW p |
|---|---|---|--:|---|--:|--:|--:|
{oos}

The expanding gap improves the US forecast at both horizons; the rolling-window gap — the same
idea, detrended over ten years instead of the whole history — does not (p = {h['cw_p_gap_rol']:.2f}).
Germany's samples are short (≈ 40-60 forecasts) and show nothing.

## 7. Could you trade it? A duration overlay on a 20-year AAA par bond

**Bond return (the duration approximation, stated):** r[t+1] ≈ Y[t]/12 − D[t]·ΔY + ½·C[t]·ΔY²,
D and C the modified duration and convexity of a 20-year semi-annual par bond at the Moody's AAA
yield (mean D ≈ {h['bond_duration_mean']:.1f}). Excess of the one-month T-bill (Fama-French RF).
**Rule:** at each month-end, 1.5× duration if the 12-month forecast of the yield change is below
the prevailing mean change, else 0.5×; the weight set at *t* earns month *t+1* (one execution lag),
CPI lagged one month on top. Costs {h['cost_bps']:.0f} bp one-way × NAV traded. Benchmark:
constant 1× duration. All Sharpe ratios are **excess-of-cash vs excess-of-cash**.

| Timer (12m forecast) | Gross Sharpe | Net Sharpe | Constant Sharpe | Net − constant /yr | HAC t | Mean position | Turnover /yr | Alpha vs constant (t) | By regime (net − constant) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
{tim}

Cost sweep for the gap timer:

| One-way cost (bp) | Net − constant /yr | HAC t | Net Sharpe |
|--:|--:|--:|--:|
{sw}

The gap timer beats constant duration by {t['diff_net_ann']:+.2%} a year (t
{t['diff_net_t']:+.2f}) and the yield-only timer by {t['diff_vs_yield_only_ann']:+.2%} (t
{t['diff_vs_yield_only_t']:+.2f}). It trades {t['turnover_ann']:.2f}× NAV a year, so costs are
irrelevant (break-even {t['breakeven_bps']:.0f} bp). What it is: a near-static long-duration call
held through the 1982-2018 bond bull, with nothing to show in 1965-81 — one regime, one trade.

## 8. Synthetic control — can the machinery tell the worlds apart?

Same price path; the yield is a Fisher world (s = 0), a Gibson world (s = 1) or an even blend.
A **machinery proof only** — never evidence about markets.

| signal_strength | Gibson legs (L1 coint · L2 Δ · L3 OOS) | Fisher legs | t gap | t infl. | EG p gap | EG p infl. |
|--:|---|---|--:|--:|--:|--:|
{syn}

## Caveats

- **No gold standard.** No price index before 1957 on these tapes; the paradox's home era is
  untested. A fiat-era answer is not an answer about 1820-1913.
- **Moody's AAA is a seasoned corporate index**, not a Treasury: it carries a credit spread and an
  average maturity of 20+ years that drifts. The par-bond return is an approximation, not an
  investable index's total return; there is no long AAA futures contract, so the overlay would be
  run with Treasury futures and carry basis risk.
- **Final-vintage macro.** Core CPI is barely revised and is lagged one month here; the German
  deflator and the macrodata CPI are revised series.
- **The price gap is a choice.** Expanding-from-1957 detrending works, rolling 10-year does not.
  A different start date would move the expanding trend too; the result is specification-
  sensitive and the desk says so on the Signal axis.
- **Clark-West with overlapping 12-month targets** uses Newey-West with h + 2 lags; the timer's
  t-statistics use 12 lags on monthly P&L.
- **Germany is short** (107 quarters, ≈ 40-60 out-of-sample forecasts): low power, but also no
  hint of the effect at all.

## Verdict

Produced by `strategy.verdict` (rules in `signal_stamp` / `trad_stamp`), fixed before the run
and unit-tested in [`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1026-gibsons-paradox](../README.md). Not investment advice.*
"""


def _jsonable(o):
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    t = h["timing"]
    head = {"study": "1026-gibsons-paradox", "as_of": h["as_of"],
            "signal": h["_verdict"]["signal"], "trad": h["_verdict"]["trad"],
            "t_gap": h["t_gap"], "t_pi": h["t_pi"], "eg_p_gap": h["eg_p_gap"],
            "eg_p_ewma": h["eg_p_ewma"], "cw_p_gap": h["cw_p_gap"], "de_t_gap": h["de_t_gap"],
            "timer_diff_net_ann": t["diff_net_ann"], "timer_diff_net_t": t["diff_net_t"],
            "fingerprints": {r["tape"]: r["fingerprint"] for r in h["tapes"]},
            "one_sentence": h["_verdict"]["one_sentence"]}
    print("##HEADLINE## " + json.dumps(head, default=_jsonable))


if __name__ == "__main__":
    main()
