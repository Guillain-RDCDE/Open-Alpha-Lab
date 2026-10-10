"""Real-tape verification — Study 1021 (Stability Breeds Instability). Regenerates docs/results.md.

Builds the prolonged-calm score on 92 years of Fama-French monthly total returns, regresses
the next 1-36 months' maximum drawdown, crash odds, volatility and excess return on it (HAC
and non-overlapping), then asks the question that matters: is the real slope bigger than
what volatility **mean reversion alone** delivers? Two no-feedback models are fitted to the
same tape — GARCH(1,1)-t and long-memory FIGARCH-t — and the identical regression is run on
hundreds of simulated tapes from each. The daily S&P 500 tape supplies the short-horizon
sign (clustering) and a post-1990 illustration; the VIX tape supplies one anecdote, labelled
as such. Finally a calm de-risk rule is raced against buy-and-hold and a plain vol-target.

    python studies/1021-calm-before-the-storm/examples/verify.py

Everything is offline — the tapes ship inside ``arch`` and the pinned ``skfolio`` wheel.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from calm import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(HERE, "..", "docs"))
N_PATHS = 400
GRID_PATHS = 200


def report() -> dict:
    t_start = time.time()
    ff = data.load_ff()
    h: dict = {"as_of": data.AS_OF, "as_of_spx": data.AS_OF_SPX, "as_of_vix": data.AS_OF_VIX,
               "fingerprint": data.fingerprint(ff), "provenance": data.provenance()}
    r, rf = ff["mkt"], ff["rf"]
    print(f"as-of {data.AS_OF} (FF)  fingerprint {h['fingerprint']}")
    for p in h["provenance"]:
        print(f"  {p['tape']:45s} {p['first']} -> {p['last']}  {p['fingerprint']}")

    # ------------------------------------------------------------------ 1. the signal
    print("\n=== 1. the prolonged-calm score ===")
    sig = st.calm_signal(r)
    c = sig["calm"].dropna()
    h["signal_start"] = str(c.index[0].date())
    h["n_signal"] = int(len(c))
    peaks = []
    for dec in range(c.index[0].year // 10 * 10, 2020, 10):
        s = c[(c.index.year >= dec) & (c.index.year < dec + 10)]
        if len(s):
            peaks.append({"decade": f"{dec}s", "peak_date": str(s.idxmax().date()),
                          "peak_calm": float(s.max()), "mean_calm": float(s.mean())})
    h["peaks"] = peaks
    for p in peaks:
        print(f"  {p['decade']}: peak calm {p['peak_calm']:.3f} on {p['peak_date']}")

    # ------------------------------------------------------------------ 2. horizons
    print("\n=== 2. calm -> forward outcomes, every horizon ===")
    tab = st.horizon_table(r, rf)
    print(tab.round(4).to_string(index=False))
    h["horizon_table"] = tab.to_dict("records")
    prim = tab[(tab["H"] == st.PRIMARY_H) & (tab["outcome"] == "mdd")].iloc[0]
    h["primary_slope"] = float(prim["slope"])
    h["primary_t"] = float(prim["t_hac"])
    h["primary_t_nonoverlap"] = float(prim["t_nonoverlap"])
    h["n_indep"] = int(prim["n_indep"])
    h["primary_mean_mdd"] = float(prim["mean_y"])
    long = tab[(tab["H"] >= 12) & (tab["outcome"].isin(["mdd", "crash", "vol"]))]
    h["long_all_positive"] = bool((long["slope"] > 0).all())
    h["long_max_t"] = float(long["t_hac"].max())
    print(f"  PRIMARY (24m max drawdown): slope {h['primary_slope']:+.4f} per SD, "
          f"HAC t {h['primary_t']:.2f}, non-overlap t {h['primary_t_nonoverlap']:.2f}")

    # ------------------------------------------------------------------ 3. crash odds
    print("\n=== 3. crash odds: top-quartile calm vs the rest ===")
    odds = {H: st.crash_odds(r, rf, H) for H in (12, 24, 36)}
    h["odds"] = {str(k): v for k, v in odds.items()}
    for H, o in odds.items():
        print(f"  H={H:2d}: P(DD>=20%) calm {o['p_calm']:.1%} [{o['ci_calm'][0]:.1%},"
              f"{o['ci_calm'][1]:.1%}]  rest {o['p_rest']:.1%}")
    h["n_episodes"] = int(odds[24]["n_episodes"])
    print(f"  independent >=20% drawdown episodes after {h['signal_start']}: "
          f"{h['n_episodes']}")

    # ------------------------------------------------------------------ 4. the nulls
    print("\n=== 4. is it more than mean reversion? ===")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_g = st.fit_vol_model(ff["mkt_rf"], "garch")
        res_f = st.fit_vol_model(ff["mkt_rf"], "figarch")
        nulls = {}
        for kind, res in (("garch", res_g), ("figarch", res_f)):
            m = st.mean_reversion_null(ff, kind, st.PRIMARY_H, N_PATHS, seed=1021, res=res)
            nulls[kind] = m
            print(f"  {kind}: params {', '.join(f'{k}={v:.3f}' for k, v in m['params'].items())}")
            for o in st.OUTCOMES:
                q = m[o]
                print(f"    {o:5s} real {q['real']:+.4f} | null mean {q['null_mean']:+.4f} "
                      f"90% [{q['null_q05']:+.4f},{q['null_q95']:+.4f}] | p {q['p_value']:.3f}")
    h["nulls"] = {k: {kk: vv for kk, vv in v.items() if kk != "_null_samples"}
                  for k, v in nulls.items()}
    h["p_garch"] = nulls["garch"]["mdd"]["p_value"]
    h["p_figarch"] = nulls["figarch"]["mdd"]["p_value"]
    h["p_garch_vol"] = nulls["garch"]["vol"]["p_value"]
    h["null_garch_mean"] = nulls["garch"]["mdd"]["null_mean"]
    h["garch_halflife"] = float(np.log(0.5) / np.log(res_g.params["alpha[1]"]
                                                       + res_g.params["beta[1]"]))

    # ------------------------------------------------------------------ 5. robustness grid
    print("\n=== 5. robustness: signal windows (HAC t, GARCH p) ===")
    grid = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tw in (60, 120, 180):
            for cw in (36, 60, 84):
                win = {"trend_window": tw, "calm_window": cw}
                s2 = st.calm_signal(r, **win)
                fo = st.forward_outcomes(r, rf, st.PRIMARY_H)
                rr = st.predictive_regression(s2["calm"], fo["mdd"], st.PRIMARY_H)
                m = st.mean_reversion_null(ff, "garch", st.PRIMARY_H, GRID_PATHS,
                                           seed=7, res=res_g, windows=win)
                grid.append({"trend": tw, "calm": cw, "slope": rr["slope"], "t": rr["t"],
                             "p_garch": m["mdd"]["p_value"],
                             "pre_registered": tw == 120 and cw == 60})
                print(f"  trend {tw:3d}m calm {cw:2d}m: slope {rr['slope']:+.4f} "
                      f"t {rr['t']:+.2f} GARCH p {m['mdd']['p_value']:.3f}")
    h["grid"] = grid
    h["grid_share_p10"] = float(np.mean([g["p_garch"] < 0.10 for g in grid]))
    h["grid_share_t2"] = float(np.mean([g["t"] >= 2 for g in grid]))

    halves = st.half_sample_slopes(r, rf)
    h["halves"] = halves
    h["half_slopes"] = [float(x["slope"]) for x in halves]
    for x in halves:
        print(f"  half {x['start']} -> {x['end']}: slope {x['slope']:+.4f} t {x['t']:+.2f}")

    # ------------------------------------------------------------------ 6. both horizons
    print("\n=== 6. the two horizons: clustering vs the Minsky claim ===")
    if data.have_spx():
        spx = data.load_spx_daily()
        rvm = data.monthly_rv_from_daily(spx)
        ar = st.clustering_ar1(rvm)
        h["ar1_phi"], h["ar1_t"] = ar["phi"], ar["t"]
        h["ar1_half_life"] = ar["half_life_months"]
        h["ar1_n"] = ar["n"]
        print(f"  daily S&P monthly RV, log AR(1): phi {ar['phi']:.3f} (HAC t {ar['t']:.1f}), "
              f"half-life {ar['half_life_months']:.1f} months, n={ar['n']}")
    else:
        h["ar1_phi"], h["ar1_t"], h["ar1_half_life"], h["ar1_n"] = (np.nan,) * 4
    prof_dev = st.vol_horizon_profile(r, -sig["dev"])
    prof_calm = st.vol_horizon_profile(r, sig["calm"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nprof = st.null_horizon_profile(res_g, "garch", ff, n_paths=60)
    prof = prof_calm.set_index("h")[["corr"]].rename(columns={"corr": "calm"})
    prof["low_now"] = prof_dev.set_index("h")["corr"]
    prof = prof.join(nprof.rename(columns={"mean": "null_mean", "q05": "null_q05",
                                           "q95": "null_q95"}))
    h["profile"] = prof.reset_index().to_dict("records")
    h["profile_above_q95"] = [int(i) for i in prof.index if prof.loc[i, "calm"] > prof.loc[i, "null_q95"]]
    h["low_now_first_positive"] = int(prof.index[prof["low_now"] > 0][0]) \
        if (prof["low_now"] > 0).any() else None
    print(prof.round(3).to_string())

    # ------------------------------------------------------------------ 7. post-1990 & VIX
    print("\n=== 7. illustrations (not inference): post-1990 S&P and the VIX tape ===")
    illus = []
    if data.have_spx():
        mret = data.monthly_returns_from_daily(spx)
        s3 = st.calm_signal(mret)["calm"]
        fo3 = st.forward_outcomes(mret, None, 24)
        for d in ("2006-12-31", "2007-06-30", "2014-06-30", "2017-12-31", "2019-12-31"):
            ts = pd.Timestamp(d)
            if ts in s3.index and np.isfinite(s3.loc[ts]):
                pct = float((s3.loc[:ts].dropna() <= s3.loc[ts]).mean())
                illus.append({"date": d, "calm": float(s3.loc[ts]), "pct_own_past": pct,
                              "fwd24_mdd": float(fo3.loc[ts, "mdd"])
                              if np.isfinite(fo3.loc[ts, "mdd"]) else np.nan})
        for x in illus:
            print(f"  {x['date']}: calm {x['calm']:.3f} ({x['pct_own_past']:.0%} of own past), "
                  f"next-24m max DD (price) {x['fwd24_mdd']:.1%}")
    h["illus"] = illus
    vx = {}
    if data.have_vix() and data.have_spx():
        vix = data.load_vix()
        v17 = vix[vix.index.year == 2017]
        v18 = vix[vix.index.year == 2018]
        s18 = spx[(spx.index >= "2018-01-01") & (spx.index <= "2018-12-31")]
        vx = {"vix_2017_mean": float(v17.mean()), "vix_2017_min": float(v17.min()),
              "vix_2017_share_below_11": float((v17 < 11).mean()),
              "vix_2018_max": float(v18.max()),
              "vix_2018_max_date": str(v18.idxmax().date()),
              "spx_2018_max_dd": float((s18 / s18.cummax() - 1).min()),
              "vix_all_mean": float(vix.mean()), "vix_n_years": 5}
        print(f"  VIX 2017 mean {vx['vix_2017_mean']:.1f}, min {vx['vix_2017_min']:.2f}; "
              f"2018 peak {vx['vix_2018_max']:.1f} on {vx['vix_2018_max_date']}; "
              f"S&P 2018 max DD {vx['spx_2018_max_dd']:.1%}. ONE episode.")
    h["vix"] = vx

    # ------------------------------------------------------------------ 8. the trade
    print("\n=== 8. could you trade it? ===")
    W = {k: st.rule_weights(r, k) for k in ("buy_hold", "calm_derisk", "vol_target")}
    start = max(W["calm_derisk"].first_valid_index(), W["vol_target"].first_valid_index())
    ffb = ff.loc[start:]
    bts = {k: st.backtest(ffb, w.loc[start:], 10.0) for k, w in W.items()}
    summ = {k: st.summary(b) for k, b in bts.items()}
    h["backtest"] = summ
    for k, s_ in summ.items():
        print(f"  {k:12s} Sharpe gross {s_['sharpe_gross']:.3f} net {s_['sharpe_net']:.3f} "
              f"CAGR {s_['cagr_net']:.2%} vol {s_['vol']:.2%} maxDD {s_['max_dd']:.1%} "
              f"terminal {s_['terminal']:,.0f}x turnover {s_['turnover_yr']:.2f}/yr "
              f"avg w {s_['avg_weight']:.2f}")
    ex = {k: b["net"] - b["rf"] for k, b in bts.items()}
    bs = st.sharpe_diff_bootstrap(ex["calm_derisk"], ex["buy_hold"])
    bs_vt = st.sharpe_diff_bootstrap(ex["vol_target"], ex["buy_hold"])
    h["boot_derisk"], h["boot_voltarget"] = bs, bs_vt
    print(f"  calm de-risk - B&H Sharpe {bs['diff']:+.3f} CI [{bs['ci'][0]:+.3f},"
          f"{bs['ci'][1]:+.3f}] p(<=0) {bs['p_le_zero']:.2f}")
    print(f"  vol target  - B&H Sharpe {bs_vt['diff']:+.3f} CI [{bs_vt['ci'][0]:+.3f},"
          f"{bs_vt['ci'][1]:+.3f}] p(<=0) {bs_vt['p_le_zero']:.2f}")
    sweep = st.cost_sweep(ffb, W["calm_derisk"].loc[start:])
    h["sweep"] = sweep.to_dict("records")
    print(sweep.round(4).to_string(index=False))
    # In which months was it de-risked, and what did the market do then?
    w = bts["calm_derisk"]["w"]
    mex = (ffb["mkt"] - ffb["rf"]).reindex(w.index)
    h["share_derisked"] = float((w < 1).mean())
    h["mkt_ex_when_derisked"] = float(mex[w < 1].mean() * 12)
    h["mkt_ex_otherwise"] = float(mex[w >= 1].mean() * 12)
    print(f"  de-risked {h['share_derisked']:.0%} of months; market excess return then "
          f"{h['mkt_ex_when_derisked']:.1%}/yr vs {h['mkt_ex_otherwise']:.1%}/yr otherwise")

    h["derisk_sharpe"] = summ["calm_derisk"]["sharpe_net"]
    h["bh_sharpe"] = summ["buy_hold"]["sharpe_net"]
    h["voltarget_sharpe"] = summ["vol_target"]["sharpe_net"]
    h["derisk_sharpe_diff"] = h["derisk_sharpe"] - h["bh_sharpe"]
    h["derisk_p"] = bs["p_le_zero"]
    h["derisk_max_dd"] = summ["calm_derisk"]["max_dd"]
    h["bh_max_dd"] = summ["buy_hold"]["max_dd"]
    h["derisk_terminal"] = summ["calm_derisk"]["terminal"]
    h["bh_terminal"] = summ["buy_hold"]["terminal"]
    h["derisk_turnover"] = summ["calm_derisk"]["turnover_yr"]
    h["bt_years"] = len(ffb) / 12.0
    h["bt_start"] = str(ffb.index[0].date())

    h["_verdict"] = st.verdict(h)
    h["runtime_s"] = time.time() - t_start
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    print(f"  {h['_verdict']['one_sentence']}")
    print(f"  runtime {h['runtime_s']:.0f}s")
    return h


def _f(x, fmt):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, fmt)


def results_md(h: dict) -> str:
    v = h["_verdict"]
    prov = "\n".join(
        f"| {p['tape']} | `{p['package']}` · `{p['file']}` | `{p['sha256'][:16]}…` | "
        f"{p['label']} | {p['first']} → {p['last']} | {p['rows']:,} | `{p['fingerprint']}` |"
        for p in h["provenance"])
    peaks = "\n".join(f"| {p['decade']} | {p['peak_date']} | {p['peak_calm']:.3f} | "
                      f"{p['mean_calm']:.3f} |" for p in h["peaks"])
    names = {"mdd": "max drawdown", "crash": "P(DD ≥ 20%)", "vol": "realised vol",
             "ret": "excess log return"}
    hz = "\n".join(
        (f"| **{int(x['H'])}** | **{names[x['outcome']]}** | **{x['slope']:+.4f}** | "
         f"**{x['t_hac']:+.2f}** | **{_f(x['t_nonoverlap'], '+.2f')}** | {x['n_indep']} | "
         f"{x['mean_y']:.3f} |")
        if (x["H"] == 24 and x["outcome"] == "mdd") else
        (f"| {int(x['H'])} | {names[x['outcome']]} | {x['slope']:+.4f} | {x['t_hac']:+.2f} | "
         f"{_f(x['t_nonoverlap'], '+.2f')} | {x['n_indep']} | {x['mean_y']:.3f} |")
        for x in h["horizon_table"])
    od = "\n".join(
        f"| {k} | {o['p_calm']:.1%} [{o['ci_calm'][0]:.1%}, {o['ci_calm'][1]:.1%}] | "
        f"{o['p_rest']:.1%} [{o['ci_rest'][0]:.1%}, {o['ci_rest'][1]:.1%}] | "
        f"{o['n_calm']} / {o['n_rest']} |" for k, o in h["odds"].items())
    nl = []
    for kind in ("garch", "figarch"):
        m = h["nulls"][kind]
        for o in st.OUTCOMES:
            q = m[o]
            bold = "**" if o == "mdd" else ""
            nl.append(f"| {kind.upper()}-t | {names[o]} | {bold}{q['real']:+.4f}{bold} | "
                      f"{q['null_mean']:+.4f} | [{q['null_q05']:+.4f}, {q['null_q95']:+.4f}] | "
                      f"{bold}{q['p_value']:.3f}{bold} |")
    nl = "\n".join(nl)
    pg = h["nulls"]["garch"]["params"]
    pf = h["nulls"]["figarch"]["params"]
    gr = "\n".join(
        f"| {g['trend']} | {g['calm']} | {g['slope']:+.4f} | {g['t']:+.2f} | "
        f"{g['p_garch']:.3f} |{' ← pre-registered' if g['pre_registered'] else ''}"
        for g in h["grid"])
    hv = "\n".join(f"| {x['start']} → {x['end']} | {x['slope']:+.4f} | {x['t']:+.2f} |"
                   for x in h["halves"])
    pr = "\n".join(
        f"| {int(x['h'])} | {x['low_now']:+.3f} | {x['calm']:+.3f} | {x['null_mean']:+.3f} | "
        f"[{x['null_q05']:+.3f}, {x['null_q95']:+.3f}] |"
        for x in h["profile"] if int(x["h"]) in (0, 3, 6, 12, 18, 24, 30, 36, 48, 60))
    il = "\n".join(f"| {x['date']} | {x['calm']:.3f} | {x['pct_own_past']:.0%} | "
                   f"{_f(x['fwd24_mdd'], '.1%')} |" for x in h["illus"])
    vx = h["vix"]
    bt = h["backtest"]
    lab = {"buy_hold": "Buy & hold", "calm_derisk": "Calm de-risk (½ in bills)",
           "vol_target": "Vol-target (unlevered)"}
    btr = "\n".join(
        f"| {lab[k]} | {s['sharpe_gross']:.3f} | {s['sharpe_net']:.3f} | {s['cagr_net']:.2%} | "
        f"{s['vol']:.1%} | {s['max_dd']:.1%} | {s['terminal']:,.0f}× | {s['turnover_yr']:.2f} | "
        f"{s['avg_weight']:.2f} |" for k, s in bt.items())
    sw = "\n".join(f"| {int(x['cost_bps'])} | {x['sharpe_rule']:.3f} | {x['sharpe_bh']:.3f} | "
                   f"{x['diff']:+.3f} |" for x in h["sweep"])
    b1, b2 = h["boot_derisk"], h["boot_voltarget"]
    vix_block = ""
    if vx:
        vix_block = f"""
**The VIX tape (2014-01 → 2018-12), one episode.** In 2017 the VIX averaged
**{vx['vix_2017_mean']:.1f}**, closed below 11 on {vx['vix_2017_share_below_11']:.0%} of days and
touched {vx['vix_2017_min']:.2f} — the calmest year in the index's history to that point. In 2018 it
spiked to **{vx['vix_2018_max']:.1f}** ({vx['vix_2018_max_date']}, "Volmageddon") and the S&P 500
price index fell **{vx['spx_2018_max_dd']:.1%}** peak to trough within the year. That is exactly
the story the believers tell — and it is **one** observation on a five-year tape. It cannot
support or refute anything; it is here because it is the picture everyone has in mind.
"""
    return f"""# Results — Study 1021 (Stability Breeds Instability) on the real tapes

*Generated by [`examples/verify.py`](../examples/verify.py). Primary
tape as-of **{h['as_of']}** (Fama-French monthly, last full month); S&P 500 daily as-of
**{h['as_of_spx']}**; VIX as-of **{h['as_of_vix']}** (partial final months dropped). Headline
fingerprint `{h['fingerprint']}`. Null simulations: {N_PATHS} paths per model, seed 1021.*

## 0. Data provenance

All tapes are read offline through [`quantlab/bundled.py`](../../../quantlab/bundled.py), which
refuses any file whose SHA-256 differs from the pin.

| Tape | Package · file | SHA-256 pin | Label | Span | Rows | Fingerprint |
|---|---|---|---|---|--:|---|
{prov}

The **total-return** Fama-French market (`mkt = Mkt-RF + RF`) carries every return-level
number: drawdowns, terminal wealth, Sharpe. The S&P 500 tape is a **price index** and is used
only for realised volatility from daily returns and a labelled illustration. Everything is
nominal.

## 1. The prolonged-calm score

`rv` = 12-month realised vol; `trend` = its trailing 10-year mean; `low = max(−log(rv/trend), 0)`;
**`calm` = the trailing 5-year mean of `low`** — all past-only, known at the month's close. The
first usable reading is **{h['signal_start']}** ({h['n_signal']} months).

| Decade | Calmest month | Peak calm | Mean calm |
|---|---|--:|--:|
{peaks}

## 2. Calm → what follows, at every horizon

Slope of each forward outcome on the **standardised** calm score (per 1 SD), Newey-West with
2H lags; the non-overlapping t keeps every H-th month (median over the H offsets). The
pre-registered primary test is in bold.

| H (months) | Outcome | Slope per SD | HAC t | Non-overlap t | Independent windows | Mean outcome |
|--:|---|--:|--:|--:|--:|--:|
{hz}

The primary slope is **{h['primary_slope']:+.4f}** — a {abs(h['primary_slope']):.1%}-point
{'deeper' if h['primary_slope'] > 0 else 'shallower'} 24-month drawdown per SD of calm, against an
average 24-month drawdown of {h['primary_mean_mdd']:.1%}. HAC t = **{h['primary_t']:.2f}**.
{'Every long-horizon (H ≥ 12) drawdown, crash and volatility slope points the Minsky way' if h['long_all_positive'] else 'The long-horizon slopes do not even agree in sign'};
the largest HAC t among them is {h['long_max_t']:.2f}{' — none clears 2' if h['long_max_t'] < 2 else ''}.

## 3. Crash odds — top-quartile calm vs the rest

Share of months followed by a ≥20% drawdown within H. Wilson intervals count overlapping
months and are therefore **too narrow**; the honest sample size is the number of separate
crashes — **{h['n_episodes']}** ≥20% drawdown episodes after {h['signal_start']}.

| H | Calm top quartile | The rest | Months (calm / rest) |
|---|--:|--:|--:|
{od}

## 4. The confound — is it more than mean reversion?

Each model is fitted to the real monthly excess returns and simulated {N_PATHS} times at the
real length, with the real bill path added back; the identical regression (24-month horizon)
runs on every simulated tape. p = share of no-feedback tapes with a slope at least as large
as the real one (one-sided, in Minsky's direction).

| Null | Outcome | Real slope | Null mean | Null 90% band | p |
|---|---|--:|--:|--:|--:|
{nl}

GARCH(1,1)-t: ω = {pg['omega']:.3f}, α = {pg['alpha[1]']:.3f}, β = {pg['beta[1]']:.3f},
ν = {pg['nu']:.1f} (persistence half-life {h['garch_halflife']:.0f} months). FIGARCH-t:
d = {pf['d']:.3f}, φ = {pf['phi']:.3f}, β = {pf['beta']:.3f}, ν = {pf['nu']:.1f}.

This is the table the study exists for. **In a world with no risk-taking feedback, calm is
followed by *smaller* drawdowns** (null mean {h['null_garch_mean']:+.4f}): vol is persistent,
and a quiet market stays quiet for a while. The real slope is positive, so it is
{h['primary_slope'] - h['null_garch_mean']:+.4f} above what mean reversion predicts — that gap
is the Minsky channel's best case, and it lands at **p = {h['p_garch']:.3f}** (GARCH) and
**p = {h['p_figarch']:.3f}** (FIGARCH).

## 5. Robustness — signal windows and the two halves

The pre-registered definition is one point on this grid; the rest are shown so a reader can
see whether it was lucky.

| Trend window (m) | Calm window (m) | Slope | HAC t | GARCH p ({GRID_PATHS} paths) |
|--:|--:|--:|--:|--:|
{gr}

{h['grid_share_p10']:.0%} of the nine definitions reach GARCH p < 0.10; {h['grid_share_t2']:.0%}
reach a raw HAC t ≥ 2.

| Half-sample | Slope | HAC t |
|---|--:|--:|
{hv}

{'The two halves disagree in sign: the second half carries the whole effect (1987, 2000-02 and 2008 all sit in it), the first half points the other way.' if h['half_slopes'][0] * h['half_slopes'][1] < 0 else 'The two halves agree in sign.'}

## 6. The two horizons — clustering vs the Minsky claim

**Short horizon (daily S&P 500, 1990-2022).** Log monthly realised vol from daily returns has
AR(1) **φ = {_f(h['ar1_phi'], '.3f')}** (HAC t = {_f(h['ar1_t'], '.1f')}, half-life
{_f(h['ar1_half_life'], '.1f')} months, n = {h['ar1_n']}). Calm this month predicts calm next
month — overwhelmingly.

**Every horizon (Fama-French).** Correlation of today's state with log realised vol over the
6 months starting h months ahead. `low_now` is how far vol sits below trend *today*; `calm` is
the 5-year score; the null columns are the calm correlation on 60 GARCH tapes.

| h (months ahead) | low_now | calm | GARCH null mean | GARCH null 90% |
|--:|--:|--:|--:|--:|
{pr}

`low_now` is the clustering signature: low vol today goes with low vol over the next months
(negative correlation), and the sign only turns positive from about h = {h['low_now_first_positive']}
months out. The 5-year `calm` score correlates *positively* with future vol at every horizon
up to about four years, while the no-feedback GARCH tapes centre slightly below zero. The real
curve pokes above the null's pointwise 95th percentile at h = {', '.join(str(x) for x in h['profile_above_q95']) or 'none'}
(60 null tapes). Taken at face value this is the strongest-looking evidence in the study, and
it points Minsky's way: prolonged calm goes with *more* volatility one to three years later
than mean reversion allows. But it was not the pre-registered test — these are overlapping,
highly correlated horizons scanned after the fact, the band is pointwise and built on 60 paths
— so it moves the stamp nowhere. It is the first thing a fork should pre-register: calm →
forward 24-month volatility, scored by correlation against a no-feedback null.

## 7. Illustrations — not inference

**Post-1990 S&P 500 (price index, monthly).** The calm score needs 191 months of history, so
on this tape it starts in late 2005 — about one and a half crash cycles.

| Date | Calm | Percentile of its own past | Next-24-month max DD (price) |
|---|--:|--:|--:|
{il}
{vix_block}
## 8. Could you trade it?

All legs on the total-return tape from **{h['bt_start']}** ({h['bt_years']:.0f} years): weight
decided at the close of month t, held through t+1; turnover = one-way × NAV from the drifted
weight; **10 bp** one-way; the cash sleeve earns T-bills. Sharpe is on **excess-of-bills**
returns for every leg.

| Rule | Sharpe gross | Sharpe net | CAGR net | Vol | Max DD | Terminal wealth | Turnover/yr | Avg equity weight |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{btr}

Block bootstrap (24-month circular blocks, 2,000 draws) of the net excess Sharpe difference:

| Comparison | Δ Sharpe | 95% CI | p(Δ ≤ 0) |
|---|--:|--:|--:|
| Calm de-risk − buy & hold | {b1['diff']:+.3f} | [{b1['ci'][0]:+.3f}, {b1['ci'][1]:+.3f}] | {b1['p_le_zero']:.2f} |
| Vol-target − buy & hold | {b2['diff']:+.3f} | [{b2['ci'][0]:+.3f}, {b2['ci'][1]:+.3f}] | {b2['p_le_zero']:.2f} |

Cost sweep (net excess Sharpe):

| One-way cost (bp) | Calm de-risk | Buy & hold | Δ |
|--:|--:|--:|--:|
{sw}

The rule was de-risked in **{h['share_derisked']:.0%}** of months. The market's excess return
in those months was **{h['mkt_ex_when_derisked']:.1%} a year**, against
{h['mkt_ex_otherwise']:.1%} in the rest{' — calm months are not bad months, so stepping out of them gives up premium' if h['mkt_ex_when_derisked'] > 0 else ''}. At {h['derisk_turnover']:.2f}× NAV a year costs are irrelevant; whatever the rule earns
or loses, it earns or loses on *timing*, not on friction.

## Caveats

- **Few independent crashes.** 92 years of monthly data, but the calm score needs 16 years of
  warm-up and the outcome is a multi-year drawdown: the long-horizon regressions rest on
  ~{h['n_indep']} non-overlapping 24-month windows and {h['n_episodes']} separate ≥20% drawdowns.
  Any verdict on a crisis predictor at this horizon is low-powered by construction, which is
  why the null comparison (rather than a bare t) carries the inference.
- **US equity drawdowns are not "crises".** Danielsson, Valenzuela & Zer study banking crises
  across 60 countries; this study tests the practitioner's translation — US stock-market
  drawdowns. A null here does not refute their cross-country banking result.
- **The nulls are fitted on the full sample.** They are null-generating processes, not
  forecasters, so no trading rule sees their parameters. GARCH and FIGARCH are two
  mean-reversion shapes; a regime-switching null could be wider still, which would only
  weaken the case.
- **Monthly realised vol from 12 monthly returns is a slow, noisy instrument** on the
  Fama-French tape (no daily data before 1990 here). It is the price of the 92-year window.
- **The robustness grid is shown, not chosen.** The verdict uses only the pre-registered
  definition (10-year trend, 5-year calm, 24-month drawdown).
- **VIX: five years, one episode.** Illustration only.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1021-calm-before-the-storm](../README.md). Not investment advice.*
"""


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    keep = {k: h[k] for k in (
        "as_of", "fingerprint", "primary_slope", "primary_t", "primary_t_nonoverlap",
        "p_garch", "p_figarch", "p_garch_vol", "null_garch_mean", "n_episodes", "n_indep",
        "ar1_phi", "ar1_t", "derisk_sharpe", "bh_sharpe", "voltarget_sharpe",
        "derisk_sharpe_diff", "derisk_p", "derisk_max_dd", "bh_max_dd", "derisk_terminal",
        "bh_terminal", "half_slopes", "grid_share_p10", "runtime_s")}
    keep["signal"] = h["_verdict"]["signal"]
    keep["trad"] = h["_verdict"]["trad"]
    print("##HEADLINE## " + json.dumps(keep, default=float))


if __name__ == "__main__":
    main()
