"""Real-tape verification — Study 1020 (More Volatile Than Ever?). Regenerates docs/results.md.

Runs the whole teardown on the three frozen bundled tapes:

1. the trend in annual log realised volatility over 1927-2017 (Fama-French monthly total
   return), 1990-2021 (S&P 500 daily price index) and 1999-2018 (S&P 500 intraday range),
   each tested with Newey-West, KVB fixed-b and a residual block bootstrap;
2. the four eras and the start-year sensitivity of the long trend;
3. extreme days per decade — fixed long-run σ and trailing σ, 3σ and 4σ — with exact Poisson
   intervals and a crisis-respecting count-trend test;
4. points versus percent on the S&P 500 level;
5. the decade people remember against the 1930s;
6. the risk manager's forecast race (long-run, recent, trend-extrapolated) out of sample;
7. the synthetic control — the trend bar must fire on a planted rise and not on the null;

then applies the pre-registered ``strategy.verdict``.

    python studies/1020-more-volatile-than-ever/examples/verify.py
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

from volhistory import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

# Pre-registered settings (fixed before the real run).
REMEMBERED = (2008, 2017)       # the last ten complete years of the century tape
THIRTIES = (1930, 1939)
DAILY_BUCKETS = (("1990-99", 1990, 1999), ("2000-09", 2000, 2009),
                 ("2010-19", 2010, 2019), ("2020-21", 2020, 2021))
ANNUAL_RACE = {"recent": 1, "trend_window": 20, "burn": 20}     # forecasts 1947-2017
MONTHLY_RACE = {"recent": 12, "trend_window": 60, "burn": 60}   # forecasts 1995-2021
START_MIN_LEN = 20


def _tr_row(name, tr):
    return {"series": name, **{k: tr[k] for k in ("n", "slope", "pct_per_decade", "t_ols",
                                                   "t_nw", "t_kvb", "p_kvb", "p_boot",
                                                   "rho1")}}


def report() -> dict:
    t0 = time.time()
    m = data.load_monthly()
    d = data.load_daily()
    o = data.load_ohlc()
    h: dict = {"as_of": data.AS_OF, "provenance": data.provenance()}
    for p in h["provenance"]:
        print(f"  {p['tape']:8s} {p['package']:14s} {p['dataset']:12s} {p['first']} -> "
              f"{p['last']}  rows {p['rows']:5d}  sha256 {p['sha256'][:12]}  fp {p['fingerprint']}")

    # ---------------------------------------------------------------- 1. trends
    print("\n=== 1. is volatility trending? ===")
    rv_c = st.annual_rv_from_monthly(m, data.FIRST_FULL_YEAR_MONTHLY,
                                     data.LAST_FULL_YEAR_MONTHLY)
    rv_d = st.annual_rv_from_daily(d)
    rv_p = st.annual_rv_parkinson(o)
    rv_o = st.annual_rv_from_daily(o["close"])
    h["century"] = st.trend_test(np.log(rv_c))
    h["daily"] = st.trend_test(np.log(rv_d))
    h["ohlc"] = st.trend_test(np.log(rv_p))
    h["ohlc_cc"] = st.trend_test(np.log(rv_o))
    h["century_span"] = f"{rv_c.index[0]}-{rv_c.index[-1]}"
    h["century_last"] = int(rv_c.index[-1])
    h["daily_span"] = f"{rv_d.index[0]}-{rv_d.index[-1]}"
    h["ohlc_span"] = f"{rv_p.index[0]}-{rv_p.index[-1]}"
    h["kvb_crit"] = st.kvb_critical()
    h["trend_rows"] = [
        _tr_row(f"Century, monthly total return ({h['century_span']})", h["century"]),
        _tr_row(f"Daily S&P 500 close-to-close ({h['daily_span']})", h["daily"]),
        _tr_row(f"Daily S&P 500 close-to-close, arch tape ({h['ohlc_span']})", h["ohlc_cc"]),
        _tr_row(f"Daily S&P 500 Parkinson range ({h['ohlc_span']})", h["ohlc"])]
    for r in h["trend_rows"]:
        print(f"  {r['series']:55s} {r['pct_per_decade']:+7.1%}/decade  t_ols {r['t_ols']:+5.2f}"
              f"  t_nw {r['t_nw']:+5.2f}  t_kvb {r['t_kvb']:+5.2f} (p {r['p_kvb']:.2f})"
              f"  boot p {r['p_boot']:.3f}")
    print(f"  KVB fixed-b 5% two-sided critical value: {h['kvb_crit']:.2f} (simulated)")
    h["rv_century"] = {int(k): float(v) for k, v in rv_c.items()}
    h["rv_daily"] = {int(k): float(v) for k, v in rv_d.items()}
    h["rv_parkinson"] = {int(k): float(v) for k, v in rv_p.items()}
    h["parkinson_over_cc"] = float((rv_p / rv_o).mean())

    # ---------------------------------------------------------------- 2. eras, starts
    print("\n=== 2. eras and start-year sensitivity ===")
    era = st.era_table(m, data.ERAS)
    print(era.round(4).to_string())
    h["eras"] = era.reset_index().to_dict("records")
    starts = st.trend_from_every_start(np.log(rv_c), min_len=START_MIN_LEN)
    sig_up = (starts["slope"] > 0) & (starts["t_nw"] >= 2) & (starts["p_kvb"] < 0.05)
    sig_dn = (starts["slope"] < 0) & (starts["t_nw"] <= -2) & (starts["p_kvb"] < 0.05)
    h["share_starts_up"] = float((starts["slope"] > 0).mean())
    h["share_starts_sig_up"] = float(sig_up.mean())
    h["share_starts_sig_down"] = float(sig_dn.mean())
    h["starts_first"], h["starts_last"] = int(starts.index[0]), int(starts.index[-1])
    h["starts"] = {int(k): float(v) for k, v in starts["pct_per_decade"].items()}
    h["starts_up_years"] = [int(y) for y in starts.index[sig_up]]
    print(f"  start years {h['starts_first']}-{h['starts_last']}: slope > 0 from "
          f"{h['share_starts_up']:.0%}, significantly up from {h['share_starts_sig_up']:.0%}, "
          f"significantly down from {h['share_starts_sig_down']:.0%}")
    dv = st.decade_vols(m)
    h["decades"] = dv.reset_index().rename(columns={"index": "decade"}).to_dict("records")
    print(dv.round(4).to_string())

    # ---------------------------------------------------------------- 3. extremes
    print("\n=== 3. extreme days ===")
    lr = st.log_returns_from_price(d)
    h["sigma_fixed_daily"] = float(lr.std(ddof=1))
    h["counts"] = {}
    for k in (3, 4):
        for mode in ("fixed", "trailing"):
            f = st.extreme_days(lr, k, mode)
            tab = st.count_table(f, DAILY_BUCKETS)
            ct = st.count_trend(f)
            key = f"{mode}{k}"
            h["counts"][key] = {"table": tab.reset_index().to_dict("records"),
                                "trend": {a: b for a, b in ct.items() if a != "yearly"}}
            if key == "fixed3":
                h["extremes_fixed3"] = h["counts"][key]["trend"]
                h["yearly_fixed3"] = {int(y): int(c) for y, c in ct["yearly"]["count"].items()}
            print(f"  |r| > {k}σ, {mode} σ: total {int(ct['total'])}, trend "
                  f"{ct['pct_per_decade']:+.0%}/decade, HAC z {ct['z_hac']:+.2f}, "
                  f"block-perm p {ct['p_perm']:.3f}")
            print("    " + tab.round(2).to_string().replace("\n", "\n    "))
    me = st.monthly_extremes_by_decade(m)
    h["monthly_extremes"] = me.reset_index().rename(columns={"index": "decade"}).to_dict("records")
    print("  months beyond 3σ (fixed σ, century tape):", me["count"].to_dict())

    # ---------------------------------------------------------------- 4. points vs percent
    print("\n=== 4. points versus percent ===")
    pp = st.points_vs_percent(d)
    h["points"] = {k: v for k, v in pp.items()
                   if k not in ("yearly", "top_points", "top_pct", "records_points",
                                "records_pct")}
    h["points"]["records_points"] = [(str(i.date()), float(v))
                                     for i, v in pp["records_points"].items()]
    h["points"]["records_pct"] = [(str(i.date()), float(v)) for i, v in pp["records_pct"].items()]
    h["points"]["top5_points"] = [(str(i.date()), float(v))
                                  for i, v in pp["top_points"].head(5).items()]
    h["points"]["top5_pct"] = [(str(i.date()), float(v)) for i, v in pp["top_pct"].head(5).items()]
    for lab, key in (("points", "trend_points"), ("percent", "trend_pct"),
                     ("level", "trend_level")):
        tr = pp[key]
        print(f"  mean |daily move| in {lab:8s}: {tr['pct_per_decade']:+7.1%}/decade "
              f"(NW t {tr['t_nw']:+.2f}, KVB p {tr['p_kvb']:.3f})")
    print(f"  top-20 point drops in the last decade ({pp['last_decade_start']}+): "
          f"{pp['top_points_share_last_decade']:.0%}; top-20 percent drops: "
          f"{pp['top_pct_share_last_decade']:.0%}")
    print(f"  'record' point drops after a 5y burn-in: {len(pp['records_points'])}; "
          f"record percent drops: {len(pp['records_pct'])}")
    print(f"  days with |move| > {pp['big_points']:.0f} points: first half "
          f"{pp['days_over_big_first_half']}, second half {pp['days_over_big_second_half']}")

    # ---------------------------------------------------------------- 5. recency
    print("\n=== 5. the decade people remember vs the 1930s ===")
    dc = st.decade_contrast(m, THIRTIES, REMEMBERED)
    h["thirties_vs_remembered"] = dc
    print(f"  1930-39 vol {dc['vol_a']:.1%} vs {REMEMBERED[0]}-{REMEMBERED[1]} {dc['vol_b']:.1%}"
          f"  ratio {dc['ratio']:.2f}  diff CI [{dc['ci_lo']:+.1%}, {dc['ci_hi']:+.1%}]  "
          f"p {dc['p']:.4f}")
    print(f"  months down 10%+: {dc['n_down10_a']} vs {dc['n_down10_b']}")

    # ---------------------------------------------------------------- 6. forecast race
    print("\n=== 6. the risk manager's forecast race ===")
    fa = st.forecast_race(rv_c ** 2, **ANNUAL_RACE)
    mv = st.monthly_rv_from_daily(d)
    fm = st.forecast_race(mv, **MONTHLY_RACE)
    h["trend_window_annual"] = ANNUAL_RACE["trend_window"]
    for name, f in (("race_annual", fa), ("race_monthly", fm)):
        h[name] = {"n": f["n"], "first": str(f["first"])[:10], "last": str(f["last"])[:10],
                   "mean_qlike": f["mean_qlike"], "rmse_vol": f["rmse_vol"], "dm": f["dm"],
                   "winner": f["winner"]}
        print(f"  {name}: {f['n']} forecasts {str(f['first'])[:10]} -> {str(f['last'])[:10]}")
        for mdl, v in f["mean_qlike"].items():
            print(f"     {mdl:9s} QLIKE {v:.3f}   RMSE(vol) {f['rmse_vol'][mdl]:.3f}")
        for k, v in f["dm"].items():
            print(f"     DM {k:20s} mean diff {v['mean_diff']:+.3f}  t {v['t']:+.2f}  p {v['p']:.3f}")

    # ---------------------------------------------------------------- 7. synthetic
    print("\n=== 7. synthetic control (SYNTHETIC — machinery proof, not market evidence) ===")
    h["synthetic"] = [st.synthetic_detection(s, seeds=range(40)) for s in (1.0, 0.5, 0.0)]
    for r in h["synthetic"]:
        print(f"  signal_strength {r['signal_strength']:.1f}: bar fires {r['fire_rate']:.0%} "
              f"(NW alone {r['nw_rate']:.0%}, naive OLS {r['ols_rate']:.0%}); mean slope "
              f"{r['mean_slope']:+.4f} vs true {r['true_slope']:+.4f}")
    hp = [st.synthetic_detection(0.0, seeds=range(40), monthly_phi=0.99)]
    h["synthetic_persistent_null"] = hp[0]
    print(f"  very persistent null (monthly phi 0.99): bar fires {hp[0]['fire_rate']:.0%}, "
          f"NW alone {hp[0]['nw_rate']:.0%}, naive OLS {hp[0]['ols_rate']:.0%}")

    h["_verdict"] = st.verdict(h)
    v = h["_verdict"]
    print("\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {v['signal']}   Tradability: {v['trad']}   Myth: {v['myth']}")
    print(f"  flags: {v['flags']}")
    h["runtime_s"] = time.time() - t0
    return h


def _f(x, fmt):
    return format(x, fmt) if x is not None and np.isfinite(x) else "—"


def results_md(h: dict) -> str:
    v = h["_verdict"]
    prov = "\n".join(
        f"| {p['tape']} | {p['label']} | `{p['package']}` / `{p['file']}` | "
        f"`{p['sha256'][:16]}…` | {p['first']} → {p['last']} | {p['rows']:,} | "
        f"`{p['fingerprint']}` |" for p in h["provenance"])
    trows = "\n".join(
        f"| {r['series']} | {r['n']} | **{r['pct_per_decade']:+.1%}** | {r['t_ols']:+.2f} | "
        f"{r['t_nw']:+.2f} | {r['t_kvb']:+.2f} | {r['p_kvb']:.2f} | {r['p_boot']:.3f} | "
        f"{r['rho1']:+.2f} |" for r in h["trend_rows"])
    erows = "\n".join(
        f"| {r['era']} | {r['months']} | **{r['vol']:.1%}** | [{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] | "
        f"{r['worst_month']:+.1%} | {_f(r['within_slope_pct_decade'], '+.1%')} | "
        f"{_f(r['within_t_nw'], '+.2f')} |" for r in h["eras"])
    drows = " | ".join(f"{r['decade']}: {r['vol']:.1%}" for r in h["decades"])

    def ctab(key):
        c = h["counts"][key]
        rows = "\n".join(
            f"| {r['bucket']} | {r['days']:,} | {r['count']} | **{r['rate']:.1f}** | "
            f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] |" for r in c["table"])
        t = c["trend"]
        return rows, t

    ctext = []
    for key, title in (("fixed3", "3σ, fixed long-run σ"), ("fixed4", "4σ, fixed long-run σ"),
                       ("trailing3", "3σ, trailing one-year σ"),
                       ("trailing4", "4σ, trailing one-year σ")):
        rows, t = ctab(key)
        ctext.append(
            f"**{title}** — {t['total']:.0f} days; trend {t['pct_per_decade']:+.0%}/decade, HAC "
            f"z = {t['z_hac']:+.2f}, block-permutation p = {t['p_perm']:.3f}\n\n"
            f"| Years | Sessions | Count | Per 1,000 sessions | Exact 95% CI |\n"
            f"|---|--:|--:|--:|--:|\n{rows}\n")
    ctext = "\n".join(ctext)
    mex = " | ".join(f"{r['decade']}: {int(r['count'])}" for r in h["monthly_extremes"])
    pp = h["points"]
    tp, tc, tl = pp["trend_points"], pp["trend_pct"], pp["trend_level"]
    rec_pts = ", ".join(f"{dte} ({val:+.0f})" for dte, val in pp["records_points"])
    rec_pct = ", ".join(f"{dte} ({val:+.1%})" for dte, val in pp["records_pct"])
    top_pts = ", ".join(f"{dte} ({val:+.0f})" for dte, val in pp["top5_points"])
    top_pct = ", ".join(f"{dte} ({val:+.1%})" for dte, val in pp["top5_pct"])
    dc = h["thirties_vs_remembered"]

    def race(key, unit):
        r = h[key]
        q, e = r["mean_qlike"], r["rmse_vol"]
        best = min(q, key=q.get)
        rows = "\n".join(
            f"| {mdl} | {'**' if mdl == best else ''}{q[mdl]:.3f}{'**' if mdl == best else ''} | "
            f"{e[mdl]:.3f} |" for mdl in ("LONG-RUN", "RECENT", "TREND", "WINDOW", "BLEND"))
        dms = "\n".join(
            f"| {k.replace('_vs_', ' vs ')} | {dmv['mean_diff']:+.3f} | {dmv['t']:+.2f} | "
            f"{dmv['p']:.3f} |" for k, dmv in r["dm"].items())
        return (f"{r['n']} out-of-sample {unit}, {r['first']} → {r['last']}.\n\n"
                f"| Forecast | Mean QLIKE | RMSE (vol) |\n|---|--:|--:|\n{rows}\n\n"
                f"| Diebold-Mariano (QLIKE) | Mean loss diff | t | p |\n|---|--:|--:|--:|\n{dms}\n")

    syn = "\n".join(
        f"| {r['signal_strength']:.1f} | {r['n_worlds']} | **{r['fire_rate']:.0%}** | "
        f"{r['nw_rate']:.0%} | {r['ols_rate']:.0%} | {r['mean_slope']:+.4f} | "
        f"{r['true_slope']:+.4f} |" for r in h["synthetic"])
    hp = h["synthetic_persistent_null"]
    syn += (f"\n| 0.0, very persistent (φ = 0.99/month) | {hp['n_worlds']} | "
            f"**{hp['fire_rate']:.0%}** | {hp['nw_rate']:.0%} | {hp['ols_rate']:.0%} | "
            f"{hp['mean_slope']:+.4f} | {hp['true_slope']:+.4f} |")
    c = h["century"]
    return f"""# Results — Study 1020 (More Volatile Than Ever?) on the frozen bundled tapes

*Generated by [`examples/verify.py`](../examples/verify.py) in {h['runtime_s']:.0f} s. Study as-of
**{h['as_of']}** (the last complete year of the latest tape); each tape carries its own as-of
below. Every real number on this page comes from these three files and nothing else.*

## Data provenance

| Tape | What it is | Package / file | SHA-256 pin | Span used | Rows | Fingerprint |
|---|---|---|---|---|--:|---|
{prov}

- **Total return vs price index.** The century tape is a *total* return (dividends in); both
  S&P 500 tapes are *price* indices. Dividends shift the level of a return, not its dispersion,
  so volatility comparisons across tapes are fair; the points-versus-percent section is about
  the index *level*, which is why it uses the price index.
- **Partial periods dropped.** Annual statistics on the century tape use complete years
  {data.FIRST_FULL_YEAR_MONTHLY}-{data.LAST_FULL_YEAR_MONTHLY} (1926 starts in July; 2018 stops
  in November). The skfolio tape ends 2022-12-28, so 2022 is dropped and daily statistics stop
  at 2021-12-31.

## 1. Is volatility trending?

Annual realised volatility, log scale, linear trend. Three robust tests: Newey-West (data-
dependent bandwidth), Kiefer-Vogelsang-Bunzel fixed-b (Bartlett, bandwidth = sample; the 5%
two-sided critical value, simulated, is **{h['kvb_crit']:.2f}**, not 1.96), and a residual
circular block bootstrap under the null of no trend.

| Series | Years | Trend per decade | naive t | NW t | KVB t | KVB p | Boot p | ρ₁ resid |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{trows}

The century slope is **{c['pct_per_decade']:+.1%} per decade** — *downward*, carried by the
1930s at the start of the sample. Newey-West and the bootstrap call the decline significant;
the fixed-b test, which is the one honest about how persistent volatility is, does not (p =
{c['p_kvb']:.2f}). Read it as "no upward trend, and no reliable downward one either." The
range-based estimator averages {h['parkinson_over_cc']:.2f}× close-to-close over the same years
(the usual small downward bias of a range estimator on an index, plus no overnight gap) and
tells the same story: no rise. Over 1999-2018 both daily estimators even slope *down* — a
window that opens in the dot-com bust and closes in the calm of 2017 — which is the
start-date effect the next paragraph measures.

**Start-year sensitivity.** Refitting the century trend from every start year
{h['starts_first']}-{h['starts_last']} to {h['century_last']}: the slope is positive from
{h['share_starts_up']:.0%} of starts, significantly positive (the full bar) from
**{h['share_starts_sig_up']:.0%}**, significantly negative from {h['share_starts_sig_down']:.0%}.

## 2. Four eras

Annualised RMS volatility of monthly total returns, 95% circular-block-bootstrap interval
(12-month blocks), and the trend *within* each era (≈20 annual points — indicative only).

| Era | Months | Volatility | 95% CI | Worst month | Within-era trend/decade | NW t |
|---|--:|--:|--:|--:|--:|--:|
{erows}

By calendar decade: {drows}.

## 3. Are extreme days getting more frequent?

Daily S&P 500 price index, log returns, 1990-2021. Fixed σ = the full-sample daily standard deviation ({h['sigma_fixed_daily']:.2%}) — a
descriptive yardstick; trailing σ = the previous 252 sessions, ending the day before.

{ctext}
The count trend is a Poisson regression of yearly counts on the year (exposure offset, HAC
z), and the decisive p-value is a **block permutation** of 3-year blocks — extreme days arrive
in crisis clusters, so a test that treats them as independent finds trends in two bursts.
Against a fixed σ the decade rates are dominated by 2008-09 and 2020; against a trailing σ the
rate of *surprising* days drifts up (calm regimes broken suddenly), but no version clears the
pre-registered bar.

The century tape cannot see days, but it can see months: months beyond 3σ (fixed σ) per
decade — {mex}.

## 4. Points versus percent — why it *feels* more volatile

Per calendar year on the S&P 500 price index: mean absolute daily move in **index points**,
in **percent**, and the mean index level.

| Mean per year of | Trend per decade | NW t | KVB p |
|---|--:|--:|--:|
| \\|daily move\\|, points | **{tp['pct_per_decade']:+.0%}** | {tp['t_nw']:+.2f} | {tp['p_kvb']:.3f} |
| \\|daily move\\|, percent | **{tc['pct_per_decade']:+.1%}** | {tc['t_nw']:+.2f} | {tc['p_kvb']:.3f} |
| index level | {tl['pct_per_decade']:+.0%} | {tl['t_nw']:+.2f} | {tl['p_kvb']:.3f} |

Since ΔP = P × r, the points trend is the level trend plus the percent trend, and the percent
trend is nil. {pp['top_points_share_last_decade']:.0%} of the 20 largest point drops on the
tape fall in its last ten years ({pp['last_decade_start']}-2021), against
{pp['top_pct_share_last_decade']:.0%} of the 20 largest percent drops. The S&P first moved
more than {pp['big_points']:.0f} points in a day in {pp['first_year_over_big']}; such days
number {pp['days_over_big_first_half']} in the first half of the tape and
{pp['days_over_big_second_half']} in the second.

- Largest point drops: {top_pts}.
- Largest percent drops: {top_pct}.
- "Biggest point drop ever" days (after a 5-year burn-in): {len(pp['records_points'])} —
  {rec_pts}.
- "Biggest percent drop ever" days: {len(pp['records_pct'])} — {rec_pct}.

(1987's −20.5% is before this tape starts; on a percent basis it remains the record.)

## 5. The decade people remember vs the 1930s

Century tape, monthly total returns:

| | {THIRTIES[0]}-{THIRTIES[1]} | {REMEMBERED[0]}-{REMEMBERED[1]} |
|---|--:|--:|
| Annualised volatility | **{dc['vol_a']:.1%}** | **{dc['vol_b']:.1%}** |
| Worst month | {dc['worst_month_a']:+.1%} | {dc['worst_month_b']:+.1%} |
| Months down 10% or more | {dc['n_down10_a']} | {dc['n_down10_b']} |

Ratio **{dc['ratio']:.2f}×**; difference {dc['diff']:+.1%}, 95% block-bootstrap CI
[{dc['ci_lo']:+.1%}, {dc['ci_hi']:+.1%}], p = {dc['p']:.4f}. The decade that contains the
global financial crisis — the most volatile stretch most readers have lived through — was
less than half as volatile as the 1930s.

## 6. Could you use it? The risk manager's forecast race

Forecast next period's variance from information to the end of this period; score by QLIKE
(lower is better). Pre-registered contenders: **LONG-RUN** (expanding mean), **RECENT** (last
year / last 12 months), **TREND** (OLS line through the last {h['trend_window_annual']} years
/ 60 months of log vol, extrapolated one step — the claim as a rule). References: **WINDOW**
(the same window as TREND, no slope) and **BLEND** (½ LONG-RUN + ½ RECENT).

**Century tape (annual):** {race('race_annual', 'years')}
**Daily tape (monthly realised variance):** {race('race_monthly', 'months')}

## 7. Synthetic control (SYNTHETIC — a machinery proof, never market evidence)

Monthly stochastic-volatility worlds, 90 years each, log volatility a persistent AR(1); at
`signal_strength = 1` a trend is planted that triples volatility across the sample. The
pre-registered bar (slope > 0, NW t ≥ 2, KVB p < 0.05) must fire on the plant and stay quiet
on the null.

| signal_strength | Worlds | Full bar fires | NW t ≥ 2 alone | naive OLS t ≥ 2 | Mean slope | True slope |
|---|--:|--:|--:|--:|--:|--:|
{syn}

## Caveats

- **One market, one country.** The US is the market that did best in the twentieth century; its
  volatility history is not the world's. The claim is usually made about the US, so that is the
  test.
- **Monthly data for the long view.** Realised volatility from twelve monthly returns is a noisy
  estimate of a year's volatility, and it cannot see intramonth crashes (October 1987 shows up as
  one bad month). The daily tapes cover only 1990-2021.
- **The fixed-σ extreme-day count uses the full-sample σ** — a descriptive yardstick with
  look-ahead, deliberately; the trailing-σ count is the one a risk manager could have computed.
- **The 2020-21 bucket is two years**, so its interval is wide; do not read the point rate as a
  new regime.
- **Index composition changed.** The S&P 500 of 2021 is more concentrated in large technology
  firms than the index of 1990; a constant-volatility *market* and a constant-volatility
  *index* are not the same statement. Neither tape can separate the two.
- **Individual stocks are a different question.** Campbell, Lettau, Malkiel & Xu (2001) found
  rising *idiosyncratic* volatility through the 1990s with no trend in market volatility. This
  study measures the market only.
- **The fixed-b p-value is simulated** (6,000 draws, T = 300, i.i.d. Gaussian errors); its
  finite-sample size under very persistent volatility is better than Newey-West's but not
  exact.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

**More volatile than ever? {v['myth']}.** The 1930s were {dc['ratio']:.1f}× as volatile as
{REMEMBERED[0]}-{REMEMBERED[1]} (block-bootstrap p = {dc['p']:.4f}); the headline "records"
are point records, which a constant-percent market sets automatically as it grows.

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1020-more-volatile-than-ever](../README.md). Not investment advice.*
"""


def headline(h: dict) -> dict:
    v = h["_verdict"]
    c, d, x = h["century"], h["daily"], h["extremes_fixed3"]
    return {"study": "1020-more-volatile-than-ever", "as_of": h["as_of"],
            "fingerprints": {p["tape"]: p["fingerprint"] for p in h["provenance"]},
            "signal": v["signal"], "trad": v["trad"], "myth": v["myth"],
            "century_pct_per_decade": c["pct_per_decade"], "century_t_nw": c["t_nw"],
            "century_p_kvb": c["p_kvb"], "daily_pct_per_decade": d["pct_per_decade"],
            "daily_t_nw": d["t_nw"], "extremes_trend_p_perm": x["p_perm"],
            "thirties_ratio": h["thirties_vs_remembered"]["ratio"],
            "points_pct_per_decade": h["points"]["trend_points"]["pct_per_decade"],
            "percent_pct_per_decade": h["points"]["trend_pct"]["pct_per_decade"],
            "qlike_annual": h["race_annual"]["mean_qlike"],
            "qlike_monthly": h["race_monthly"]["mean_qlike"],
            "synthetic_fire_planted": h["synthetic"][0]["fire_rate"],
            "synthetic_fire_null": h["synthetic"][2]["fire_rate"],
            "runtime_s": h["runtime_s"]}


def main() -> None:
    h = report()
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    print("##HEADLINE## " + json.dumps(headline(h), default=float))


if __name__ == "__main__":
    main()
