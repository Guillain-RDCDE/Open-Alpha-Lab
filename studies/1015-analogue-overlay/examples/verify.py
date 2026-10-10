"""Real-tape verification — Study 1015 (The 1929 Overlay). Regenerates docs/results.md.

Runs the analogue forecaster out of sample on both real tapes — the Fama-French monthly total
return (1926-2018, the tape that contains 1929) and the daily S&P 500 price index (1990-2022) —
across the full grid of windows, horizons, analogue counts and matching metrics, reports every
combination, corrects for the search (Holm across all predictive tests, White's Reality Check
across all timing rules), and sets the famous-looking matches against what a random walk with
the same drift and volatility produces when searched the same way. A synthetic tape with a
genuinely recurring template shows the detector has power when there is something to find.

    python studies/1015-analogue-overlay/examples/verify.py

Everything is offline: both tapes come from ``quantlab.bundled`` (SHA-256 pinned).
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

from overlay import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

OOS_MONTHLY = "1950-01-31"   # 23.5 years of history before the first forecast
OOS_DAILY = "2000-01-03"     # 10 years of history before the first forecast
DAILY_STEP = 5               # the daily forecaster re-runs weekly
COST_BPS = 10.0              # one-way, on traded NAV
TEMPLATE_END = "1929-08-31"  # the last month-end before the 1929 crash
SHOWCASE_END = "2014-01-31"  # an early-2014 overlay, the kind that circulated widely
SYN_STRENGTHS = (0.0, 0.25, 0.5, 1.0)
SYN_SEEDS = tuple(range(1015, 1035))


def _key(spec: dict) -> tuple:
    return (spec["metric"], spec["L"], spec["h"], spec["k"])


def _row(tbl: pd.DataFrame, spec: dict) -> dict:
    q = tbl[(tbl["metric"] == spec["metric"]) & (tbl["L"] == spec["L"])
            & (tbl["h"] == spec["h"]) & (tbl["k"] == spec["k"])]
    return q.iloc[0].to_dict()


def report() -> dict:
    t_start = time.time()
    m = data.load_monthly()
    d = data.load_daily()
    mm = data.tape_moments(m["lr"], data.MONTHS_PER_YEAR)
    dm = data.tape_moments(d["lr"], data.TRADING_DAYS_PER_YEAR)
    h: dict = {"as_of": data.AS_OF, "as_of_monthly": data.AS_OF_MONTHLY,
               "as_of_daily": data.AS_OF_DAILY,
               "fp_monthly": data.fingerprint(m[["ret", "rf"]]),
               "fp_daily": data.fingerprint(d[["px"]]),
               "m_first": str(m.index[0].date()), "m_last": str(m.index[-1].date()),
               "d_first": str(d.index[0].date()), "d_last": str(d.index[-1].date()),
               "m_n": int(len(m)), "d_n": int(len(d)), "m_mom": mm, "d_mom": dm,
               "provenance": data.PROVENANCE}
    print(f"as-of {data.AS_OF}  monthly fp {h['fp_monthly']}  daily fp {h['fp_daily']}")
    print(f"  monthly TR {h['m_first']}..{h['m_last']} ({len(m)} months), drift "
          f"{mm['mu_ann']:.2%}/yr, vol {mm['sigma_ann']:.2%}/yr")
    print(f"  daily price {h['d_first']}..{h['d_last']} ({len(d)} days), drift "
          f"{dm['mu_ann']:.2%}/yr, vol {dm['sigma_ann']:.2%}/yr")

    # ------------------------------------------------------------------ 2
    print("\n=== 2. how cheap is a 0.9? two INDEPENDENT random walks ===")
    pc = {}
    for lbl, L, mo in (("monthly, 24 months", 24, mm), ("monthly, 36 months", 36, mm),
                       ("daily, 250 days", 250, dm), ("daily, 63 days", 63, dm)):
        p = st.pair_correlation_null(L, mo["mu"], mo["sigma"], n_sims=20000)
        pc[lbl] = {k: v for k, v in p.items() if k.startswith("p_")}
        pc[lbl]["median_price"] = float(np.median(p["price"]))
        pc[lbl]["iqr_price"] = [float(np.percentile(p["price"], 25)),
                                float(np.percentile(p["price"], 75))]
        print(f"  {lbl:20s} P(price corr>0.9) {p['p_price_gt_09']:.1%}  >0.7 "
              f"{p['p_price_gt_07']:.1%}  |  P(return corr>0.5) {p['p_returns_gt_05']:.2%}")
    h["pair_null"] = pc

    # ------------------------------------------------------------------ 3
    print("\n=== 3. searching makes it cheaper: best match in the whole past ===")
    bm = {}
    for lbl, n, L, mo, metric in (("monthly logprice L24", len(m), 24, mm, "logprice"),
                                  ("monthly zret L24", len(m), 24, mm, "zret"),
                                  ("daily logprice L250", len(d), 250, dm, "logprice")):
        b = st.best_match_null(n, L, mo["mu"], mo["sigma"], metric=metric,
                               n_sims=300 if n < 2000 else 100)
        bm[lbl] = {"p5": float(np.percentile(b, 5)), "p50": float(np.median(b)),
                   "p95": float(np.percentile(b, 95)), "share_gt_09": float((b > 0.9).mean())}
        print(f"  {lbl:22s} null best match: median {bm[lbl]['p50']:.3f}  "
              f"[5%..95%] {bm[lbl]['p5']:.3f}..{bm[lbl]['p95']:.3f}  "
              f"P(>0.9) {bm[lbl]['share_gt_09']:.0%}")
    h["best_match_null"] = bm

    # like-for-like: same dates, same search, random walks vs the real tape
    idx_m = st.eval_positions(m.index, OOS_MONTHLY)
    idx_d = st.eval_positions(d.index, OOS_DAILY, DAILY_STEP)
    lfl = {}
    for lbl, tape, idx, L, mo, metric, nsim in (
            ("monthly logprice L24", m, idx_m, 24, mm, "logprice", 20),
            ("monthly zret L24", m, idx_m, 24, mm, "zret", 20),
            ("daily logprice L250", d, idx_d, 250, dm, "logprice", 3)):
        real = st.analogue_search(tape["logp"].to_numpy(), L, idx, k_max=1,
                                  metric=metric)["corr"][:, 0]
        null = st.null_search_distribution(len(tape), L, mo["mu"], mo["sigma"], idx,
                                           metric=metric, n_sims=nsim)
        med_null = np.nanmedian(null, axis=1)
        lfl[lbl] = {"real_median": float(np.nanmedian(real)),
                    "real_share_gt_09": float(np.nanmean(real > 0.9)),
                    "null_median": float(np.nanmedian(null)),
                    "null_share_gt_09": float(np.nanmean(null > 0.9)),
                    "null_median_lo": float(np.min(med_null)),
                    "null_median_hi": float(np.max(med_null)), "n_sims": nsim,
                    "real_series": real.tolist() if lbl.startswith("monthly logprice")
                    else None}
        print(f"  {lbl:22s} real median best match {lfl[lbl]['real_median']:.3f} "
              f"(>0.9 on {lfl[lbl]['real_share_gt_09']:.0%} of dates) | random walk "
              f"{lfl[lbl]['null_median']:.3f} (>0.9 on {lfl[lbl]['null_share_gt_09']:.0%}); "
              f"null medians span {lfl[lbl]['null_median_lo']:.3f}..{lfl[lbl]['null_median_hi']:.3f}")
    # richer nulls for the monthly price-path search: keep the real vol path / short memory
    lr_m = m["lr"].to_numpy()
    vp = st.null_search_paths(st.vol_path_walks(lr_m, 10), 24, idx_m)
    bb = st.null_search_paths(st.block_bootstrap_walks(lr_m, 10, mean_block=6.0), 24, idx_m)
    lfl["monthly logprice L24"]["volpath_median"] = float(np.nanmedian(vp))
    lfl["monthly logprice L24"]["block_median"] = float(np.nanmedian(bb))
    print(f"  richer nulls (monthly L24): real-vol-path walk {np.nanmedian(vp):.3f}, "
          f"6-month block bootstrap {np.nanmedian(bb):.3f}")
    h["like_for_like"] = lfl
    h["m_match_median"] = lfl["monthly logprice L24"]["real_median"]
    h["null_match_median"] = lfl["monthly logprice L24"]["null_median"]

    # ------------------------------------------------------------------ 4
    print("\n=== 4. the 1929 overlay, tested directly ===")
    ov = st.template_overlay(m["logp"], TEMPLATE_END, 24, 12)
    eps = {th: st.overlay_episodes(ov, th) for th in (0.8, 0.9, 0.95)}
    pos = int(np.searchsorted(m.index.values, np.datetime64(pd.Timestamp(TEMPLATE_END))))
    tmpl = m["logp"].to_numpy()[pos - 23: pos + 1]
    tn = st.template_null_share(tmpl, mm["mu"], mm["sigma"])
    for th, e in eps.items():
        print(f"  match >= {th:.2f}: {e['n_match']} of {e['n_dates']} months "
              f"({e['share_match']:.0%}; random walk {tn[th]:.0%}) | next-12m log return "
              f"{e['fwd_match']:+.1%} vs {e['fwd_all']:+.1%} all (HAC t {e['diff_t']:+.2f}) | "
              f"fell 20%+: {e['crash_rate_match']:.1%} vs {e['crash_rate_all']:.1%}")
    ovd = ov.dropna()
    trail = (m["logp"] - m["logp"].shift(24)).reindex(ovd.index).to_numpy()
    ind = (ovd["corr"] >= 0.9).to_numpy(dtype=float)
    res = ind - np.polyval(np.polyfit(trail, ind, 1), trail)
    h["overlay_trailing"] = {
        "corr_match_trailing": float(np.corrcoef(ovd["corr"], trail)[0, 1]),
        "t_trailing": st.ols_hac(ovd["fwd"].to_numpy(), trail, 12)["t_b"],
        "t_match_orth": st.ols_hac(ovd["fwd"].to_numpy(), res, 12)["t_b"]}
    print(f"  the match is mostly the 24-month run-up: corr(match, trailing return) "
          f"{h['overlay_trailing']['corr_match_trailing']:.2f}; match t after removing it "
          f"{h['overlay_trailing']['t_match_orth']:+.2f}")
    sc = ov.loc[SHOWCASE_END]
    h["overlay"] = {str(k): v for k, v in eps.items()}
    h["overlay_null"] = {str(k): v for k, v in tn.items() if k != "corr"}
    h["showcase"] = {"date": SHOWCASE_END, "corr": float(sc["corr"]), "fwd": float(sc["fwd"])}
    top = ov.dropna().sort_values("corr", ascending=False).head(8)
    h["overlay_top"] = [{"date": str(i.date()), "corr": float(r["corr"]),
                         "fwd": float(r["fwd"])} for i, r in top.iterrows()]
    h["crash_rate_match"] = eps[0.9]["crash_rate_match"]
    h["crash_rate_all"] = eps[0.9]["crash_rate_all"]
    print(f"  {SHOWCASE_END}: 24-month path vs the 1927-29 run-up, corr {sc['corr']:.3f}; "
          f"the next 12 months returned {np.expm1(sc['fwd']):+.1%}")

    # ------------------------------------------------------------------ 5-7
    print("\n=== 5-7. the forecaster: full sweep, out of sample ===")
    sm = st.sweep(m, st.GRID_MONTHLY, data.MONTHS_PER_YEAR, OOS_MONTHLY, step=1,
                  cost_bps=COST_BPS, keep=(_key(st.HEADLINE_MONTHLY),))
    sd = st.sweep(d, st.GRID_DAILY, data.TRADING_DAYS_PER_YEAR, OOS_DAILY, step=DAILY_STEP,
                  cost_bps=COST_BPS, keep=(_key(st.HEADLINE_DAILY),))
    tm, td = sm["table"].copy(), sd["table"].copy()
    tm["tape"], td["tape"] = "monthly", "daily"
    allt = pd.concat([tm, td], ignore_index=True)
    allt["p_two"] = 2.0 * np.minimum(allt["p"], 1.0 - allt["p"])
    allt["p_holm"] = st.holm(allt["p"].to_numpy())
    allt["p_holm_two"] = st.holm(allt["p_two"].to_numpy())
    n_combos = int(len(allt))
    h["n_combos"], h["n_combos_m"], h["n_combos_d"] = n_combos, int(len(tm)), int(len(td))
    h["share_raw_sig"] = float((allt["p"] < 0.05).mean())
    h["n_raw_sig"] = int((allt["p"] < 0.05).sum())
    h["holm_min_p"] = float(allt["p_holm"].min())
    h["bonf_min_p"] = float(min(1.0, allt["p"].min() * n_combos))
    best = allt.loc[allt["t"].idxmax()]
    worst = allt.loc[allt["t"].idxmin()]
    h["best_combo"] = {k: (v.item() if hasattr(v, "item") else v)
                       for k, v in best[["tape", "metric", "L", "h", "k", "t", "corr", "p",
                                         "p_holm"]].items()}
    h["worst_combo"] = {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in worst[["tape", "metric", "L", "h", "k", "t", "corr",
                                           "p_two", "p_holm_two"]].items()}
    h["n_neg_sig_two"] = int(((allt["t"] < 0) & (allt["p_holm_two"] < 0.05)).sum())
    h["n_pos_sig_two"] = int(((allt["t"] > 0) & (allt["p_holm_two"] < 0.05)).sum())
    h["table"] = allt.drop(columns=[]).to_dict("records")
    print(f"  {n_combos} combinations ({len(tm)} monthly, {len(td)} daily)")
    print(f"  raw one-sided p<0.05: {h['n_raw_sig']} ({h['share_raw_sig']:.1%}; chance 5%)")
    print(f"  best: {best['tape']} {best['metric']} L{best['L']} h{best['h']} k{best['k']} "
          f"t {best['t']:+.2f}, Holm p {best['p_holm']:.2f}")
    print(f"  most NEGATIVE: {worst['tape']} {worst['metric']} L{worst['L']} h{worst['h']} "
          f"k{worst['k']} t {worst['t']:+.2f}, two-sided Holm p {worst['p_holm_two']:.4f}")

    hm, hd = _row(allt[allt.tape == "monthly"], st.HEADLINE_MONTHLY), \
        _row(allt[allt.tape == "daily"], st.HEADLINE_DAILY)
    km = sm["kept"][_key(st.HEADLINE_MONTHLY)]
    kd = sd["kept"][_key(st.HEADLINE_DAILY)]
    ci_m = st.stationary_bootstrap_corr(km["fc"], km["real"], n_boot=500,
                                        mean_block=2 * st.HEADLINE_MONTHLY["h"])
    ci_d = st.stationary_bootstrap_corr(kd["fc"], kd["real"], n_boot=500,
                                        mean_block=2 * st.HEADLINE_DAILY["h"] / DAILY_STEP)
    for tag, row, kept, ci in (("m", hm, km, ci_m), ("d", hd, kd, ci_d)):
        r = kept["rule"]
        h[f"{tag}_head"] = {k: row[k] for k in ("n", "corr", "slope", "t", "p", "p_holm",
                                                "hit", "hit_t", "hit_raw", "up_share",
                                                "oos_r2", "match_corr_median")}
        h[f"{tag}_head"]["ci"] = list(ci)
        h[f"{tag}_rule"] = {k: r[k] for k in ("sharpe_net", "sharpe_gross", "sharpe_bh",
                                              "ann_ret_net", "ann_ret_bh", "alpha_ann",
                                              "alpha_t", "beta", "time_in_market",
                                              "turnover_per_year", "max_dd", "max_dd_bh",
                                              "n", "start")}
        h[f"{tag}_head_t"] = float(row["t"])
        h[f"{tag}_head_corr"] = float(row["corr"])
        h[f"{tag}_head_r2"] = float(row["oos_r2"])
        h[f"{tag}_head_alpha_t"] = float(r["alpha_t"])
        h[f"{tag}_head_sharpe_net"] = float(r["sharpe_net"])
        h[f"{tag}_bh_sharpe"] = float(r["sharpe_bh"])
    print(f"  headline monthly {st.HEADLINE_MONTHLY}: t {hm['t']:+.2f} corr {hm['corr']:+.3f} "
          f"[{ci_m[0]:+.2f},{ci_m[1]:+.2f}] hit {hm['hit']:.1%} OOS R2 {hm['oos_r2']:+.1%}")
    print(f"  headline daily   {st.HEADLINE_DAILY}: t {hd['t']:+.2f} corr {hd['corr']:+.3f} "
          f"[{ci_d[0]:+.2f},{ci_d[1]:+.2f}] hit {hd['hit']:.1%} OOS R2 {hd['oos_r2']:+.1%}")
    for tag in ("m", "d"):
        r = h[f"{tag}_rule"]
        print(f"  rule {tag}: Sharpe net {r['sharpe_net']:.2f} gross {r['sharpe_gross']:.2f} "
              f"vs B&H {r['sharpe_bh']:.2f}; alpha {r['alpha_ann']:+.2%} t {r['alpha_t']:+.2f}; "
              f"in market {r['time_in_market']:.0%}, turnover {r['turnover_per_year']:.1f}/yr")

    rc_m = st.reality_check(sm["diffs"], data.MONTHS_PER_YEAR, n_boot=1000)
    rc_d = st.reality_check(sd["diffs"], data.TRADING_DAYS_PER_YEAR, n_boot=1000)
    rv_m = st.reality_check(sm["diffs_vm"], data.MONTHS_PER_YEAR, n_boot=1000)
    rv_d = st.reality_check(sd["diffs_vm"], data.TRADING_DAYS_PER_YEAR, n_boot=1000)
    h["rc_ret_p_monthly"], h["rc_ret_p_daily"] = (rc_m["reality_check_pvalue"],
                                                  rc_d["reality_check_pvalue"])
    h["rc_vm_p_monthly"], h["rc_vm_p_daily"] = (rv_m["reality_check_pvalue"],
                                                rv_d["reality_check_pvalue"])
    h["rc_vm_best_monthly"], h["rc_vm_best_daily"] = rv_m["best_rule"], rv_d["best_rule"]
    h["rc_vm_ir_monthly"], h["rc_vm_ir_daily"] = (rv_m["observed_max_sharpe"],
                                                  rv_d["observed_max_sharpe"])
    # the verdict reads the kinder of the two races per tape
    h["rc_p_monthly"] = min(h["rc_ret_p_monthly"], h["rc_vm_p_monthly"])
    h["rc_p_daily"] = min(h["rc_ret_p_daily"], h["rc_vm_p_daily"])
    h["rc_best_monthly"], h["rc_best_daily"] = rc_m["best_rule"], rc_d["best_rule"]
    h["rc_ir_monthly"], h["rc_ir_daily"] = rc_m["observed_max_sharpe"], rc_d["observed_max_sharpe"]
    rules = allt.groupby("tape").agg(n=("sharpe_net", "size"),
                                     beat=("sharpe_net", lambda s: 0),
                                     med_sharpe=("sharpe_net", "median"),
                                     max_sharpe=("sharpe_net", "max"),
                                     bh=("sharpe_bh", "first"))
    for tape in ("monthly", "daily"):
        q = allt[allt.tape == tape]
        rules.loc[tape, "beat"] = int((q["sharpe_net"] > q["sharpe_bh"]).sum())
    h["rules_summary"] = rules.reset_index().to_dict("records")
    print(f"  Reality Check, vol-matched Sharpe race: monthly p {h['rc_vm_p_monthly']:.3f} "
          f"(best {rv_m['best_rule']}), daily p {h['rc_vm_p_daily']:.3f} (best {rv_d['best_rule']})")
    print(f"  Reality Check: monthly p {h['rc_ret_p_monthly']:.3f} (best {rc_m['best_rule']}, IR "
          f"{rc_m['observed_max_sharpe']:+.2f}); daily p {h['rc_p_daily']:.3f} (best "
          f"{rc_d['best_rule']}, IR {rc_d['observed_max_sharpe']:+.2f})")

    # ------------------------------------------------------------------ 8 wrong-way probe
    print("\n=== 8. the wrong-way result, probed ===")
    w = h["worst_combo"]
    tape = d if w["tape"] == "daily" else m
    step = DAILY_STEP if w["tape"] == "daily" else 1
    idx = idx_d if w["tape"] == "daily" else idx_m
    logp = tape["logp"].to_numpy()
    srch = st.analogue_search(logp, int(w["L"]), idx, k_max=int(w["k"]), metric=w["metric"],
                              gap=max(int(w["L"]), int(w["h"])))
    fc = st.analogue_forecast(logp, srch, int(w["h"]), int(w["k"]))
    y = st.realised_forward(logp, idx, int(w["h"]))
    bmk = st.historical_drift_forecast(logp, idx, int(w["h"]))
    base_l = max(1, int(np.ceil(int(w["h"]) / step)))
    probe = {"lags": {}, "halves": {}}
    for mult in (1, 2, 4):
        probe["lags"][mult] = st.evaluate_forecast(fc, y, bmk, base_l * mult)["t"]
    dates = tape.index[idx]
    mid = dates[len(dates) // 2]
    for nm, sel in (("first half", dates < mid), ("second half", dates >= mid)):
        e = st.evaluate_forecast(fc[sel], y[sel], bmk[sel], base_l)
        probe["halves"][nm] = {"t": e["t"], "corr": e["corr"], "from": str(dates[sel][0].date())}
    trail = logp[idx] - logp[np.maximum(idx - int(w["h"]), 0)]
    ok = np.isfinite(y) & np.isfinite(fc)
    resid = fc[ok] - np.polyval(np.polyfit(trail[ok], fc[ok], 1), trail[ok])
    probe["t_orth_trailing"] = st.ols_hac(y[ok], resid, base_l)["t_b"]
    contra = st.timing_rule(tape, idx, -fc, 252 if w["tape"] == "daily" else 12,
                            cost_bps=COST_BPS)
    probe["contrarian_sharpe"] = contra["sharpe_net"]
    probe["contrarian_bh"] = contra["sharpe_bh"]
    probe["contrarian_alpha_t"] = contra["alpha_t"]
    h["wrong_way"] = probe
    print(f"  {w['tape']} {w['metric']} L{w['L']} h{w['h']} k{w['k']}: t at 1x/2x/4x lags "
          + " / ".join(f"{v:+.2f}" for v in probe["lags"].values())
          + f"; halves " + ", ".join(f"{k} {v['t']:+.2f}" for k, v in probe["halves"].items())
          + f"; orthogonal to trailing return {probe['t_orth_trailing']:+.2f}")
    print(f"  contrarian rule (picked AFTER seeing the sign): net Sharpe "
          f"{contra['sharpe_net']:.2f} vs B&H {contra['sharpe_bh']:.2f}, alpha t "
          f"{contra['alpha_t']:+.2f}")

    # ------------------------------------------------------------------ 9 synthetic
    print("\n=== 9. the control: a template that genuinely recurs ===")
    syn = []
    for s in SYN_STRENGTHS:
        res = st.null_sweep_power(
            lambda sd_, s=s: data.synthetic_tape(n_periods=len(m), signal_strength=s,
                                                 mu=mm["mu"], sigma=mm["sigma"], seed=sd_)[0],
            None, 12, OOS_MONTHLY, SYN_SEEDS, st.HEADLINE_MONTHLY)
        syn.append({"signal_strength": s, "reject_rate": float((res["t"] >= 2).mean()),
                    "median_t": float(res["t"].median()),
                    "median_corr": float(res["corr"].median()),
                    "median_match": float(res["match_corr_median"].median()),
                    "n_seeds": int(len(res))})
        print(f"  strength {s:.2f}: t>=2 in {syn[-1]['reject_rate']:.0%} of "
              f"{len(res)} tapes, median t {syn[-1]['median_t']:+.2f}, median forecast corr "
              f"{syn[-1]['median_corr']:+.3f}, median best match {syn[-1]['median_match']:.3f}")
    h["synthetic"] = syn

    h["_verdict"] = st.verdict(h)
    h["_myth"] = st.myth_check(h)
    h["runtime_s"] = round(time.time() - t_start, 1)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}   "
          f"Myth: {h['_myth']['stamp']}")
    print(f"  {h['_verdict']['one_sentence']}")
    print(f"  runtime {h['runtime_s']}s")
    return h


def _fmt_table(rows) -> str:
    out = []
    for r in rows:
        out.append(
            f"| {r['tape']} | {r['metric']} | {r['L']} | {r['h']} | {r['k']} | "
            f"{r['corr']:+.3f} | {r['t']:+.2f} | {r['p']:.3f} | {r['p_holm']:.2f} | "
            f"{r['hit']:.1%} | {r['oos_r2']:+.1%} | {r['sharpe_net']:.2f} | "
            f"{r['alpha_t']:+.2f} | {r['time_in_market']:.0%} |")
    return "\n".join(out)


def results_md(h: dict) -> str:
    v = h["_verdict"]
    my = h["_myth"]
    P = h["provenance"]
    pc = "\n".join(
        f"| {k} | {r['median_price']:+.2f} | {r['iqr_price'][0]:+.2f} .. {r['iqr_price'][1]:+.2f} | "
        f"**{r['p_price_gt_09']:.1%}** | {r['p_price_gt_07']:.1%} | {r['p_returns_gt_05']:.2%} |"
        for k, r in h["pair_null"].items())
    bm = "\n".join(
        f"| {k} | {r['p5']:.3f} | **{r['p50']:.3f}** | {r['p95']:.3f} | {r['share_gt_09']:.0%} |"
        for k, r in h["best_match_null"].items())
    lfl = "\n".join(
        f"| {k} | **{r['real_median']:.3f}** | {r['real_share_gt_09']:.0%} | "
        f"**{r['null_median']:.3f}** | {r['null_median_lo']:.3f} .. {r['null_median_hi']:.3f} | "
        f"{r['null_share_gt_09']:.0%} | {r['n_sims']} |"
        for k, r in h["like_for_like"].items())
    ovl = "\n".join(
        f"| ≥ {float(th):.2f} | {e['n_match']} of {e['n_dates']} ({e['share_match']:.0%}) | "
        f"{h['overlay_null'][th]:.0%} | {np.expm1(e['fwd_match']):+.1%} | "
        f"{np.expm1(e['fwd_all']):+.1%} | {e['diff_t']:+.2f} | {e['crash_rate_match']:.1%} | "
        f"{e['crash_rate_all']:.1%} |"
        for th, e in h["overlay"].items())
    top = "\n".join(f"| {r['date']} | {r['corr']:.3f} | {np.expm1(r['fwd']):+.1%} |"
                    for r in h["overlay_top"])
    hm, hd, rm, rd = h["m_head"], h["d_head"], h["m_rule"], h["d_rule"]
    head = (
        f"| Monthly TR, L=24 m, h=12 m, k=5, price path | {hm['n']} | {hm['corr']:+.3f} "
        f"[{hm['ci'][0]:+.2f}, {hm['ci'][1]:+.2f}] | {hm['slope']:+.2f} | **{hm['t']:+.2f}** | "
        f"{hm['p_holm']:.2f} | {hm['hit']:.1%} ({hm['hit_t']:+.2f}) | {hm['hit_raw']:.1%} vs "
        f"{hm['up_share']:.1%} | {hm['oos_r2']:+.1%} |\n"
        f"| Daily price, L=250 d, h=63 d, k=5, price path | {hd['n']} | {hd['corr']:+.3f} "
        f"[{hd['ci'][0]:+.2f}, {hd['ci'][1]:+.2f}] | {hd['slope']:+.2f} | **{hd['t']:+.2f}** | "
        f"{hd['p_holm']:.2f} | {hd['hit']:.1%} ({hd['hit_t']:+.2f}) | {hd['hit_raw']:.1%} vs "
        f"{hd['up_share']:.1%} | {hd['oos_r2']:+.1%} |")
    rules = (
        f"| Monthly TR (cash = T-bill) | {rm['start']} | {rm['sharpe_gross']:.2f} | "
        f"**{rm['sharpe_net']:.2f}** | **{rm['sharpe_bh']:.2f}** | {rm['ann_ret_net']:.2%} | "
        f"{rm['ann_ret_bh']:.2%} | {rm['alpha_ann']:+.2%} ({rm['alpha_t']:+.2f}) | "
        f"{rm['time_in_market']:.0%} | {rm['turnover_per_year']:.1f} | {rm['max_dd']:.0%} / "
        f"{rm['max_dd_bh']:.0%} |\n"
        f"| Daily price (cash = 0) | {rd['start']} | {rd['sharpe_gross']:.2f} | "
        f"**{rd['sharpe_net']:.2f}** | **{rd['sharpe_bh']:.2f}** | {rd['ann_ret_net']:.2%} | "
        f"{rd['ann_ret_bh']:.2%} | {rd['alpha_ann']:+.2%} ({rd['alpha_t']:+.2f}) | "
        f"{rd['time_in_market']:.0%} | {rd['turnover_per_year']:.1f} | {rd['max_dd']:.0%} / "
        f"{rd['max_dd_bh']:.0%} |")
    rs = "\n".join(
        f"| {r['tape']} | {int(r['n'])} | {int(r['beat'])} | {r['med_sharpe']:.2f} | "
        f"{r['max_sharpe']:.2f} | {r['bh']:.2f} |" for r in h["rules_summary"])
    syn = "\n".join(
        f"| {r['signal_strength']:.2f} | {r['n_seeds']} | **{r['reject_rate']:.0%}** | "
        f"{r['median_t']:+.2f} | {r['median_corr']:+.3f} | {r['median_match']:.3f} |"
        for r in h["synthetic"])
    tbl = sorted(h["table"], key=lambda r: (r["tape"] != "monthly", r["metric"], r["L"],
                                           r["h"], r["k"]))
    w, wp = h["worst_combo"], h["wrong_way"]
    lfl_m = h["like_for_like"]["monthly logprice L24"]
    ot = h["overlay_trailing"]
    b = h["best_combo"]
    mm, dm = h["m_mom"], h["d_mom"]
    return f"""# Results — Study 1015 (The 1929 Overlay) on the real tapes

*Generated by [`examples/verify.py`](../examples/verify.py) in {h['runtime_s']:.0f} s. As-of
**{h['as_of']}** (monthly tape to {h['as_of_monthly']}, daily tape to {h['as_of_daily']}; the
daily tape's partial December 2022 is dropped). Everything offline and deterministic.*

## Data provenance

| Tape | Loader | Shipped in | SHA-256 pin | Span | Rows | Fingerprint |
|---|---|---|---|---|--:|---|
| US market, monthly **total return**, nominal (Fama-French `Mkt-RF + RF`) | `{P['monthly']['loader']}` | {P['monthly']['package']} | `{P['monthly']['sha256'][:16]}…` | {h['m_first']} → {h['m_last']} | {h['m_n']} | `{h['fp_monthly']}` |
| S&P 500 daily **price index** (no dividends), nominal | `{P['daily']['loader']}` | {P['daily']['package']} | `{P['daily']['sha256'][:16]}…` | {h['d_first']} → {h['d_last']} | {h['d_n']} | `{h['fp_daily']}` |

Calibration of every random-walk null below: monthly log drift {mm['mu']:.4f} ({mm['mu_ann']:.2%}/yr),
vol {mm['sigma']:.4f} ({mm['sigma_ann']:.2%}/yr); daily log drift {dm['mu']:.5f}
({dm['mu_ann']:.2%}/yr), vol {dm['sigma']:.4f} ({dm['sigma_ann']:.2%}/yr). The monthly timing rule's
flat leg earns the T-bill; the daily tape ships no cash rate, so its flat leg earns zero.

## 1. How cheap is a 0.9? Two independent random walks

Twenty thousand pairs of **independent** Gaussian random walks with the tape's drift and vol.
They share no information whatsoever.

| Window | Median price-path corr | Interquartile range | P(price corr > 0.9) | P(> 0.7) | P(return corr > 0.5) |
|---|--:|--:|--:|--:|--:|
{pc}

Price paths of unrelated random walks correlate all over the map: over 24 months the middle half
of the pairs alone runs from {h['pair_null']['monthly, 24 months']['iqr_price'][0]:+.2f} to
{h['pair_null']['monthly, 24 months']['iqr_price'][1]:+.2f}, and one pair in
{1 / h['pair_null']['monthly, 24 months']['p_price_gt_07']:.0f} clears 0.7 — because a trending
series correlates with any other trending series. Correlate the *returns* of the very same pairs
and the effect disappears; that is the comparison that carries information, and it is the one
the charts never show. A single pair rarely reaches 0.9 — but a single pair is not what the
chart shows.

## 2. Searching makes it cheaper: the best match in the whole past

The overlay chart is not one pair; it is the best of a search. Here a random walk of the tape's
length is searched, exactly as the forecaster searches, for the past window that best matches its
own final window:

| Search | 5% | Median best match | 95% | P(best > 0.9) |
|---|--:|--:|--:|--:|
{bm}

And like for like — the same evaluation dates, the same search, the real tape against random
walks with its drift and vol:

| Search | Real tape: median best match | Real: > 0.9 | Random walk: median | RW per-path medians | RW: > 0.9 | RW paths |
|---|--:|--:|--:|--:|--:|--:|
{lfl}

Two richer nulls for the monthly price-path search, built to keep what a Gaussian walk lacks:
a walk with the real tape's own **volatility path** (calm and turbulent eras where they were,
directions coin-flipped) gives a median best match of **{lfl_m['volpath_median']:.3f}**; a
**6-month block bootstrap** of the real returns (fat tails and short memory kept, anything
recurring years apart destroyed) gives **{lfl_m['block_median']:.3f}**.

The real tape's matches are about as good as a random walk's — a shade tighter than a Gaussian
walk, and most of that gap is closed once volatility clustering and short memory are put back. A 0.9 printed on an overlay chart
is the expected output of the search, not evidence of anything. Matching z-scored returns
instead of price paths drops the "best match" to the 0.5-0.7 range on both — the honest scale of
similarity once the shared trend is removed.

## 3. The 1929 overlay, tested directly

Template: the 24 months of total-return path ending **{TEMPLATE_END}**, the run-up into the 1929
peak. Every later month is scored by how closely its own trailing 24 months match it, and what
followed over the next 12 months is recorded (overlapping windows; HAC with 12 lags).

| Match | Months matching | Random walk would | Next 12m (matching) | Next 12m (all) | HAC t of difference | Fell 20%+ (matching) | Fell 20%+ (all) |
|---|--:|--:|--:|--:|--:|--:|--:|
{ovl}

The best matches to the 1929 run-up since 1931:

| Month | Corr with 1927-29 | Next 12 months |
|---|--:|--:|
{top}

Matching months did go on to earn less than average — but not by crashing, and mostly because a
high correlation with a two-year rally is just a two-year rally: the match score correlates
**{ot['corr_match_trailing']:.2f}** with the trailing 24-month return, and once that return is
removed the match's own t falls to **{ot['t_match_orth']:+.2f}** (the trailing return alone:
t {ot['t_trailing']:+.2f}). Below-average returns after big run-ups is an old, contested
long-horizon reversal story; it is not 1929 repeating.

An early-2014 overlay of the kind that circulated widely: on {h['showcase']['date']} the trailing
24-month path correlated **{h['showcase']['corr']:.3f}** with the 1927-29 run-up. The next 12
months returned **{np.expm1(h['showcase']['fwd']):+.1%}**. A rising market looks like every other
rising market, including the one in 1929; that is all the correlation measures.

## 4. The forecaster — the pre-registered headline specifications

At each date, the `k` best non-overlapping past windows (ending at least `max(L, h)` periods
earlier, so every analogue's follow-on is observed and none overlaps today) forecast the next
`h`-period log return as the average of what followed them. Monthly forecasts every month from
{OOS_MONTHLY}; daily forecasts every {DAILY_STEP} trading days from {OOS_DAILY}. Slope t-statistics
are Newey-West with `ceil(h / step)` lags; correlation intervals are 95% stationary-bootstrap.
"Hit" is same side of the historical drift (its HAC t against 50% in brackets); the raw sign hit
rate is printed next to the share of up periods so the usual confusion is visible.

| Specification | n | Corr [95% CI] | Slope | **HAC t** | Holm p | Hit vs drift | Raw sign hit vs up-share | OOS R² vs drift |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{head}

## 5. The full sweep — every combination, every number

**{h['n_combos']} combinations** were tried: monthly L ∈ {{12, 24, 36, 60}} × h ∈ {{1, 6, 12}} ×
k ∈ {{1, 5, 10}} × {{price path, z-scored returns}} = {h['n_combos_m']}; daily L ∈ {{63, 126, 250}}
× h ∈ {{5, 21, 63}} × k ∈ {{1, 5, 10}} × 2 = {h['n_combos_d']}. Nothing else was run on the real
tapes.

- Raw one-sided p < 0.05 (the claim's direction): **{h['n_raw_sig']} of {h['n_combos']}
  ({h['share_raw_sig']:.1%})** — chance alone gives 5%.
- Best combination: {b['tape']} {b['metric']} L={b['L']} h={b['h']} k={b['k']}, t {b['t']:+.2f};
  Holm-adjusted p **{b['p_holm']:.2f}**; smallest Holm p in the grid **{h['holm_min_p']:.2f}**
  (Bonferroni {h['bonf_min_p']:.2f}).
- Most negative: {w['tape']} {w['metric']} L={w['L']} h={w['h']} k={w['k']}, t {w['t']:+.2f}
  (two-sided Holm p {w['p_holm_two']:.4f}) — see section 7.

| Tape | Metric | L | h | k | Corr | HAC t | p (1-sided) | Holm p | Hit vs drift | OOS R² | Rule Sharpe (net) | Timing α t | In market |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{_fmt_table(tbl)}

## 6. Could you trade it? The long/flat rule

Long the market when the analogues forecast a positive `h`-period return, flat otherwise. The
forecast made at the close of period `t` is **executed at the close of `t + 1`** (one full period
of lag) and first earns period `t + 2`. Costs {COST_BPS:.0f} bp one-way on traded NAV. All Sharpe
ratios are excess over cash for both legs; timing α is the intercept of the rule's net excess
return on buy-and-hold's (Newey-West).

| Headline rule | From | Sharpe gross | **Sharpe net** | **Buy & hold** | Ann. ret net | Ann. ret B&H | Timing α (t) | In market | Turnover/yr | Max DD rule / B&H |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{rules}

Across the whole grid:

| Tape | Rules | Beat B&H on net Sharpe | Median net Sharpe | Best net Sharpe | B&H Sharpe |
|---|--:|--:|--:|--:|--:|
{rs}

White's Reality Check over every rule on each tape (stationary bootstrap, 1,000 resamples), run
two ways. **Return race** (H0: no rule earns more excess return than buy-and-hold): monthly
**p = {h['rc_ret_p_monthly']:.3f}** (best {h['rc_best_monthly']}, information ratio
{h['rc_ir_monthly']:+.2f}); daily **p = {h['rc_ret_p_daily']:.3f}** (best {h['rc_best_daily']}, IR
{h['rc_ir_daily']:+.2f}). **Sharpe race** — each rule's excess return first scaled to buy-and-hold's
volatility, so a rule that de-risks gets credit for it: monthly **p = {h['rc_vm_p_monthly']:.3f}**
(best {h['rc_vm_best_monthly']}, IR {h['rc_vm_ir_monthly']:+.2f}); daily
**p = {h['rc_vm_p_daily']:.3f}** (best {h['rc_vm_best_daily']}, IR {h['rc_vm_ir_daily']:+.2f}). The
two monthly rules that edge buy-and-hold on net Sharpe are what a search of {h['n_combos_m']}
variants is expected to throw up. The verdict reads the smaller p of the two races per tape.

## 7. The wrong-way result, probed

The single most extreme number in the grid has the **opposite** sign to the claim:
{w['tape']} price-path matching, L={w['L']}, h={w['h']}, k={w['k']}, t {w['t']:+.2f}. It is not an
artefact of the lag choice (t at 1×/2×/4× the lags: {' / '.join(f'{x:+.2f}' for x in wp['lags'].values())}),
appears in both halves ({', '.join(f"{k} from {x['from']}: t {x['t']:+.2f}" for k, x in wp['halves'].items())}),
and is not the trailing return in disguise (t {wp['t_orth_trailing']:+.2f} after removing it).
Flipping the rule to go long when the analogues say "down" — a choice made *after* seeing the
sign, so every number here is post-hoc — earns a net Sharpe of {wp['contrarian_sharpe']:.2f}
against {wp['contrarian_bh']:.2f} for buy-and-hold (timing α t {wp['contrarian_alpha_t']:+.2f}).
{h['n_neg_sig_two']} combination(s) are significantly negative after a two-sided Holm correction,
{h['n_pos_sig_two']} significantly positive. It does not carry over to the monthly tape, it is
one corner of a grid that was searched, and the pre-registered verdict tests the claim's
direction; it is recorded here as a lead, not a finding.

## 8. The control — a template that genuinely recurs

The same headline monthly test (L=24, h=12, k=5, price path) on synthetic tapes of the same
length, drift and vol, in which a distinctive 24-period shape is followed by the same 6-period
fall every time it appears, scaled by `signal_strength` (0 = pure random walk). Twenty seeds each.

| signal_strength | Tapes | Share with t ≥ 2 | Median t | Median forecast corr | Median best match |
|---|--:|--:|--:|--:|--:|
{syn}

At zero the test rejects at about its nominal rate — while still printing best matches as high
as the real tape's. With a real recurring template it fires. The detector works; the real tapes
simply give it nothing to detect.

## Caveats

- **Price index vs total return.** The daily S&P tape excludes dividends; the monthly tape
  includes them. No cash rate ships with the daily tape, so the daily rule's flat leg earns zero:
  that *penalises* the rule (a real implementation would earn bill interest while flat), but
  with bill rates low for most of 2000-2022 and the rule flat a minority of the time, the
  omission is worth a fraction of a percent a year — too little to change any conclusion.
- **US only, one market.** Both tapes are US large-cap equity. The 1929 template is a US event.
- **One crash in the template.** Section 3 tests one template (1929). 1987 and 2000 sit inside the
  monthly tape and are matched by the general search in sections 4-6, which is the fairer test.
- **Overlap.** h-period outcomes sampled every period overlap; all slope and alpha inference is
  Newey-West, and bootstrap intervals use blocks of twice the horizon. The Reality Check uses
  the stationary bootstrap of `quantlab.bayes`.
- **The grid is a choice.** {h['n_combos']} combinations is generous but finite; DTW, scaled
  Euclidean distance or kernel-weighted analogues are not covered. They change the distance,
  not the null: any similarity score on trending paths inherits the problem in section 1.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

**Does the past match itself better than a random walk? {my['stamp']}.** {my['why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1015-analogue-overlay](../README.md). Not investment advice.*
"""


def main() -> None:
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    slim = {k: v for k, v in h.items() if k not in ("table", "like_for_like", "provenance")}
    print("##HEADLINE## " + json.dumps(slim, default=lambda o: float(o)
                                       if isinstance(o, (np.floating, np.integer)) else str(o)))


if __name__ == "__main__":
    main()
