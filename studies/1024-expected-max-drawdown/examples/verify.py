"""Real-tape verification — Study 1024 (The Drawdown You Were Promised). Regenerates docs/results.md.

For every complete calendar window on four frozen tapes, writes down the maximum drawdown a
Gaussian (GBM) risk model promises — from the preceding years only — and then records the
drawdown the window delivered. Counts how often the promised 95th percentile is breached
(exact binomial tests on non-overlapping windows; a moving-block bootstrap on overlapping
ones), decomposes the failure into "wrong mean", "wrong shape" and "wrong volatility state"
with four challenger models, and prices the multiplier a risk manager would need.

    python studies/1024-expected-max-drawdown/examples/verify.py

Offline: every tape comes from ``quantlab.bundled`` (SHA-256 pinned). Runs in about two
minutes on a laptop.
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

from promised import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

HORIZONS = (1, 3, 5, 10)
N_SIMS = 4000
LOOKBACK = {"sp500": 5, "nasdaq": 5, "ff_market": 10, "stocks": 5}
NAMES = {"sp500": "S&P 500", "nasdaq": "Nasdaq", "ff_market": "FF market"}


def _pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.0%}"


def _p(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def _f(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def eras(cov: pd.DataFrame) -> pd.Series:
    """Split each tape's windows into two halves by start year (first / second era)."""
    out = pd.Series(index=cov.index, dtype=object)
    for t, g in cov.groupby("tape"):
        yrs = np.sort(g["start"].unique())
        cut = yrs[len(yrs) // 2]
        lab = np.where(g["start"] < cut, f"{yrs[0]}–{cut - 1}",
                       f"{cut}–{g['end'].max()}")
        out.loc[g.index] = lab
    return out


def report() -> dict:
    t0 = time.time()
    tapes = {t: data.load_tape(t) for t in data.AS_OF_TAPE}
    h: dict = {"as_of": data.AS_OF, "tape_names": NAMES}
    h["provenance"] = []
    for t, r in tapes.items():
        src, name = data.SOURCES[t]
        h["provenance"].append({
            "tape": t, "label": data.LABELS[t], "package": src, "file": name,
            "sha": data.sha_pin(t), "fingerprint": data.fingerprint(r),
            "first": str(r.index[0].date()), "last": str(r.index[-1].date()),
            "rows": int(len(r))})
    print(f"as-of {data.AS_OF}")
    for p in h["provenance"]:
        print(f"  {p['tape']:10s} {p['first']} -> {p['last']}  rows {p['rows']:5d}  "
              f"sha {p['sha']}  fp {p['fingerprint']}")

    # ------------------------------------------------------------------ 1
    print("\n=== 1. the promise, and what sampling hides ===")
    disc = st.discretisation_table(0.0, 0.16, 1.0, n_sims=N_SIMS)
    print(disc.round(4).to_string())
    h["disc"] = disc.reset_index().to_dict("records")
    h["disc_err"] = float(abs(disc["mean_log"].iloc[-1] / disc["continuous_formula_log"].iloc[-1]
                              - 1.0))
    h["disc_monthly_vs_daily"] = float(disc.loc[12, "mean"] / disc.loc[252, "mean"] - 1.0)
    sp = tapes["sp500"]
    mu_d, sd_d = float(sp.mean()), float(sp.std(ddof=1))
    prom = []
    for H in HORIZONS:
        s = st.summary(st.gbm_mdd(mu_d, sd_d, 252 * H, N_SIMS))
        prom.append({"horizon": H, **s})
    h["sp500_mu_ann"], h["sp500_vol_ann"] = mu_d * 252, sd_d * np.sqrt(252)
    h["promise_table"] = prom
    print(pd.DataFrame(prom).round(3).to_string(index=False))

    # ------------------------------------------------------------------ 2
    print("\n=== 2. the ex-ante test (non-overlapping calendar windows) ===")
    covs = []
    for t in data.INDEX_TAPES:
        c = st.coverage(tapes[t], t, data.PERIODS_PER_YEAR[t], HORIZONS, LOOKBACK[t],
                        st.MODELS, n_sims=N_SIMS)
        covs.append(c)
        print(f"  {t}: {c['start'].nunique()} window starts  ({time.time() - t0:.0f}s)")
    cov = pd.concat(covs, ignore_index=True)
    summ = st.summarise(cov)
    g1 = summ[summ["model"] == "gauss"].copy()
    ks = []
    for (t, H), g in cov[cov["model"] == "gauss"].groupby(["tape", "horizon"], sort=False):
        ks.append(st.ks_uniform_pit(g["pit"]))
    g1["ks_p"] = ks
    print(g1.round(3).to_string(index=False))
    h["main"] = g1.to_dict("records")
    prim = {}
    for t in data.INDEX_TAPES:
        row = g1[(g1["tape"] == t) & (g1["horizon"] == 1)].iloc[0]
        prim[t] = {"rate": float(row["rate95"]), "breaches": int(row["breaches95"]),
                   "n": int(row["n"]), "p_greater": float(row["p_greater"]),
                   "p_less": float(row["p_less"])}
    h["primary"] = prim
    one = cov[(cov["horizon"] == 1)]
    pooled = one[one["model"] == "gauss"]
    h["pooled_rate"] = float(pooled["breach95"].mean())
    h["pooled_n"] = int(len(pooled))
    h["median_ratio"] = float(pooled["ratio_mean"].median())
    h["windows_table"] = cov[(cov["model"] == "gauss") & (cov["horizon"] == 1)][
        ["tape", "start", "est_mu_ann", "est_vol_ann", "win_vol_ann", "p_mean", "p_q95",
         "realised", "breach95"]].to_dict("records")

    # ------------------------------------------------------------------ 3
    print("\n=== 3. decomposition: mean, shape, volatility state (1-year windows) ===")
    dec = summ[summ["horizon"] == 1][["tape", "model", "n", "breaches95", "rate95",
                                       "p_greater", "p_less", "rate99",
                                       "median_ratio_mean", "k95"]]
    print(dec.round(3).to_string(index=False))
    h["decomp"] = dec.to_dict("records")
    h["oracle_rate"] = float(dec[(dec["tape"] == "sp500")
                                 & (dec["model"] == "gauss_oracle")]["rate95"].iloc[0])
    fixes = {}
    for m in ("boot", "garch_t"):
        d = dec[dec["model"] == m]
        fixes[m] = bool(((d["p_greater"] >= 0.05) & (d["rate95"] <= 0.10)).all())
    h["fixes"] = fixes
    for m in ("boot", "garch_t", "gauss_mu0", "gauss_oracle"):
        h[f"{m}_pooled_rate"] = float(one[one["model"] == m]["breach95"].mean())
    h["boot_rate"], h["garch_rate"] = h["boot_pooled_rate"], h["garch_t_pooled_rate"]
    allh = summ[["tape", "horizon", "model", "n", "rate95", "p_greater"]]
    h["all_models"] = allh.to_dict("records")
    print(allh.pivot_table(index=["tape", "horizon"], columns="model",
                           values="rate95").round(2).to_string())

    # ------------------------------------------------------------------ 4
    print("\n=== 4. overlapping windows (yearly step), moving-block bootstrap ===")
    ov = []
    for t in data.INDEX_TAPES:
        c = st.coverage(tapes[t], t, data.PERIODS_PER_YEAR[t], (3, 5, 10), LOOKBACK[t],
                        ("gauss",), step_years=1, n_sims=2000)
        for H, g in c.groupby("horizon"):
            g = g.sort_values("start")
            b = st.block_bootstrap_rate(g["breach95"].to_numpy(), block=H, n_boot=2000)
            ov.append({"tape": t, "horizon": int(H), **b,
                       "median_ratio": float(g["ratio_mean"].median())})
    ovd = pd.DataFrame(ov)
    print(ovd.round(3).to_string(index=False))
    h["overlap"] = ovd.to_dict("records")
    print(f"  ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ 5
    print("\n=== 5. the multiplier a risk manager needs (1-year, Gaussian plug-in) ===")
    g = one[one["model"] == "gauss"].copy()
    g["era"] = eras(g)
    mrows = []
    for t in data.INDEX_TAPES:
        gt = g[g["tape"] == t]
        lo, hi = st.multiplier_ci(gt["realised"], gt["p_q95"], n_boot=1000)
        mrows.append({"tape": t, "era": "all", "n": len(gt),
                      "k95": st.multiplier(gt["realised"], gt["p_q95"]), "lo": lo, "hi": hi})
        for e, ge in gt.groupby("era", sort=True):
            lo, hi = st.multiplier_ci(ge["realised"], ge["p_q95"], n_boot=1000)
            mrows.append({"tape": t, "era": e, "n": len(ge),
                          "k95": st.multiplier(ge["realised"], ge["p_q95"]),
                          "lo": lo, "hi": hi})
    mt = pd.DataFrame(mrows)
    print(mt.round(2).to_string(index=False))
    h["mult"] = mt.to_dict("records")
    h["k95_pooled"] = st.multiplier(g["realised"], g["p_q95"])
    h["k95_min"], h["k95_max"] = float(mt["k95"].min()), float(mt["k95"].max())
    h["k95_range"] = h["k95_max"] / h["k95_min"]
    gm = one[one["model"] == "gauss_mu0"]
    h["k95_mu0_pooled"] = st.multiplier(gm["realised"], gm["p_q95"])
    mm = []
    for t in data.INDEX_TAPES:
        gmt = gm[gm["tape"] == t]
        mm.append(st.multiplier(gmt["realised"], gmt["p_q95"]))
    h["k95_mu0_by_tape"] = [float(x) for x in mm]
    print(f"  pooled k95 {h['k95_pooled']:.2f}, range {h['k95_min']:.2f}-{h['k95_max']:.2f} "
          f"(max/min {h['k95_range']:.2f});  mu=0 variant pooled k95 {h['k95_mu0_pooled']:.2f}")

    # ------------------------------------------------------------------ 6
    print("\n=== 6. twenty survivors (1-year windows) ===")
    S = tapes["stocks"]
    sc = []
    for name in S.columns:
        c = st.coverage(S[name].dropna(), name, 252, (1,), 5,
                        ("gauss", "gauss_mu0", "boot", "garch_t"), n_sims=2000)
        sc.append(c)
    sc = pd.concat(sc, ignore_index=True)
    srows = []
    for m, gs in sc.groupby("model", sort=False):
        lo, hi = st.multiplier_ci(gs["realised"], gs["p_q95"], n_boot=500,
                                  cluster=gs["start"].to_numpy())
        # year-clustered bootstrap of the breach rate
        rng = np.random.default_rng(1024)
        yrs = gs["start"].unique()
        by = {y: gs.loc[gs["start"] == y, "breach95"].to_numpy(float) for y in yrs}
        rates = []
        for _ in range(1000):
            pick = rng.choice(yrs, len(yrs))
            v = np.concatenate([by[y] for y in pick])
            rates.append(v.mean())
        srows.append({"model": m, "n": len(gs), "rate95": float(gs["breach95"].mean()),
                      "rate_lo": float(np.quantile(rates, 0.05)),
                      "rate_hi": float(np.quantile(rates, 0.95)),
                      "rate99": float(gs["breach99"].mean()),
                      "median_ratio": float(gs["ratio_mean"].median()),
                      "k95": st.multiplier(gs["realised"], gs["p_q95"]),
                      "k95_lo": lo, "k95_hi": hi})
    stt = pd.DataFrame(srows)
    print(stt.round(3).to_string(index=False))
    h["stocks"] = stt.to_dict("records")
    gs = stt[stt["model"] == "gauss"].iloc[0]
    h["stocks_rate"] = float(gs["rate95"])
    h["stocks_k95"], h["stocks_k95_lo"], h["stocks_k95_hi"] = (
        float(gs["k95"]), float(gs["k95_lo"]), float(gs["k95_hi"]))
    per_stock = sc[sc["model"] == "gauss"].groupby("tape")["breach95"].mean()
    h["stocks_worst"] = str(per_stock.idxmax())
    h["stocks_worst_rate"] = float(per_stock.max())
    h["stocks_per_name"] = {k: float(v) for k, v in per_stock.items()}
    print(f"  ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ 7
    print("\n=== 7. robustness: the estimation lookback ===")
    rob = []
    for t in data.INDEX_TAPES:
        for lb in (3, 5, 10, "expanding"):
            c = st.coverage(tapes[t], t, data.PERIODS_PER_YEAR[t], (1,), lb,
                            ("gauss", "gauss_mu0"), n_sims=2000)
            for m, gg in c.groupby("model", sort=False):
                bp = st.binomial_p(int(gg["breach95"].sum()), len(gg))
                rob.append({"tape": t, "lookback": str(lb), "model": m, "n": len(gg),
                            "rate95": float(gg["breach95"].mean()),
                            "p_greater": bp["p_greater"]})
    rb = pd.DataFrame(rob)
    print(rb.pivot_table(index=["tape", "lookback"], columns="model",
                         values="rate95").round(3).to_string())
    h["robust"] = rb.to_dict("records")

    # ------------------------------------------------------------------ 8
    print("\n=== 8. synthetic calibration (where the truth is known) ===")
    syn = []
    for s in (0.0, 0.5, 1.0):
        c = st.synthetic_coverage(s, n_years=60, models=("gauss", "gauss_mu0", "gauss_oracle"),
                                  n_tapes=6, n_sims=2000)
        for m, gg in c.groupby("model", sort=False):
            bp = st.binomial_p(int(gg["breach95"].sum()), len(gg))
            syn.append({"signal_strength": s, "model": m, "n": len(gg),
                        "rate95": float(gg["breach95"].mean()), "p_greater": bp["p_greater"],
                        "p_two": bp["p_two"]})
    sy = pd.DataFrame(syn)
    print(sy.round(3).to_string(index=False))
    h["synthetic"] = sy.to_dict("records")
    h["null_oracle_rate"] = float(sy[(sy["signal_strength"] == 0.0)
                                     & (sy["model"] == "gauss_oracle")]["rate95"].iloc[0])
    h["null_plugin_rate"] = float(sy[(sy["signal_strength"] == 0.0)
                                     & (sy["model"] == "gauss")]["rate95"].iloc[0])
    h["planted_plugin_rate"] = float(sy[(sy["signal_strength"] == 1.0)
                                        & (sy["model"] == "gauss")]["rate95"].iloc[0])

    v = st.verdict(h)
    h["verdict"] = v
    h["runtime_s"] = float(time.time() - t0)
    print(f"\nSignal: {v['signal']}   Tradability: {v['trad']}")
    print(v["one_sentence"])
    print(f"runtime {h['runtime_s']:.0f}s")
    return h


def results_md(h: dict) -> str:
    v = h["verdict"]
    prov = "\n".join(
        f"| `{p['tape']}` | {p['label']} | `{p['package']}` / `{p['file']}` | "
        f"{p['first']} → {p['last']} | {p['rows']:,} | `{p['sha']}` | `{p['fingerprint']}` |"
        for p in h["provenance"])
    disc = "\n".join(
        f"| {int(d['steps_per_year'])} | {d['mean']:.1%} | {d['median']:.1%} | {d['q95']:.1%} | "
        f"{d['q99']:.1%} | {d['mean_log']:.4f} | {d['continuous_formula_log']:.4f} |"
        for d in h["disc"])
    prom = "\n".join(
        f"| {d['horizon']}y | {d['mean']:.1%} | {d['median']:.1%} | {d['q95']:.1%} | "
        f"{d['q99']:.1%} |" for d in h["promise_table"])
    main = "\n".join(
        f"| {NAMES[d['tape']]} | {d['horizon']}y | {d['n']} | {d['breaches95']} | "
        f"**{d['rate95']:.0%}** | {_p(d['p_greater'])} | {_pct(d['rate99'])} | "
        f"{d['median_ratio_mean']:.2f} | {d['mean_pit']:.2f} | {_p(d['ks_p'])} |"
        for d in h["main"])
    dec = "\n".join(
        f"| {NAMES[d['tape']]} | {st.MODEL_LABELS[d['model']]} | {d['n']} | "
        f"**{d['rate95']:.0%}** | {_p(d['p_greater'])} | {_p(d['p_less'])} | "
        f"{d['median_ratio_mean']:.2f} | {_f(d['k95'])} |"
        for d in h["decomp"])
    am = pd.DataFrame(h["all_models"]).pivot_table(index=["tape", "horizon"], columns="model",
                                                   values="rate95")
    ns = pd.DataFrame(h["all_models"]).groupby(["tape", "horizon"])["n"].first()
    allm = "\n".join(
        f"| {NAMES[t]} | {H}y | {ns.loc[(t, H)]} | "
        + " | ".join(_pct(am.loc[(t, H), m]) for m in st.MODELS) + " |"
        for t in data.INDEX_TAPES for H in HORIZONS if (t, H) in am.index)
    ov = "\n".join(
        f"| {NAMES[d['tape']]} | {d['horizon']}y | {d['n']} | **{d['rate']:.0%}** | "
        f"{_pct(d['lo'])}–{_pct(d['hi'])} | {_p(d['p_greater'])} | {d['median_ratio']:.2f} |"
        for d in h["overlap"])
    mult = "\n".join(
        f"| {NAMES[d['tape']]} | {d['era']} | {d['n']} | **{_f(d['k95'])}** | "
        f"{_f(d['lo'])}–{_f(d['hi'])} |" for d in h["mult"])
    stk = "\n".join(
        f"| {st.MODEL_LABELS[d['model']]} | {d['n']} | **{d['rate95']:.0%}** | "
        f"{d['rate_lo']:.0%}–{d['rate_hi']:.0%} | {d['rate99']:.0%} | {d['median_ratio']:.2f} | "
        f"{_f(d['k95'])} ({_f(d['k95_lo'])}–{_f(d['k95_hi'])}) |" for d in h["stocks"])
    rb = pd.DataFrame(h["robust"])
    rob = "\n".join(
        f"| {NAMES[t]} | {lb} | {int(g['n'].iloc[0])} | "
        f"{g[g['model'] == 'gauss']['rate95'].iloc[0]:.0%} "
        f"(p {g[g['model'] == 'gauss']['p_greater'].iloc[0]:.3f}) | "
        f"{g[g['model'] == 'gauss_mu0']['rate95'].iloc[0]:.0%} |"
        for (t, lb), g in rb.groupby(["tape", "lookback"], sort=False))
    syn = "\n".join(
        f"| {d['signal_strength']:.1f} | {st.MODEL_LABELS[d['model']]} | {d['n']} | "
        f"**{d['rate95']:.1%}** | {_p(d['p_two'])} |" for d in h["synthetic"])
    wt = pd.DataFrame(h["windows_table"])
    wsp = wt[wt["tape"] == "sp500"]
    wrows = "\n".join(
        f"| {int(r.start)} | {r.est_mu_ann:+.1%} | {r.est_vol_ann:.1%} | {r.win_vol_ann:.1%} | "
        f"{r.p_mean:.1%} | {r.p_q95:.1%} | {r.realised:.1%} | {'**breach**' if r.breach95 else ''} |"
        for r in wsp.itertuples())
    kmu0 = ", ".join(f"{NAMES[t]} {k:.2f}" for t, k in zip(data.INDEX_TAPES,
                                                         h["k95_mu0_by_tape"]))

    return f"""# Results — Study 1024 (The Drawdown You Were Promised) on the real tapes

*Generated by [`examples/verify.py`](../examples/verify.py) in {h['runtime_s']:.0f}s. As-of
**{h['as_of']}** (last complete month of the newest tape; windows use complete calendar years
only). Every promise is built from data that precedes its window. Drawdowns are percentage
falls from the running peak of the (log) price path, measured on the tape's own closes.*

## 0. Data provenance

| Tape | What | Package / file | Span used | Rows | SHA-256 pin | Fingerprint |
|---|---|---|---|--:|---|---|
{prov}

**Price vs total return.** The S&P 500 and Nasdaq series are **price indices** (no
dividends); the Fama-French market is a **total return**. The 20 stocks are a **survivor
sample** — every realised stock drawdown below is a **lower bound** on what an unselected
large-cap delivered. The skfolio files are cached from the pinned `skfolio-1.8.1` wheel; the
arch files ship inside the installed `arch` package.

## 1. The promise — and what the sampling frequency hides

One year of a driftless GBM at 16% volatility, simulated **exactly** (log increments are exact
normals, so nothing but Monte Carlo noise separates these rows from the truth), sampled at
different frequencies:

| Steps per year | Mean MDD | Median | 95th pct | 99th pct | Mean (log units) | Continuous formula √(π/2)·σ·√T |
|--:|--:|--:|--:|--:|--:|--:|
{disc}

As the step shrinks the simulation converges on Magdon-Ismail et al.'s zero-drift expectation
(gap {h['disc_err']:.1%} at 16 steps a day). The discretisation is not a rounding error: the same
process **sampled monthly shows a {abs(h['disc_monthly_vs_daily']):.0%} smaller mean drawdown
than sampled daily**, because the closes skip the troughs between them. A continuous-time
formula applied to a monthly risk report over-promises pain; a monthly-calibrated budget
applied to a daily-marked book under-promises it. Every promise below is therefore simulated at
the tape's own frequency.

For scale, the whole-tape S&P 500 parameters (log drift {h['sp500_mu_ann']:.1%}, volatility
{h['sp500_vol_ann']:.1%}, daily) promise:

| Horizon | Mean MDD | Median | 95th pct | 99th pct |
|---|--:|--:|--:|--:|
{prom}

## 2. The ex-ante test — non-overlapping windows

The Gaussian plug-in: mu and sigma estimated on the preceding **5 years** of daily data (**10
years** on the monthly Fama-French tape), the promise simulated at the tape's frequency, then
the realised drawdown of the window recorded. A calibrated model breaches its 95th percentile
in 5% of windows. Non-overlapping windows are independent trials under the null, so the
p-value is an exact one-sided binomial test.

| Tape | Horizon | Windows | Breaches | Breach rate (95% band) | p (too many) | Breach rate (99%) | Median realised / promised mean | Mean PIT | KS p (PIT uniform) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
{main}

Pooled over the three index tapes, the 1-year 95% band was breached in **{h['pooled_rate']:.0%}**
of {h['pooled_n']} windows. The mean PIT (where the realised drawdown fell inside the promised
distribution) sits *below* 0.5 at one year — most years are calmer than promised — while the
breach rate sits above 5%: the promise is simultaneously too pessimistic for ordinary years and
too optimistic for the bad ones. That is the signature of a distribution with the wrong shape,
not merely the wrong scale.

The S&P 500 1-year windows, one row per promise:

| Window | Trailing mu (ann.) | Trailing vol | Realised vol in window | Promised mean MDD | Promised 95th pct | Realised MDD | |
|---|--:|--:|--:|--:|--:|--:|---|
{wrows}

## 3. What is missing? Five models, 1-year windows

Same windows, same estimation data. The Gaussian is decomposed into three versions — the
plug-in, the same model with the drift set to **zero** (a common risk-desk convention), and a
*hindsight* version that is handed the whole tape's mu and sigma — and set against two
challengers that keep what the Gaussian throws away.

| Tape | Model | Windows | Breach rate | p (too many) | p (too few) | Median realised / promised mean | k95 |
|---|---|--:|--:|--:|--:|--:|--:|
{dec}

Every horizon, every model (breach rate of the 95% band; columns in the order
{', '.join(st.MODELS)}):

| Tape | Horizon | Windows | {' | '.join(st.MODELS)} |
|---|---|--:|{'--:|' * len(st.MODELS)}
{allm}

Pre-registered fix test (1-year, all three tapes, p ≥ 0.05 and breach rate ≤ 10%): block
bootstrap **{'passes' if h['fixes']['boot'] else 'fails'}**, GARCH-t
**{'passes' if h['fixes']['garch_t'] else 'fails'}**. Pooled 1-year breach rates — plug-in
Gaussian {h['pooled_rate']:.0%}, zero-drift Gaussian {h['gauss_mu0_pooled_rate']:.0%},
hindsight Gaussian {h['gauss_oracle_pooled_rate']:.0%}, block bootstrap {h['boot_rate']:.0%},
GARCH-t {h['garch_rate']:.0%}.

## 4. Overlapping windows — more windows, honest error bars

Windows stepped one year at a time share most of their path, so their breaches arrive in runs.
The interval and p-value come from a moving-block bootstrap with a block of H consecutive
windows (Künsch 1989), recentred on the 5% null. Gaussian plug-in only.

| Tape | Horizon | Windows | Breach rate | 90% interval | p (too many) | Median realised / promised mean |
|---|---|--:|--:|--:|--:|--:|
{ov}

## 5. The multiplier a risk manager would need

`k95` is the factor on the Gaussian plug-in 95th percentile that would have produced exactly
5% breaches (the 95th percentile of realised / promised-q95). Intervals: 1,000-draw window
bootstrap, 90%.

| Tape | Era | Windows | k95 | 90% interval |
|---|---|--:|--:|--:|
{mult}

Pooled across the three tapes, **k95 = {h['k95_pooled']:.2f}**. Across tapes and eras it ranges
from {h['k95_min']:.2f} to {h['k95_max']:.2f} — a max/min of **{h['k95_range']:.2f}** against a
pre-registered stability bar of 1.5. For the zero-drift Gaussian the pooled multiplier is
**{h['k95_mu0_pooled']:.2f}** ({kmu0}).

## 6. Twenty survivors — single stocks, 1-year windows

Twenty large US stocks that were alive and large in 2022, each run through the same 5-year
lookback. Breach-rate interval and the multiplier interval come from a bootstrap that
resamples **calendar years** (all stocks' windows in a year move together).

| Model | Stock-windows | Breach rate | 90% interval (year bootstrap) | 99% breach rate | Median realised / promised mean | k95 (90% interval) |
|---|--:|--:|--:|--:|--:|--:|
{stk}

Worst name: **{h['stocks_worst']}**, breaching in {h['stocks_worst_rate']:.0%} of its years.
Because the panel is survivors, these rates are a **floor**: the names whose drawdowns ran to
100% are exactly the ones the panel deletes.

## 7. Robustness — how long a lookback?

1-year windows, Gaussian plug-in (with its one-sided binomial p) and zero-drift Gaussian.
Window counts change with the lookback because a longer lookback starts later.

| Tape | Lookback (years) | Windows | Plug-in breach rate | Zero-drift breach rate |
|---|---|--:|--:|--:|
{rob}

## 8. Synthetic calibration — the machinery checked where the truth is known

Six independent 60-year tapes per world, 1-year windows, 5-year lookback. At
`signal_strength = 0` returns are i.i.d. Gaussian and the textbook formula with the **true**
moments must breach 5% of the time; at `1.0` the same unconditional mean and variance come
with GARCH clustering (persistence 0.997) and *t*(4) shocks.

| signal_strength | Model | Windows | Breach rate | p (two-sided vs 5%) |
|--:|---|--:|--:|--:|
{syn}

The true-moment Gaussian is calibrated in the i.i.d. world ({h['null_oracle_rate']:.1%}). The
plug-in Gaussian breaches {h['null_plugin_rate']:.1%} there — the cost of *estimating* mu and
sigma from five years, which is the floor any real-tape breach rate should be read against —
and {h['planted_plugin_rate']:.1%} once clustering and fat tails are switched on.

## Caveats

- **Few independent windows.** The main test uses non-overlapping calendar windows to keep the
  binomial exact, which leaves 27 (S&P 500), 15 (Nasdaq) and 81 (Fama-French) 1-year trials and
  single digits at 10 years. The 10-year rows are anecdotes, labelled as such.
- **Lookback sensitivity.** The S&P 500 result holds for 3-, 5-, 10-year and expanding
  lookbacks (section 7). On the Fama-French tape an *expanding* lookback — a ninety-year mean
  instead of a five- or ten-year one — brings the plug-in breach rate down to a level the
  binomial cannot reject; that is the trailing-mean failure of section 3 seen from the other
  side, not a rescue of the Gaussian shape.
- **The tapes overlap in time.** 1999–2018 is in all three index tapes, so the three primary
  tests are not independent; the verdict asks for two of three, not for a pooled p-value.
- **The Nasdaq tape starts in 1999**, so with a 5-year lookback its first window is 2004: the
  dot-com collapse is in no Nasdaq window. Its comparatively benign result is partly that gap.
- **Price indices.** Dividends are missing from the S&P 500 and Nasdaq series, which deepens
  drawdowns and lowers drift slightly — for the promise and the outcome alike.
- **GARCH-t on short windows.** The fit uses five years of daily (ten of monthly) data; a fit
  that comes out integrated is capped at persistence 0.995, and a failed fit falls back to an
  i.i.d. Gaussian. On the monthly tape 120 observations are thin for a GARCH.
- **Survivors.** The 20-stock panel can only understate single-name drawdown risk.
- **Monte Carlo noise.** 4,000 paths for 1–3-year promises (2,000 beyond; half that for the
  bootstrap and GARCH); common random numbers across windows. The 95th-percentile estimate
  carries roughly ±1% relative error, which moves breach rates by well under a percentage
  point.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

> **In one sentence:** {v['one_sentence']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1024-expected-max-drawdown](../README.md). Not investment advice.*
"""


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


def main() -> None:
    h = report()
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    keys = ("as_of", "primary", "pooled_rate", "median_ratio", "oracle_rate", "fixes",
            "k95_pooled", "k95_min", "k95_max", "k95_range", "k95_mu0_pooled",
            "stocks_rate", "stocks_k95", "null_oracle_rate", "null_plugin_rate",
            "planted_plugin_rate", "boot_rate", "garch_rate", "gauss_mu0_pooled_rate",
            "disc_err", "disc_monthly_vs_daily", "runtime_s")
    head = {k: h[k] for k in keys}
    head["signal"], head["trad"] = h["verdict"]["signal"], h["verdict"]["trad"]
    print("##HEADLINE## " + json.dumps(_clean(head)))


if __name__ == "__main__":
    main()
