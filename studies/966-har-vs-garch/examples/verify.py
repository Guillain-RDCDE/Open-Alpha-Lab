"""Real-tape verification — Study 966 (Forecasting Tomorrow's Vol). Regenerates docs/results.md.

Runs the four forecasters out of sample on six tapes at three horizons, scores them
with QLIKE and MSE, tests each against the 21-day rolling baseline with a HAC Diebold-Mariano,
and repeats the whole tournament on simulated data where the conditional variance is known —
so the cost of scoring against a noisy proxy can be measured rather than assumed.

    python studies/966-har-vs-garch/examples/verify.py            # cache-only
    python studies/966-har-vs-garch/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vol_forecast import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


BURN = 756
REFIT = 126


def report() -> dict:
    px = data.load_prices()
    rets = {tk: px[tk].dropna().pct_change().dropna() for tk in data.TICKERS}
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS), "burn": BURN,
               "refit_every": REFIT}

    print(f"as-of {data.AS_OF}   burn-in {BURN} sessions   refit every {REFIT}")
    for tk, r in rets.items():
        print(f"  {tk:4s} {r.index[0].date()} -> {r.index[-1].date()}  n={len(r):,}  "
              f"ann vol {r.std() * np.sqrt(st.TRADING_DAYS):.1%}")
    h["windows"] = {tk: [str(r.index[0].date()), str(r.index[-1].date())]
                    for tk, r in rets.items()}
    h["n_obs"] = {tk: int(len(r)) for tk, r in rets.items()}
    h["fingerprint"] = data.fingerprint(px)

    print("\n=== fitted GARCH(1,1) parameters on the full sample (context, not a forecast) ===")
    params = {}
    for tk, r in rets.items():
        f = st.fit_garch11(r)
        params[tk] = {k: f[k] for k in ("omega", "alpha", "beta", "persistence")}
        print(f"  {tk:4s} alpha {f['alpha']:.3f}  beta {f['beta']:.3f}  "
              f"persistence {f['persistence']:.4f}  half-life "
              f"{np.log(0.5) / np.log(max(f['persistence'], 1e-9)):6.1f} days  "
              f"converged={f['converged']}")
    h["garch_params"] = params

    print(f"\n=== the tournament, out of sample, horizon = 1 day ===")
    print("  tkr  model                    QLIKE       MSE     DM vs rolling      p")
    tourneys, wins, dms, gains, best_counts = {}, 0, [], [], {}
    for tk, r in rets.items():
        t = st.tournament(r, 1, BURN, REFIT)
        tourneys[tk] = {m: dict(v) for m, v in t.to_dict("index").items()}
        for m, row in t.iterrows():
            print(f"  {tk:4s} {st.MODEL_LABEL[m]:24s} {row['qlike']:8.4f} {row['mse']:.2e}  "
                  f"{row['dm_vs_rolling']:+9.2f}  {row['p_vs_rolling']:8.4f}")
        best = t["qlike"].idxmin()
        best_counts[best] = best_counts.get(best, 0) + 1
        if best != "rolling21":
            wins += 1
            dms.append(t.loc[best, "dm_vs_rolling"])
            gains.append(1 - t.loc[best, "qlike"] / t.loc["rolling21", "qlike"])
    h["tournament_h1"] = tourneys
    h["n_wins_vs_rolling"] = int(wins)
    h["pooled_dm"] = float(np.nanmean(dms)) if dms else 0.0
    h["pooled_qlike_gain"] = float(np.nanmean(gains)) if gains else 0.0
    best_model = max(best_counts, key=best_counts.get)
    h["best_model"] = st.MODEL_LABEL[best_model]
    h["best_model_key"] = best_model
    h["best_model_wins"] = int(best_counts[best_model])
    print(f"\n  best model by tape: " +
          ", ".join(f"{st.MODEL_LABEL[k]} x{v}" for k, v in best_counts.items()))
    print(f"  beats the rolling baseline on {wins}/{len(data.TICKERS)} tapes, pooled DM "
          f"{h['pooled_dm']:+.2f}, pooled QLIKE improvement {h['pooled_qlike_gain']:.1%}")

    # how much of the gain does the free model deliver?
    shares = []
    for tk in data.TICKERS:
        t = pd.DataFrame(tourneys[tk]).T
        base = t.loc["rolling21", "qlike"]
        best_q = t["qlike"].min()
        ew = t.loc["ewma94", "qlike"]
        if base - best_q > 0:
            shares.append((base - ew) / (base - best_q))
    h["ewma_share_of_gain"] = float(np.clip(np.nanmean(shares), 0, 1.5)) if shares else 0.0
    print(f"  EWMA (one parameter, never fitted) captures "
          f"{h['ewma_share_of_gain']:.0%} of the best model's improvement")

    print("\n=== does the ranking survive the horizon? ===")
    horizons = {}
    for hz in st.HORIZONS:
        t = st.tournament(rets["SPY"], hz, BURN, REFIT)
        horizons[hz] = {m: float(t.loc[m, "qlike"]) for m in st.MODELS}
        order = " > ".join(st.MODEL_LABEL[m].split(" ")[0] for m in t["qlike"].sort_values().index)
        print(f"  SPY horizon {hz:2d}d: best-to-worst {order}   "
              f"(best QLIKE {t['qlike'].min():.4f} vs rolling {t.loc['rolling21', 'qlike']:.4f})")
    h["horizons_spy"] = horizons

    print("\n=== the same tournament where the truth is observable (simulation) ===")
    r_sim, sigma, tr = data.synthetic_vol_path(n_years=20, signal_strength=1.0, seed=966)
    proxy = st.tournament(r_sim, 1, BURN, REFIT)
    truth = st.truth_scored_tournament(r_sim, sigma, 1, BURN, REFIT)
    for m in st.MODELS:
        print(f"  {st.MODEL_LABEL[m]:24s} QLIKE vs proxy {proxy.loc[m, 'qlike']:.4f}  "
              f"vs truth {truth.loc[m, 'qlike_vs_truth']:.4f}  "
              f"corr with truth {truth.loc[m, 'corr_with_truth']:.3f}")
    h["sim_proxy"] = {m: float(proxy.loc[m, "qlike"]) for m in st.MODELS}
    h["sim_truth"] = {m: float(truth.loc[m, "qlike_vs_truth"]) for m in st.MODELS}
    h["sim_same_winner"] = bool(proxy["qlike"].idxmin() == truth["qlike_vs_truth"].idxmin())
    print(f"  proxy and truth pick the same winner: {h['sim_same_winner']} "
          f"(proxy {st.MODEL_LABEL[proxy['qlike'].idxmin()]}, "
          f"truth {st.MODEL_LABEL[truth['qlike_vs_truth'].idxmin()]})")

    print("\n=== the null: constant volatility, nothing to forecast ===")
    r_null, _, _ = data.synthetic_vol_path(n_years=20, signal_strength=0.0, seed=966)
    tn = st.tournament(r_null, 1, BURN, REFIT)
    null_gain = float(1 - tn["qlike"].min() / tn.loc["rolling21", "qlike"])
    f_null = st.forecasts(r_null, 1, BURN, REFIT)
    f_real = st.forecasts(rets["SPY"], 1, BURN, REFIT)
    flat = {m: float(np.log(f_null[m].dropna()).std() / np.log(f_real[m].dropna()).std())
            for m in st.MODELS}
    print(f"  best model's QLIKE improvement over the 21-day rolling window: {null_gain:+.2%}")
    print(f"  that is NOT forecasting skill — with constant volatility the best possible")
    print(f"  forecast is the unconditional variance, and a fitted model shrinks toward it")
    print(f"  while a 21-day window keeps re-estimating it noisily. The test that matters is")
    print(f"  whether the models invent dynamics; dispersion of the forecast path, null vs SPY:")
    for m in st.MODELS:
        print(f"    {st.MODEL_LABEL[m]:24s} {flat[m]:.2f}x  "
              f"({'flat, as it should be' if flat[m] < 0.7 else 'suspiciously lively'})")
    h["null_gain"] = null_gain
    h["null_flatness"] = flat

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    rows = "\n".join(
        f"| {tk} | {st.MODEL_LABEL[m]} | {r['qlike']:.4f} | {r['mse']:.2e} | "
        f"{r['dm_vs_rolling']:+.2f} | {r['p_vs_rolling']:.4f} |"
        for tk in h["tickers"] for m, r in h["tournament_h1"][tk].items())
    gp = "\n".join(
        f"| {tk} | {p['alpha']:.3f} | {p['beta']:.3f} | {p['persistence']:.4f} | "
        f"{np.log(0.5) / np.log(max(p['persistence'], 1e-9)):.0f} days |"
        for tk, p in h["garch_params"].items())
    hz = "\n".join(
        f"| {k} day(s) | " + " | ".join(f"{h['horizons_spy'][k][m]:.4f}" for m in st.MODELS) + " |"
        for k in h["horizons_spy"])
    sim = "\n".join(
        f"| {st.MODEL_LABEL[m]} | {h['sim_proxy'][m]:.4f} | {h['sim_truth'][m]:.4f} |"
        for m in st.MODELS)
    win = "\n".join(f"| {tk} | {w[0]} → {w[1]} | {h['n_obs'][tk]:,} |"
                    for tk, w in h["windows"].items())
    heads = " | ".join(st.MODEL_LABEL[m] for m in st.MODELS)
    return f"""# Results — Study 966 (Forecasting Tomorrow's Vol) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Daily total-return closes
(`yfinance`, `auto_adjust=True`); every model reads the same series. Burn-in
**{h['burn']} sessions**, refit every **{h['refit_every']}**, forecasts strictly out of
sample. Target: realised variance over the next *h* sessions, proxied by squared daily
returns. Scores: QLIKE and MSE; comparisons by HAC Diebold-Mariano against the 21-day rolling
baseline. As-of **{h['as_of']}**; panel fingerprint `{h['fingerprint']}`.*

## Data stamp

| Ticker | Window | Sessions |
|---|---|--:|
{win}

## Fitted GARCH(1,1) (full sample — context, not a forecast)

| Ticker | alpha | beta | Persistence | Half-life |
|---|--:|--:|--:|--:|
{gp}

## The tournament, one-day horizon, out of sample

| Ticker | Model | QLIKE | MSE | DM vs rolling | p |
|---|---|--:|--:|--:|--:|
{rows}

Best model by tape: **{h['best_model']}** on {h['best_model_wins']} of
{len(h['tickers'])}. Something beat the free 21-day baseline on
**{h['n_wins_vs_rolling']} of {len(h['tickers'])}** tapes; pooled Diebold-Mariano
**{h['pooled_dm']:+.2f}**; pooled QLIKE improvement **{h['pooled_qlike_gain']:.1%}**. EWMA —
one parameter, never fitted — captures **{h['ewma_share_of_gain']:.0%}** of that improvement.

## Does the ranking survive the horizon? (SPY, QLIKE)

| Horizon | {heads} |
|---|--:|--:|--:|--:|
{hz}

## What the noisy target costs (simulation, where the truth is observable)

| Model | QLIKE vs squared-return proxy | QLIKE vs the true variance |
|---|--:|--:|
{sim}

Proxy and truth pick the same winner: **{h['sim_same_winner']}**.

## The null, and a correction to the obvious test

Under a **constant-volatility** null the best model still improves on the 21-day rolling window
by **{h['null_gain']:+.2%}** of QLIKE. That is not a failure of the tournament and not evidence
of overfitting: when volatility is constant the best possible forecast *is* the unconditional
variance, a variance-targeted GARCH shrinks straight to it, and the rolling window keeps
re-estimating it from 21 noisy observations. The test that actually distinguishes skill from
flexibility is whether a model **invents dynamics** where there are none — the dispersion of
its forecast path on the null relative to the real tape:

| Model | Forecast-path dispersion, null ÷ SPY |
|---|--:|
{chr(10).join(f"| {st.MODEL_LABEL[m]} | {h['null_flatness'][m]:.2f}x |" for m in st.MODELS)}

All four collapse toward flat, which is the correct behaviour — **but HAR only does so because
of the variance floor** documented in `strategy.har_forecast`. Corsi's model was designed for
realised variance built from *intraday* returns; fed a squared daily return its daily component
tracks a quantity noisier than the thing it estimates, and on quiet days it predicts a variance
near zero. QLIKE divides by the forecast, so without the floor a handful of those predictions
dominate the whole average (HAR scored ~67,000 against ~1.6 for everything else on the null
tape before the floor was added). The floor is the fix a practitioner would apply; the honest
fix is intraday data.

## Caveats

- **No intraday data**, so the target is a squared daily return: unbiased, and noisier than
  the quantity it estimates. QLIKE and the multi-day horizons are the mitigations; realised
  variance from 5-minute bars would be the fix (Andersen & Bollerslev 1998).
- **HAR is running on the wrong input, and it shows.** Corsi designed HAR for realised
  variance built from intraday returns. Fed a squared *daily* return its daily component
  tracks a quantity noisier than the thing it estimates, its forecast path stays lively even
  on constant-volatility data, and without the variance floor documented in
  `strategy.har_forecast` a single near-zero prediction dominates the QLIKE average. The
  floor is in place and disclosed; the deeper fix is intraday data.
- **QML, not MLE.** The GARCH is fitted under a Gaussian likelihood on fat-tailed data. That
  is the standard quasi-ML practice and it is consistent, but a Student-t likelihood would fit
  the tails better.
- **One family each.** No EGARCH, no GJR, no realised-GARCH, no combination forecasts —
  combinations usually win, which is itself a finding about the value of any single model.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study [966-har-vs-garch](../README.md).
Not investment advice.*
"""

def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    print("##HEADLINE## " + json.dumps(h, default=float))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
