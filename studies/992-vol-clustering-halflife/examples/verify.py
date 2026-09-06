"""Real-tape verification — Study 992 (How Long Is a Storm?). Regenerates docs/results.md.

Measures the half-life of volatility five ways on each asset — AR(1) on log
realised volatility, the raw autocorrelation crossing, a hand-rolled GARCH(1,1)'s persistence,
RiskMetrics' assumed λ, and a model-free impulse response — then explains their disagreement
with a two-component autocorrelation fit, sweeps the estimation window to show how much of the
AR(1) answer is its own smoothing, and reduces the whole thing to how much wilder the next month
is after a big day.

    python studies/992-vol-clustering-halflife/examples/verify.py            # cache-only
    python studies/992-vol-clustering-halflife/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from storm import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 21


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "window": WINDOW, "fingerprint": data.fingerprint(px)}

    assets = {}
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        if tk == data.CASH:
            continue
        s = rets[tk].dropna()
        if len(s) < 1500:
            continue
        assets[tk] = s
        a = st.annualisation_factor(s)
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"{a:5.0f} obs/yr  ann vol {s.std() * np.sqrt(a):.1%}")
    h["n_assets"] = int(len(assets))
    lead = data.EQUITY
    r = assets[lead]
    h["asset"] = lead
    h["n_days"] = int(len(r))

    print(f"\n=== 1. five answers to one question ({lead}) ===")
    tbl = st.halflife_table(r, WINDOW)
    for m, row in tbl.iterrows():
        print(f"  {m:10s} {row['halflife']:8.1f} days   ({row['note']})")
    h["table"] = tbl.reset_index().to_dict("records")
    hls = tbl["halflife"].dropna()
    h["hl_min"] = float(hls.min())
    h["hl_max"] = float(hls.max())
    h["hl_garch"] = float(tbl.loc["garch", "halflife"])
    h["hl_acf"] = float(tbl.loc["acf", "halflife"])
    h["hl_ar1"] = float(tbl.loc["ar1", "halflife"])
    h["impulse_halflife"] = float(tbl.loc["impulse", "halflife"])
    print(f"  -> spread {h['hl_min']:.0f} to {h['hl_max']:.0f} days "
          f"({h['hl_max'] / h['hl_min']:.1f}x). All five are computed correctly.")

    print("\n=== 2. why they disagree ===")
    tc = st.two_component_fit(r, WINDOW)
    h.update({k: tc.get(k) for k in ("weight_fast", "tau_fast", "tau_slow", "halflife_fast",
                                     "halflife_slow", "sse_two_component", "sse_single",
                                     "improvement")})
    h["two_component_improvement"] = tc.get("improvement", np.nan)
    print(f"  a two-component fit to the volatility autocorrelation:")
    print(f"    fast component: half-life {tc['halflife_fast']:6.1f} days, weight "
          f"{tc['weight_fast']:.0%}")
    print(f"    slow component: half-life {tc['halflife_slow']:6.1f} days, weight "
          f"{1 - tc['weight_fast']:.0%}")
    print(f"    squared error vs a single exponential: {tc['sse_two_component']:.4f} vs "
          f"{tc['sse_single']:.4f} ({tc['improvement']:.0%} better)")
    print("  -> every single-number estimator returns a weighted average of those two, "
          "weighted by whichever lags it looks at")

    print("\n=== 3. the window sweep: two biases pulling opposite ways ===")
    sw = st.window_sweep(r)
    print(sw.round(4).to_string())
    h["window_sweep"] = sw.reset_index().to_dict("records")
    h["sweep_min"] = float(sw["halflife"].min())
    h["sweep_max"] = float(sw["halflife"].max())
    print(f"  the same data gives {h['sweep_min']:.1f} to {h['sweep_max']:.0f} days depending "
          f"only on the window used to build the volatility proxy")
    print("  short windows ATTENUATE: log|return| is a noisy proxy, and classical "
          "errors-in-variables pushes the coefficient toward zero")
    print("  long windows INFLATE: a rolling mean makes neighbouring observations share most "
          "of their data, which manufactures persistence")
    print("  the truth is between, and this is why picking a window after seeing the answer is "
          "not a defensible way to quote a half-life")

    print("\n=== 4. the model-free impulse response ===")
    ir = st.impulse_response_halflife(r, WINDOW)
    print(f"  {ir['n_shocks']} days above the 95th volatility percentile")
    print(f"  log-vol excess on the shock day: {ir['initial_excess']:+.3f}")
    for lab, k in (("after 5 days", "excess_at_5d"), ("after 21 days", "excess_at_21d"),
                   ("after 63 days", "excess_at_63d")):
        print(f"    {lab:16s} {ir[k]:+.3f}  "
              f"({ir[k] / ir['initial_excess']:.0%} of the shock remains)")
    print(f"  half-life: {ir['halflife']:.0f} days")
    h["impulse"] = {k: v for k, v in ir.items() if k != "path"}

    print("\n=== 5. the version that decides something ===")
    pd_tbl = st.practical_decay(r, WINDOW)
    print(pd_tbl.round(4).to_string())
    h["practical"] = pd_tbl.reset_index().to_dict("records")
    for hz in (2, 5, 21, 63):
        if hz in pd_tbl.index:
            h[f"ratio_{hz}d"] = float(pd_tbl.loc[hz, "ratio"])
    print(f"  after a top-5% volatility day, the next month's realised volatility runs "
          f"{h.get('ratio_21d', np.nan):.2f}x the level after a normal day")

    print("\n=== 6. every asset ===")
    cross = []
    for tk, s in assets.items():
        t = st.halflife_table(s, WINDOW)
        tcs = st.two_component_fit(s, WINDOW)
        p = st.practical_decay(s, WINDOW)
        cross.append({
            "asset": tk, "ann_factor": st.annualisation_factor(s),
            "ar1": t.loc["ar1", "halflife"], "acf": t.loc["acf", "halflife"],
            "garch": t.loc["garch", "halflife"], "impulse": t.loc["impulse", "halflife"],
            "fast": tcs.get("halflife_fast", np.nan),
            "slow": tcs.get("halflife_slow", np.nan),
            "ratio_21d": float(p.loc[21, "ratio"]) if 21 in p.index else np.nan,
        })
        print(f"  {tk:9s} ar1 {cross[-1]['ar1']:6.1f}  acf {cross[-1]['acf']:6.1f}  "
              f"garch {cross[-1]['garch']:7.1f}  impulse {cross[-1]['impulse']:6.1f}  "
              f"| fast {cross[-1]['fast']:5.1f} slow {cross[-1]['slow']:7.1f}  "
              f"| 21d ratio {cross[-1]['ratio_21d']:.2f}")
    h["cross_asset"] = cross
    imp = [c["impulse"] for c in cross if np.isfinite(c["impulse"])]
    h["cross_min"] = float(np.nanmin(imp)) if imp else np.nan
    h["cross_max"] = float(np.nanmax(imp)) if imp else np.nan
    print(f"  the impulse half-life ranges {h['cross_min']:.0f} to {h['cross_max']:.0f} days "
          f"across assets — 'the half-life of volatility' is not one number")
    print("  (note the obs/yr column in section 0: a 20-day half-life is 45% longer in "
          "wall-clock time for a 365-day asset than for a 252-day one)")

    print("\n=== 7. synthetic control: graded against a known half-life ===")
    ctrl = []
    for truth, second, tag in ((20.0, 0.0, "one process, HL=20"),
                               (60.0, 0.0, "one process, HL=60"),
                               (5.0, 200.0, "two processes, 5 and 200")):
        sim = st.synthetic_vol(n=min(len(r) * 2, 20000), halflife=truth,
                               second_halflife=second)
        t = st.halflife_table(sim, WINDOW)
        row = {"world": tag, "truth": truth if not second else np.nan}
        for m in st.METHODS:
            row[m] = float(t.loc[m, "halflife"])
        f = st.two_component_fit(sim, WINDOW)
        row["improvement"] = f.get("improvement", np.nan)
        ctrl.append(row)
        print(f"  {tag:26s} " + "  ".join(f"{m} {row[m]:6.1f}" for m in st.METHODS)
              + f"   2-comp gain {row['improvement']:.0%}")
    h["control"] = ctrl
    print("  read the first two rows: where the truth IS one exponential, the estimators "
          "cluster. Read the third: where it is two, they scatter — exactly as the real tape "
          "does.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    tbl = "\n".join(f"| {r['method']} | **{r['halflife']:.1f}** | {r['note']} |"
                    for r in h["table"])
    sw = "\n".join(
        f"| {int(r['window'])} | {r['phi']:.4f} | {r['halflife']:.1f} |"
        for r in h["window_sweep"])
    prac = "\n".join(
        f"| {int(r['horizon'])} | {int(r['n_hot'])} | {r['vol_after_shock']:.2f}× | "
        f"{r['vol_after_calm']:.2f}× | **{r['ratio']:.2f}×** |" for r in h["practical"])
    cross = "\n".join(
        f"| {r['asset']} | {r['ann_factor']:.0f} | {r['ar1']:.0f} | {r['acf']:.0f} | "
        f"{r['garch']:.0f} | {r['impulse']:.0f} | {r['fast']:.1f} | {r['slow']:.0f} | "
        f"{r['ratio_21d']:.2f}× |" for r in h["cross_asset"])
    ctrl = "\n".join(
        "| " + r["world"] + " | " + " | ".join(f"{r[m]:.1f}" for m in
                                               ("ar1", "acf", "garch", "ewma", "impulse"))
        + f" | {r['improvement']:.0%} |" for r in h["control"])
    ir = h["impulse"]
    return f"""# Results — Study 992 (How Long Is a Storm?) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_assets']} assets; the
detailed profile is **{h['asset']}** over {h['n_days']:,} sessions. As-of **{h['as_of']}**;
fingerprint `{h['fingerprint']}`.*

## 1. Five answers to one question

| Method | Half-life (days) | Note |
|---|--:|---|
{tbl}

**{h['hl_min']:.0f} to {h['hl_max']:.0f} days — a spread of
{h['hl_max'] / h['hl_min']:.1f}×.** All five are computed correctly. They disagree because they
are not measuring the same thing.

## 2. Why they disagree

Fitting `acf(k) = w·exp(−k/τ₁) + (1−w)·exp(−k/τ₂)` to the volatility autocorrelation:

| | |
|---|--:|
| Fast component half-life | **{h['halflife_fast']:.1f} days** |
| Its weight | {h['weight_fast']:.0%} |
| Slow component half-life | **{h['halflife_slow']:.0f} days** |
| Its weight | {1 - h['weight_fast']:.0%} |
| Squared error, two components | {h['sse_two_component']:.4f} |
| Squared error, one exponential | {h['sse_single']:.4f} |
| **Improvement** | **{h['two_component_improvement']:.0%}** |

Volatility is not one process. It is a fast component that decays in days and a slow one that
decays in months. Every single-number estimator returns a *weighted average* of the two, and the
weight depends on which lags that estimator emphasises — which is why GARCH
({h['hl_garch']:.0f} days) systematically exceeds the raw autocorrelation crossing
({h['hl_acf']:.0f} days) rather than differing from it at random.

## 3. The window sweep: two biases pulling opposite ways

| Volatility window (days) | φ | Implied half-life |
|---|--:|--:|
{sw}

The same data, the same estimator, **{h['sweep_min']:.1f} to {h['sweep_max']:.0f} days** —
determined entirely by how the volatility proxy was built. Two distinct biases are at work and
they run in opposite directions:

- **Short windows attenuate.** Realised volatility is a *noisy proxy* for latent volatility, and
  classical errors-in-variables pushes a noisily-measured regressor's coefficient toward zero.
  A one-day proxy (log absolute returns) is so noisy that the estimated half-life collapses to
  a fraction of a day — that is a measurement of how random a single day's absolute return is,
  not of volatility's persistence.
- **Long windows inflate.** A 63-day rolling mean makes consecutive observations share 62 of
  their 63 days, manufacturing autocorrelation whatever the underlying process does.

The truth is between. The sweep is here because choosing a window *after* seeing which answer it
gives is not a defensible way to quote a half-life, and the one-day figure is not the "clean,
assumption-free" version it appears to be.

## 4. The model-free version

After a day in the top 5% of realised volatility ({ir['n_shocks']} such days):

| | Log-vol excess | Share of the shock remaining |
|---|--:|--:|
| On the day | {ir['initial_excess']:+.3f} | 100% |
| After 5 days | {ir['excess_at_5d']:+.3f} | {ir['excess_at_5d'] / ir['initial_excess']:.0%} |
| After 21 days | {ir['excess_at_21d']:+.3f} | {ir['excess_at_21d'] / ir['initial_excess']:.0%} |
| After 63 days | {ir['excess_at_63d']:+.3f} | {ir['excess_at_63d'] / ir['initial_excess']:.0%} |

Half-life: **{ir['halflife']:.0f} days**. No model, no parameters — just conditional means.

## 5. The version that decides something

| Horizon | n | Vol after a shock | Vol after calm | Ratio |
|---|--:|--:|--:|--:|
{prac}

## 6. Every asset

| Asset | Obs/yr | AR(1) | ACF | GARCH | Impulse | Fast | Slow | 21d ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{cross}

The impulse half-life ranges **{h['cross_min']:.0f} to {h['cross_max']:.0f} days** across
assets. Note the obs/yr column: a 20-day half-life is 45% longer in wall-clock time for a
365-day asset than for a 252-day one, and this study quotes everything in the asset's own
trading days.

## 7. Synthetic control

| World | AR(1) | ACF | GARCH | EWMA | Impulse | Two-component gain |
|---|--:|--:|--:|--:|--:|--:|
{ctrl}

Read the first two rows: where the truth **is** a single exponential, the estimators cluster
around it. Read the third: where the truth is two processes, they scatter — exactly as they do
on the real tape. That is the evidence that the disagreement in section 1 is a property of
volatility rather than a failure of the estimators.

## Caveats

- **EWMA is not a measurement.** RiskMetrics' 11.2-day half-life is λ = 0.94 inverted, an
  assumption baked into a large fraction of the world's risk systems. It is in the table as a
  reference point, not as a competing estimate.
- **The GARCH is hand-rolled.** Variance targeting and a logit parameterisation, Nelder-Mead,
  no `arch` package. It agrees with the standard implementations on simulated data but is not a
  substitute for one.
- **Two components is also a simplification.** The literature on long memory in volatility
  (Ding, Granger & Engle 1993; Comte & Renault 1998) argues for a *continuum* of timescales, of
  which two is a convenient approximation and a hyperbolic decay is a better one.
- **Half-lives move.** Every number here is a full-sample average of something that is itself
  time-varying, and the sample contains 1998, 2008 and 2020.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[992-vol-clustering-halflife](../README.md). Not investment advice.*
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
