"""Real-tape verification — Study 990 (Counting the Breaks). Regenerates docs/results.md.

Builds five standard VaR forecasts on six assets at two confidence levels, counts
the breaches, and runs Kupiec's coverage test, Christoffersen's independence test and their
joint version on every combination — then measures how much a passing grade is actually worth by
simulating the power of those tests at realistic sample sizes.

    python studies/990-var-breach-count/examples/verify.py            # cache-only
    python studies/990-var-breach-count/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from breaks import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 500


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "window": WINDOW, "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    assets = {}
    for tk in data.TICKERS:
        if tk == data.CASH:
            continue
        s = rets[tk].dropna()
        if len(s) < 1500:
            continue
        assets[tk] = s
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"ann vol {s.std() * np.sqrt(st.TRADING_DAYS):.0%}  "
              f"worst day {s.min():.1%}  kurtosis {s.kurtosis():.1f}")
    h["n_assets"] = int(len(assets))
    h["typical_n"] = int(np.median([len(s) for s in assets.values()]))

    print(f"\n=== 1. the promise, and how many events there are to check it with ===")
    for lev in st.LEVELS:
        exp = h["typical_n"] * (1 - lev)
        print(f"  at {lev:.0%}: {1 - lev:.0%} of days should breach -> about {exp:.0f} events "
              f"in a typical {h['typical_n']:,}-session history")
    print(f"  the sampling noise on {h['typical_n'] * 0.01:.0f} events is the binding "
          f"constraint on everything below")

    print("\n=== 2. every model, every asset, at 99% ===")
    grids = {}
    for tk, s in assets.items():
        g = st.grade_all(s, 0.99, WINDOW)
        grids[tk] = g
        print(f"\n  {tk}:")
        print("    model                    n   breaches  expected    rate   Kupiec  indep   "
              "joint  maxrun")
        for m, r in g.iterrows():
            flag = ""
            if r["kupiec_p"] < 0.05:
                flag += " K"
            if r["independence_p"] < 0.05:
                flag += " I"
            print(f"    {m:22s} {int(r['n']):5d} {int(r['breaches']):9d} "
                  f"{r['expected']:9.1f} {r['rate']:7.2%} {r['kupiec_p']:8.3f} "
                  f"{r['independence_p']:6.3f} {r['joint_p']:6.3f} "
                  f"{int(r['max_consecutive']):6d}{flag}")
    h["grids"] = {tk: g.reset_index().to_dict("records") for tk, g in grids.items()}

    print("\n=== 3. summary across assets (99%) ===")
    summary = []
    for m in st.MODELS:
        rows = [grids[tk].loc[m] for tk in grids]
        summary.append({
            "model": m,
            "mean_rate": float(np.mean([r["rate"] for r in rows])),
            "kupiec_reject_share": float(np.mean([r["kupiec_p"] < 0.05 for r in rows])),
            "indep_reject_share": float(np.mean([r["independence_p"] < 0.05 for r in rows])),
            "joint_pass_share": float(np.mean([r["joint_p"] >= 0.05 for r in rows])),
            "max_consecutive": int(max(r["max_consecutive"] for r in rows)),
            "mean_excess": float(np.nanmean([r["mean_excess"] for r in rows])),
            "mean_var": float(np.mean([r["mean_var"] for r in rows])),
        })
    sdf = pd.DataFrame(summary).set_index("model")
    print(sdf.round(4).to_string())
    h["summary"] = sdf.reset_index().to_dict("records")

    nrow = sdf.loc["normal"]
    h.update({"normal_rate": float(nrow["mean_rate"]),
              "normal_reject_share": float(nrow["kupiec_reject_share"]),
              "normal_indep_reject_share": float(nrow["indep_reject_share"]),
              "normal_max_consecutive": int(nrow["max_consecutive"]),
              "normal_breach_error": float(abs(nrow["mean_rate"] - 0.01))})
    sdf["error"] = (sdf["mean_rate"] - 0.01).abs()
    best = sdf.sort_values(["joint_pass_share", "error"], ascending=[False, True]).index[0]
    h.update({"best_model": str(best), "best_rate": float(sdf.loc[best, "mean_rate"]),
              "best_joint_pass_share": float(sdf.loc[best, "joint_pass_share"]),
              "best_breach_error": float(sdf.loc[best, "error"])})
    print(f"  -> best by joint-test pass rate then calibration error: {best}")

    print("\n=== 4. the same at 95%, where there are five times as many events ===")
    s95 = []
    for m in st.MODELS:
        rows = [st.grade_model(s, m, 0.95, WINDOW) for s in assets.values()]
        s95.append({"model": m,
                    "mean_rate": float(np.mean([r["rate"] for r in rows])),
                    "kupiec_reject_share": float(np.mean([r["kupiec_p"] < 0.05
                                                          for r in rows])),
                    "indep_reject_share": float(np.mean([r["independence_p"] < 0.05
                                                         for r in rows])),
                    "joint_pass_share": float(np.mean([r["joint_p"] >= 0.05 for r in rows]))})
    d95 = pd.DataFrame(s95).set_index("model")
    print(d95.round(4).to_string())
    h["summary_95"] = d95.reset_index().to_dict("records")
    print("  note how much more often the tests reject here. That is not because the models "
          "are worse at 95% — it is because there are five times as many breaches to count.")

    print("\n=== 5. when they were wrong, how wrong were they? ===")
    worst = {}
    for tk, s in assets.items():
        w = st.worst_breach_stats(s, st.build_var(s, "normal", 0.99, WINDOW))
        if "worst_loss" not in w:
            continue
        worst[tk] = w
        print(f"  {tk:9s} {w['n_breaches']:3d} breaches, average overshoot "
              f"{w['mean_excess_pct_of_var']:+.0%} of the forecast; worst day "
              f"{w['worst_loss']:.1%} against a forecast of {w['var_on_worst_day']:.1%}")
    h["worst"] = worst
    if worst:
        h["normal_mean_excess"] = float(np.mean([w["mean_excess_pct_of_var"]
                                                 for w in worst.values()]))
        k = min(worst, key=lambda t: worst[t]["worst_loss"])
        h["normal_worst_loss"] = float(worst[k]["worst_loss"])
        h["normal_var_that_day"] = float(worst[k]["var_on_worst_day"])
    print("  breach counting is blind to this column entirely — which is the whole argument "
          "for expected shortfall over VaR.")

    print("\n=== 6. how much is a passing grade worth? ===")
    pc = st.power_curve(h["typical_n"], 0.99, n_sims=1500)
    print(pc.round(4).to_string())
    h["power_curve"] = pc.reset_index().to_dict("records")
    at15 = pc[np.isclose(pc["ratio_to_promised"], 1.5)]
    h["power_at_1_5x"] = float(at15["reject_rate"].iloc[0]) if len(at15) else np.nan
    h["days_for_1_5x"] = int(st.days_needed(0.99, 1.5))
    print(f"  a model breaching 1.5x too often is caught {h['power_at_1_5x']:.0%} of the time "
          f"in a {h['typical_n']:,}-session backtest")
    need = []
    for lev in (0.95, 0.99, 0.999):
        for mis in (1.25, 1.5, 2.0):
            d = st.days_needed(lev, mis)
            need.append({"level": lev, "misstatement": mis, "days": d,
                         "years": d / st.TRADING_DAYS})
            print(f"  {lev:.1%} VaR, {mis:.2f}x too loose: need {d:,} sessions "
                  f"({d / st.TRADING_DAYS:.0f} years) for 80% power")
    h["days_needed"] = need

    print("\n=== 7. synthetic control: grading against a known truth ===")
    ctrl = []
    for dft, clus, tag in ((1000, 0.0, "iid normal (normal model is CORRECT)"),
                           (3.0, 0.0, "fat tails, no clustering"),
                           (1000, 0.98, "normal tails, heavy clustering"),
                           (3.0, 0.98, "both")):
        row = {"world": tag}
        sim = st.synthetic_returns(n=8000, df_t=dft, clustering=clus)
        for m in st.MODELS:
            g = st.grade_model(sim, m, 0.99)
            row[m] = g["rate"]
        ctrl.append(row)
        print(f"  {tag:38s} " + "  ".join(f"{m[:6]} {row[m]:.2%}" for m in st.MODELS))
    h["control"] = ctrl
    print("  read the first row: where the normal model is CORRECT by construction, it breaches "
          "at 1%. The apparatus is not biased against it.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    summ = "\n".join(
        f"| {r['model']} | {r['mean_rate']:.2%} | {r['kupiec_reject_share']:.0%} | "
        f"{r['indep_reject_share']:.0%} | {r['joint_pass_share']:.0%} | "
        f"{int(r['max_consecutive'])} | {r['mean_excess']:+.0%} |" for r in h["summary"])
    s95 = "\n".join(
        f"| {r['model']} | {r['mean_rate']:.2%} | {r['kupiec_reject_share']:.0%} | "
        f"{r['indep_reject_share']:.0%} | {r['joint_pass_share']:.0%} |"
        for r in h["summary_95"])
    per_asset = "\n".join(
        f"| {tk} | {r['model']} | {int(r['breaches'])} | {r['expected']:.0f} | "
        f"{r['rate']:.2%} | {r['kupiec_p']:.3f} | {r['independence_p']:.3f} | "
        f"{r['joint_p']:.3f} | {int(r['max_consecutive'])} |"
        for tk, rows in h["grids"].items() for r in rows)
    worst = "\n".join(
        f"| {tk} | {w['n_breaches']} | {w['mean_excess_pct_of_var']:+.0%} | "
        f"{w['worst_loss']:.1%} | {w['var_on_worst_day']:.1%} |"
        for tk, w in h["worst"].items())
    pc = "\n".join(
        f"| {r['ratio_to_promised']:.2f}× | {r['true_rate']:.2%} | "
        f"{r['expected_breaches']:.0f} | **{r['reject_rate']:.0%}** |"
        for r in h["power_curve"])
    need = "\n".join(
        f"| {r['level']:.1%} | {r['misstatement']:.2f}× | {r['days']:,} | {r['years']:.0f} |"
        for r in h["days_needed"])
    ctrl = "\n".join(
        "| " + r["world"] + " | " + " | ".join(f"{r[m]:.2%}" for m in
                                               ("historical", "normal", "student_t", "ewma",
                                                "filtered_historical")) + " |"
        for r in h["control"])
    return f"""# Results — Study 990 (Counting the Breaks) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Five VaR models on
{h['n_assets']} assets, {WINDOW}-session estimation window, forecasts strictly out of sample.
As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The promise, and how many events there are to check it

A 99% VaR should be exceeded on 1% of days. In a typical **{h['typical_n']:,}**-session history
that is about **{h['typical_n'] * 0.01:.0f} events**. Everything below is constrained by the
sampling noise on that many events, and section 6 quantifies exactly how badly.

## 2. Every model, across assets, at 99%

| Model | Mean breach rate | Kupiec rejects | Independence rejects | Joint passes | Longest run | Mean overshoot |
|---|--:|--:|--:|--:|--:|--:|
{summ}

Two failures, not one. **Kupiec rejects** means the model breaches too often — its distribution
is the wrong shape. **Independence rejects** means the breaches arrive in clusters — the model
does not know what today's volatility is. They are different diseases with the same symptom,
and a breach count alone cannot tell them apart.

## 3. Per asset

| Asset | Model | Breaches | Expected | Rate | Kupiec *p* | Indep *p* | Joint *p* | Longest run |
|---|---|--:|--:|--:|--:|--:|--:|--:|
{per_asset}

## 4. The same at 95%

| Model | Mean breach rate | Kupiec rejects | Independence rejects | Joint passes |
|---|--:|--:|--:|--:|
{s95}

The tests reject far more often here. That is **not** because the models are worse at 95% — it
is because there are five times as many breaches to count, so the tests have five times the
power. Which is a warning about how to read section 2.

## 5. When they were wrong, how wrong were they?

The normal model's breaches:

| Asset | Breaches | Average overshoot of its own forecast | Worst day | Forecast that day |
|---|--:|--:|--:|--:|
{worst}

Breach counting is entirely blind to this table. A model that breaches at exactly 1% but
overshoots by 60% when it does is far more dangerous than one that breaches at 1.3% and
overshoots by 10%. This is the whole argument for expected shortfall, and it is why Basel
moved to it.

## 6. How much is a passing grade worth?

Simulated breaches at a *true* rate different from the promised 1%, over
{h['typical_n']:,} sessions, and how often Kupiec catches it:

| True rate ÷ promised | True rate | Expected breaches | Kupiec rejects |
|---|--:|--:|--:|
{pc}

A model breaching **50% too often** is caught only **{h['power_at_1_5x']:.0%}** of the time.
Sessions needed for 80% power:

| VaR level | Misstatement | Sessions | Years |
|---|--:|--:|--:|
{need}

This is the table that should accompany every VaR backtest and never does. "The model passed"
usually means "we did not have enough data to catch it".

## 7. Synthetic control

Grading the models against a world whose true quantiles are known:

| World | historical | normal | student_t | ewma | filtered_historical |
|---|--:|--:|--:|--:|--:|
{ctrl}

Read the first row. On i.i.d. normal returns, where the normal model is **correct by
construction**, it breaches at 1%. The apparatus is not rigged against it — the real tape
simply is not i.i.d. normal.

## Caveats

- **One window length.** Everything uses a {WINDOW}-session estimation window. Shorter windows
  react faster and estimate the tail worse; the trade-off is real and is not swept here.
- **No parameter uncertainty.** The VaR forecasts are treated as known numbers. They are
  estimates, and Escanciano & Olmo (2010) show that ignoring their estimation error makes these
  tests reject too often.
- **Overlapping models, one tape.** The five models are graded on the same six assets, so their
  results are not independent draws; the "share of assets rejecting" columns should be read as
  descriptive rather than as five independent tests.
- **Independence is tested only at lag one.** Christoffersen's Markov test catches
  breach-after-breach clustering. A model whose breaches cluster at a two-week horizon but never
  land on consecutive days would pass it; the duration-based tests (Christoffersen & Pelletier
  2004) are the fix and are not implemented here.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[990-var-breach-count](../README.md). Not investment advice.*
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
