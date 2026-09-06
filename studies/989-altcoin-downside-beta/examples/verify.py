"""Real-tape verification — Study 989 (The One-Way Beta). Regenerates docs/results.md.

Fits up- and down-betas for every major altcoin against Bitcoin, tests the
difference with a block bootstrap that resamples the split itself, corroborates with the
Bawa-Lindenberg and Hogan-Warren downside betas and with coskewness, benchmarks tail correlation
against what a bivariate normal would have produced, controls for time-varying betas by fitting
within eras, and prices the whole thing as capture ratios and drawdowns.

    python studies/989-altcoin-downside-beta/examples/verify.py            # cache-only
    python studies/989-altcoin-downside-beta/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from onewaybeta import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    bench = rets[data.BENCHMARK].dropna()
    h: dict = {"as_of": data.AS_OF, "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"ann vol {s.pct_change().std() * np.sqrt(st.CRYPTO_DAYS):.0%}")

    alts = {tk: rets[tk].dropna() for tk in data.ALTS if rets[tk].notna().sum() > 800}
    h["n_alts"] = int(len(alts))
    common = st.align(list(alts.values())[0], bench)
    h["years"] = float(len(common) / st.CRYPTO_DAYS)
    print(f"  {len(alts)} altcoins with enough history; roughly "
          f"{h['years']:.1f} years each")

    print("\n=== 1. up-beta and down-beta, per coin ===")
    panel = st.panel_summary(alts, bench)
    print(panel.round(3).to_string())
    h["panel"] = panel.reset_index().to_dict("records")
    h["median_beta_up"] = float(panel["beta_up"].median())
    h["median_beta_down"] = float(panel["beta_down"].median())
    h["median_difference"] = float(panel["difference"].median())
    h["median_naive_t"] = float(panel["naive_t"].median())
    h["share_negative_coskew"] = float((panel["coskewness"] < 0).mean())
    print(f"  median: up {h['median_beta_up']:.2f}, down {h['median_beta_down']:.2f}, "
          f"difference {h['median_difference']:+.2f}")
    print(f"  {int((panel['difference'] > 0).sum())} of {len(panel)} coins have a higher "
          f"down-beta")
    print(f"  median naive two-sample t on the difference: {h['median_naive_t']:+.2f}")

    print("\n=== 2. the same difference, tested honestly ===")
    boots = {}
    for name, r in alts.items():
        at = st.asymmetry_test(r, bench, n_boot=800)
        boots[name] = at
        print(f"  {name:9s} diff {at['difference']:+.3f}  naive t "
              f"{st.naive_two_sample_t(st.conditional_betas(r, bench)):+6.2f}  "
              f"bootstrap t {at['t']:+6.2f}  95% [{at['lo']:+.2f}, {at['hi']:+.2f}]")
    h["bootstraps"] = {k: {kk: v[kk] for kk in ("difference", "boot_sd", "t", "lo", "hi",
                                                "share_positive")}
                       for k, v in boots.items()}
    h["median_boot_t"] = float(np.nanmedian([v["t"] for v in boots.values()]))
    print(f"  median bootstrap t: {h['median_boot_t']:+.2f} against a median naive "
          f"{h['median_naive_t']:+.2f}")
    print("  the gap is the price of pretending the up/down split is not itself random")

    print("\n=== 3. does anything else agree? ===")
    print("  coskewness (negative = the asset suffers when the market moves hard):")
    for name in alts:
        print(f"    {name:9s} {panel.loc[name, 'coskewness']:+.3f}   "
              f"Bawa-Lindenberg {panel.loc[name, 'bawa_lindenberg']:.2f}   "
              f"Hogan-Warren {panel.loc[name, 'hogan_warren']:.2f}")
    agree = int(((panel["difference"] > 0) & (panel["coskewness"] < 0)).sum())
    h["n_agree"] = agree
    print(f"  {agree} of {len(panel)} coins show BOTH a higher down-beta and negative "
          f"coskewness")

    print("\n=== 4. tail correlation, against what a normal would give ===")
    tails = {}
    for name, r in alts.items():
        tc = st.tail_correlation(r, bench)
        if "down_tail" not in tc:
            continue
        tails[name] = tc
        print(f"  {name:9s} overall {tc['overall']:.2f}  down tail {tc['down_tail']:.2f} "
              f"(a normal would give {tc['normal_down_tail']:.2f})  "
              f"excess {tc['excess_down']:+.2f}")
    h["tails"] = tails
    if tails:
        h["median_excess_down"] = float(np.median([t["excess_down"] for t in tails.values()]))
        print(f"  median excess over the normal benchmark: {h['median_excess_down']:+.2f}")
        print("  (Longin & Solnik 2001: measured tail correlation moves even under a normal "
              "with CONSTANT correlation. The raw number means nothing without this column.)")

    print("\n=== 5. threshold sensitivity ===")
    lead = list(alts)[0]
    sw = st.threshold_sweep(alts[lead], bench)
    print(f"  {lead}:")
    print(sw.round(3).to_string())
    h["threshold_sweep"] = sw.reset_index().to_dict("records")

    print("\n=== 6. era by era, controlling for a drifting beta ===")
    eras = st.time_varying_control(alts[lead], bench, n_eras=4)
    print(eras.round(3).to_string())
    h["eras"] = eras.reset_index().to_dict("records")
    pooled = []
    for name, r in alts.items():
        e = st.time_varying_control(r, bench, n_eras=4)
        pooled.append(float(e["difference"].median()))
    h["median_within_era_difference"] = float(np.median(pooled))
    print(f"  median WITHIN-ERA difference across the panel: "
          f"{h['median_within_era_difference']:+.3f} against the full-sample "
          f"{h['median_difference']:+.3f}")

    print("\n=== 7. equities, where the answer is known ===")
    eq = rets[data.EQUITY].dropna()
    eq_bench = eq.rolling(1).mean()
    cb_eq = st.conditional_betas(eq, eq_bench)
    print(f"  SPY on itself (a sanity check): up {cb_eq['beta_up']:.3f}, "
          f"down {cb_eq['beta_down']:.3f}")
    h["equity_selfcheck"] = {"up": cb_eq["beta_up"], "down": cb_eq["beta_down"]}

    print("\n=== 8. what it costs ===")
    caps, dds = {}, {}
    for name, r in alts.items():
        c = st.capture_ratios(r, bench)
        d = st.drawdown_comparison(px[name].dropna(), px[data.BENCHMARK].dropna())
        caps[name], dds[name] = c, d
        print(f"  {name:9s} up capture {c['up_capture']:+.2f}, down capture "
              f"{c['down_capture']:+.2f}, maxDD {d['max_dd']:.0%} vs Bitcoin's "
              f"{d['bench_max_dd']:.0%} ({d['dd_ratio']:.2f}x)")
    h["captures"] = caps
    h["drawdowns"] = dds
    h["median_up_capture"] = float(np.median([c["up_capture"] for c in caps.values()]))
    h["median_down_capture"] = float(np.median([c["down_capture"] for c in caps.values()]))
    h["median_max_dd"] = float(np.median([d["max_dd"] for d in dds.values()]))
    h["bench_max_dd"] = float(np.median([d["bench_max_dd"] for d in dds.values()]))
    h["median_dd_ratio"] = float(np.median([d["dd_ratio"] for d in dds.values()]))

    print("\n=== 9. the control that decides the verdict ===")
    naive_hits, boot_hits = [], []
    for s in range(20):
        sim = st.synthetic_world(n=len(common), beta_up=1.5, beta_down=1.5, seed=989 + s)
        cb = st.conditional_betas(sim["alt"], sim["bench"])
        naive_hits.append(abs(st.naive_two_sample_t(cb)) >= 2)
        boot_hits.append(abs(st.asymmetry_test(sim["alt"], sim["bench"],
                                               n_boot=200)["t"]) >= 2)
    h["null_false_positive"] = float(np.mean(naive_hits))
    h["null_false_positive_boot"] = float(np.mean(boot_hits))
    print(f"  on a SYMMETRIC simulated world with one beta and {len(common):,} days:")
    print(f"    the naive two-sample test declares asymmetry in "
          f"{h['null_false_positive']:.0%} of runs")
    print(f"    the block bootstrap declares it in "
          f"{h['null_false_positive_boot']:.0%}")
    power = []
    for s in range(10):
        sim = st.synthetic_world(n=len(common), beta_up=1.0, beta_down=2.0, seed=989 + s)
        power.append(abs(st.asymmetry_test(sim["alt"], sim["bench"], n_boot=200)["t"]) >= 2)
    h["null_power"] = float(np.mean(power))
    print(f"  and with a genuine 1.0-vs-2.0 asymmetry planted, the bootstrap finds it in "
          f"{h['null_power']:.0%} of runs — so it has power, it is not merely conservative")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    panel = "\n".join(
        f"| {r['asset']} | {r['beta']:.2f} | {r['beta_up']:.2f} | {r['beta_down']:.2f} | "
        f"**{r['difference']:+.2f}** | {r['naive_t']:+.2f} | {r['bawa_lindenberg']:.2f} | "
        f"{r['hogan_warren']:.2f} | {r['coskewness']:+.3f} |" for r in h["panel"])
    boots = "\n".join(
        f"| {k} | {vv['difference']:+.3f} | {vv['boot_sd']:.3f} | **{vv['t']:+.2f}** | "
        f"[{vv['lo']:+.2f}, {vv['hi']:+.2f}] | {vv['share_positive']:.0%} |"
        for k, vv in h["bootstraps"].items())
    tails = "\n".join(
        f"| {k} | {vv['overall']:.2f} | {vv['down_tail']:.2f} | {vv['normal_down_tail']:.2f} | "
        f"{vv['excess_down']:+.2f} |" for k, vv in h["tails"].items())
    sweep = "\n".join(
        f"| {r['threshold']:.0%} | {int(r['n_up'])} | {int(r['n_down'])} | "
        f"{r['beta_up']:.2f} | {r['beta_down']:.2f} | {r['difference']:+.2f} | "
        f"{r['naive_t']:+.2f} |" for r in h["threshold_sweep"])
    eras = "\n".join(
        f"| {r['era']} | {int(r['n'])} | {r['beta_all']:.2f} | {r['beta_up']:.2f} | "
        f"{r['beta_down']:.2f} | {r['difference']:+.2f} |" for r in h["eras"])
    caps = "\n".join(
        f"| {k} | {vv['up_capture']:+.2f} | {h['captures'][k]['down_capture']:+.2f} | "
        f"{h['drawdowns'][k]['max_dd']:.0%} | {h['drawdowns'][k]['bench_max_dd']:.0%} | "
        f"{h['drawdowns'][k]['dd_ratio']:.2f}× |" for k, vv in h["captures"].items())
    return f"""# Results — Study 989 (The One-Way Beta) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_alts']} major altcoins
against Bitcoin over roughly {h['years']:.1f} years on a 365-day calendar. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. Up-beta and down-beta

| Asset | Beta | Up-beta | Down-beta | Difference | Naive *t* | Bawa-Lindenberg | Hogan-Warren | Coskewness |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{panel}

Median: up **{h['median_beta_up']:.2f}**, down **{h['median_beta_down']:.2f}**, difference
**{h['median_difference']:+.2f}**.

That table is where most write-ups stop. The naive *t* column is the one to distrust.

## 2. The same difference, tested honestly

The naive two-sample *t* treats the up/down split as if it were given. It is not: which days
count as down days is itself random, and crypto returns are volatility-clustered, so a
day-level test understates the uncertainty twice. Block-bootstrapping the whole procedure —
re-deriving the split inside every resample:

| Asset | Difference | Bootstrap SD | Bootstrap *t* | 95% interval | Share > 0 |
|---|--:|--:|--:|--:|--:|
{boots}

Median bootstrap *t*: **{h['median_boot_t']:+.2f}**, against a median naive
{h['median_naive_t']:+.2f}.

## 3. Corroboration

One measurement of an asymmetry is an artefact factory. Three independent ones agreeing is
evidence. **{h['n_agree']} of {h['n_alts']}** coins show *both* a higher down-beta and negative
coskewness ({h['share_negative_coskew']:.0%} of the panel has negative coskewness).

## 4. Tail correlation, against what a normal would give

| Asset | Overall ρ | Down-tail ρ | A normal would give | Excess |
|---|--:|--:|--:|--:|
{tails}

Longin & Solnik (2001): measured correlation changes in the tails **even under a bivariate
normal with constant correlation**, purely because conditioning truncates the distribution. The
raw down-tail column means nothing without the simulated benchmark next to it. Median excess
over the benchmark: **{h.get('median_excess_down', float('nan')):+.2f}**.

## 5. How is "a down day" defined?

| Threshold | Up days | Down days | Up-beta | Down-beta | Difference | Naive *t* |
|---|--:|--:|--:|--:|--:|--:|
{sweep}

## 6. Era by era

A beta that drifted upward will fake an up/down difference if the high-beta years happened to be
bad years. Fitting within eras removes that channel:

| Era | n | Beta | Up-beta | Down-beta | Difference |
|---|--:|--:|--:|--:|--:|
{eras}

Median **within-era** difference across the panel:
**{h['median_within_era_difference']:+.3f}**, against the full-sample
{h['median_difference']:+.3f}.

## 7. What it costs

| Asset | Up capture | Down capture | Max DD | Bitcoin's DD | Ratio |
|---|--:|--:|--:|--:|--:|
{caps}

Capture is computed as a **per-day geometric mean**, not by compounding every up day and
dividing. The textbook version saturates on daily data: compound two thousand down days and
both numerator and denominator approach −100%, so the ratio approaches 1 regardless of the
truth. That failure mode is pinned as a unit test
(`test_capture_ratios_do_not_saturate_on_a_long_sample`).

## 8. The control that decides the verdict

On a **symmetric** simulated world — one beta, no asymmetry anywhere — with the same number of
days as the real sample:

| | Declares asymmetry |
|---|--:|
| The naive two-sample test | **{h['null_false_positive']:.0%}** of runs |
| The block bootstrap | {h['null_false_positive_boot']:.0%} of runs |

And with a genuine 1.0-versus-2.0 asymmetry planted, the bootstrap finds it in
**{h['null_power']:.0%}** of runs — so it has power; it is not merely conservative.

## Caveats

- **Eight years, one market cycle and a bit.** Crypto's history is short and its regimes are
  extreme. Beta estimates from 2021 and 2023 describe different worlds.
- **Survivorship.** These are the majors that *survived* to be in a 2026 ticker list. The
  altcoins that went to zero had, by definition, the highest down-betas of all, and none of
  them are in this table. This is the largest bias in the study and it runs in one direction.
- **Yahoo's crypto coverage** before late 2017 is patchy for the smaller majors, which sets the
  start date.
- **Beta is not the risk.** Even a symmetric 1.5 beta on an asset with 60-80% volatility is a
  position most people size wrong; the asymmetry question is second-order next to that.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[989-altcoin-downside-beta](../README.md). Not investment advice.*
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
