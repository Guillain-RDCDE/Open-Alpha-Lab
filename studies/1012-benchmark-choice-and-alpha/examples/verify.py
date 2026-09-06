"""Real-tape verification — Study 1012 (Choose Your Benchmark). Regenerates docs/results.md.

Runs every fund against every defensible benchmark and reports the whole alpha
surface, compares the spread across specifications against the standard error of any single one,
counts how many funds change the sign — or the significance — of their alpha depending on the
comparator, climbs a specification ladder from one factor to five, bootstraps whether the data
can identify a single best benchmark at all, and uses a synthetic fund with a **known** zero
alpha to measure how often a plausible mis-specification produces a significant false one.

    python studies/1012-benchmark-choice-and-alpha/examples/verify.py            # cache-only
    python studies/1012-benchmark-choice-and-alpha/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from benchmark import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


HEADLINE_FUND = "XLK"


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "fingerprint": data.fingerprint(px)}

    R = px.pct_change()
    fcols = [c for c in data.FUNDS if c in R.columns
             and R[c].dropna().shape[0] > 1500]
    bcols = [c for c in data.BENCHMARKS if c in R.columns
             and R[c].dropna().shape[0] > 1500]
    funds, benches = R[fcols], R[bcols]
    rf = R[data.BILLS] if data.BILLS in R.columns else None
    h["n_funds"] = int(len(fcols))
    h["n_benchmarks"] = int(len(bcols))
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {len(fcols)} funds against {len(bcols)} candidate benchmarks")
    print(f"  benchmarks: {', '.join(bcols)}")

    print("\n=== 1. the alpha surface ===")
    grid = st.alpha_grid(funds, benches, rf)
    h["n_pairs"] = int(len(grid))
    piv = grid.pivot(index="fund", columns="benchmark", values="alpha")
    print((piv * 100).round(2).to_string())
    h["grid"] = grid.to_dict("records")
    print(f"  {len(grid)} fund/benchmark pairs. Every cell is a defensible number that")
    print(f"  someone could report as 'the alpha'.")

    print("\n=== 2. how much does the choice move it? ===")
    rng_tbl = st.alpha_range(grid)
    print(rng_tbl[["n_benchmarks", "alpha_min", "alpha_max", "alpha_spread",
                   "median_se", "spread_over_se", "sign_flips"]].round(4).to_string())
    h["ranges"] = rng_tbl.reset_index().to_dict("records")
    h["median_spread"] = float(rng_tbl["alpha_spread"].median())
    h["median_spread_over_se"] = float(rng_tbl["spread_over_se"].median())
    h["share_sign_flip"] = float(rng_tbl["sign_flips"].mean())
    h["share_both_significant"] = float(rng_tbl["significant_both_ways"].mean())
    h["worst_fund"] = str(rng_tbl["alpha_spread"].idxmax())
    h["worst_spread"] = float(rng_tbl["alpha_spread"].max())
    print(f"  median spread of alpha across benchmarks: {h['median_spread']:.2%} a year")
    print(f"  median standard error of a single estimate: "
          f"{rng_tbl['median_se'].median():.2%}")
    print(f"  -> the CHOICE is {h['median_spread_over_se']:.1f}x the NOISE.")
    print(f"  {h['share_sign_flip']:.0%} of funds change the SIGN of their alpha")
    print(f"  {h['share_both_significant']:.0%} are significantly POSITIVE against one")
    print(f"     benchmark and significantly NEGATIVE against another, both at |t| > 2")
    print(f"  worst: {h['worst_fund']} spans {h['worst_spread']:.2%}")

    print("\n=== 3. the specification ladder ===")
    hf = HEADLINE_FUND if HEADLINE_FUND in funds.columns else fcols[0]
    h["headline_fund"] = hf
    L = st.specification_ladder(funds[hf], benches, rf)
    print(L.round(4).to_string())
    h["ladder"] = L.reset_index().to_dict("records")
    h["ladder_first"] = float(L["alpha"].iloc[0])
    h["ladder_last"] = float(L["alpha"].iloc[-1])
    print(f"  {hf}: alpha goes from {h['ladder_first']:+.2%} with one factor to "
          f"{h['ladder_last']:+.2%} with {int(L['n_factors'].iloc[-1])}")
    print(f"  R^2 rises from {L['r2'].iloc[0]:.3f} to {L['r2'].iloc[-1]:.3f}")
    print(f"  every rung is a specification someone publishes. There is no statistical")
    print(f"  rule that says where to stop.")
    ladders = []
    for f in fcols:
        Lf = st.specification_ladder(funds[f], benches, rf)
        if len(Lf) < 2:
            continue
        ladders.append({"fund": f, "alpha_1f": float(Lf["alpha"].iloc[0]),
                        "alpha_full": float(Lf["alpha"].iloc[-1]),
                        "shrinkage": float(abs(Lf["alpha"].iloc[0])
                                           - abs(Lf["alpha"].iloc[-1])),
                        "r2_1f": float(Lf["r2"].iloc[0]),
                        "r2_full": float(Lf["r2"].iloc[-1])})
    h["ladders"] = ladders
    shr = np.mean([l["shrinkage"] for l in ladders])
    h["mean_ladder_shrinkage"] = float(shr)
    print(f"  across all funds, adding factors shrinks |alpha| by {shr:.2%} on average")

    print("\n=== 4. can the data pick a benchmark? ===")
    choices = []
    for f in fcols:
        c = st.can_the_data_choose(funds[f], benches, rf, n_boot=120)
        if not c:
            continue
        choices.append({"fund": f, "modal": c["modal_benchmark"],
                        "modal_share": c["modal_share"], "decisive": c["decisive"]})
        print(f"  {f:6s} best benchmark is {c['modal_benchmark']:5s} in "
              f"{c['modal_share']:.0%} of resamples  "
              f"{'DECISIVE' if c['decisive'] else 'ambiguous'}")
    h["choices"] = choices
    h["share_decisive"] = float(np.mean([c["decisive"] for c in choices])) \
        if choices else np.nan
    print(f"  the data picks decisively for {h['share_decisive']:.0%} of funds.")
    print(f"  For the rest, 'which benchmark' is a judgement call, and the alpha should")
    print(f"  be reported as a range rather than a number.")

    print("\n=== 5. cherry-picking, measured ===")
    picks = []
    for f in fcols:
        b = st.best_fit_benchmark(funds[f], benches, rf)
        if not b:
            continue
        picks.append({"fund": f, **{k: v for k, v in b.items() if k != "table"}})
        print(f"  {f:6s} best FIT {b['best_r2_benchmark']:5s} -> alpha "
              f"{b['alpha_at_best_r2']:+.2%} (t {b['t_at_best_r2']:+.1f}) | "
              f"best ALPHA {b['best_alpha_benchmark']:5s} -> {b['max_alpha']:+.2%} "
              f"(t {b['t_at_max_alpha']:+.1f}) | gain {b['cherry_picking_gain']:+.2%}")
    h["picks"] = picks
    h["cherry_pick_gain"] = float(np.median([p["cherry_picking_gain"] for p in picks]))
    print(f"  median gain from choosing the most flattering benchmark rather than the")
    print(f"  best-fitting one: {h['cherry_pick_gain']:.2%} a year, from a search over")
    print(f"  {len(bcols)} candidates whose selection no reported t-statistic accounts for.")

    print("\n=== 6. encompassing tests ===")
    enc = []
    pairs = [(data.MARKET, data.SMALL), (data.MARKET, data.LARGE_GROWTH),
             (data.LARGE_VALUE, data.LARGE_GROWTH)]
    for f in fcols[:8]:
        for a, b in pairs:
            if a not in benches.columns or b not in benches.columns:
                continue
            e = st.encompassing_test(funds[f], benches[a], benches[b], rf)
            if not e:
                continue
            verdict_s = ("both needed" if e["both_needed"]
                         else f"{a} wins" if e["a_encompasses_b"]
                         else f"{b} wins" if e["b_encompasses_a"] else "neither")
            enc.append({"fund": f, "a": a, "b": b, **e, "verdict": verdict_s})
            print(f"  {f:6s} {a:4s} vs {b:4s}: t_a {e['t_a']:+6.1f}, t_b {e['t_b']:+6.1f}"
                  f"  -> {verdict_s}")
    h["encompassing"] = enc
    h["share_both_needed"] = float(np.mean([e["both_needed"] for e in enc])) if enc else 0.0
    print(f"  both benchmarks significant in {h['share_both_needed']:.0%} of tests, which")
    print(f"  means neither ALONE is the benchmark and any single-index alpha was never")
    print(f"  well defined for those funds.")

    print("\n=== 7. the control: a fund with a KNOWN alpha of zero ===")
    dmg = st.mis_specification_damage(true_alpha=0.0,
                                      factor_corr_grid=(0.0, 0.3, 0.6, 0.9),
                                      n_days=4000, n_reps=10)
    print(dmg.round(4).to_string())
    h["damage"] = dmg.reset_index().to_dict("records")
    h["false_alpha"] = float(dmg["error_only_f1"].abs().max())
    print(f"  the fund's TRUE alpha is exactly zero in every row.")
    print(f"  benchmarked correctly, measured alpha is "
          f"{dmg['alpha_correct'].abs().max():.3%} at worst.")
    print(f"  benchmarked against only one of its two factors: up to "
          f"{h['false_alpha']:.2%} a year of pure fiction.")
    print(f"  and note the direction: the damage is LARGEST when the two factors are")
    print(f"  LEAST correlated, which is the opposite of the usual intuition that")
    print(f"  similar benchmarks are interchangeable.")

    print("\n=== 8. how often is the fiction significant? ===")
    sig = st.false_alpha_significance(true_alpha=0.0, n_days=3000, n_reps=60)
    h["false_sig_rate"] = sig["share_significant_wrong_benchmark"]
    h["true_sig_rate"] = sig["share_significant_right_benchmark"]
    print(f"  across {sig['n_reps']} simulated funds with a TRUE alpha of zero:")
    print(f"    correct benchmark:  |t| > 2 in "
          f"{sig['share_significant_right_benchmark']:.0%} of cases")
    print(f"    wrong benchmark:    |t| > 2 in "
          f"{sig['share_significant_wrong_benchmark']:.0%} of cases")
    print(f"    mean false alpha {sig['mean_false_alpha']:+.2%} "
          f"+/- {sig['sd_false_alpha']:.2%}")
    print(f"  a plausible mis-specification is a machine for producing publishable alpha.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    rng_rows = "\n".join(
        f"| {r['fund']} | {r['alpha_min']:+.2%} | {r['alpha_median']:+.2%} | "
        f"{r['alpha_max']:+.2%} | **{r['alpha_spread']:.2%}** | {r['median_se']:.2%} | "
        f"{r['spread_over_se']:.1f}× | {'**yes**' if r['sign_flips'] else 'no'} | "
        f"{'**yes**' if r['significant_both_ways'] else 'no'} |"
        for r in h["ranges"])
    lad = "\n".join(
        f"| {r['model']} | {int(r['n_factors'])} | {r['alpha']:+.2%} | "
        f"{r['alpha_t']:+.2f} | {r['r2']:.3f} | {r['resid_vol']:.2%} |"
        for r in h["ladder"])
    ch = "\n".join(
        f"| {r['fund']} | {r['modal']} | {r['modal_share']:.0%} | "
        f"{'**yes**' if r['decisive'] else 'no'} |" for r in h["choices"])
    pk = "\n".join(
        f"| {r['fund']} | {r['best_r2_benchmark']} | {r['alpha_at_best_r2']:+.2%} | "
        f"{r['best_alpha_benchmark']} | {r['max_alpha']:+.2%} | "
        f"**{r['cherry_picking_gain']:+.2%}** |" for r in h["picks"])
    dm = "\n".join(
        f"| {r['factor_corr']:.1f} | {r['alpha_correct']:+.3%} | "
        f"{r['alpha_only_f1']:+.3%} | **{r['error_only_f1']:+.3%}** |"
        for r in h["damage"])
    return f"""# Results — Study 1012 (Choose Your Benchmark) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_funds']} funds against
{h['n_benchmarks']} candidate benchmarks, {h['n_pairs']} pairs, Newey-West standard errors. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. How much does the choice move it?

| Fund | Min alpha | Median | Max | Spread | Median SE | Spread/SE | Sign flips | Significant both ways |
|---|--:|--:|--:|--:|--:|--:|:--:|:--:|
{rng_rows}

The median fund's alpha ranges over **{h['median_spread']:.2%} a year** depending only on what it
is measured against — **{h['median_spread_over_se']:.1f}× the median standard error** of any
single estimate.

That comparison is the point. A published alpha comes with a confidence interval reflecting
sampling noise. The specification is a larger source of uncertainty and comes with **no error bar
at all**. {h['share_sign_flip']:.0%} of these funds change the *sign* of their alpha across
benchmarks, and {h['share_both_significant']:.0%} are significantly positive against one and
significantly negative against another, both at |t| > 2. The worst, {h['worst_fund']}, spans
{h['worst_spread']:.2%}.

## 2. The specification ladder

{h['headline_fund']} under progressively richer models:

| Model | Factors | Alpha | t | R² | Residual vol |
|---|--:|--:|--:|--:|--:|
{lad}

Alpha moves from {h['ladder_first']:+.2%} to {h['ladder_last']:+.2%}. Across all funds, adding
factors shrinks |alpha| by {h['mean_ladder_shrinkage']:.2%} on average — each factor absorbs a
piece of what was previously called skill. Every rung is a specification somebody publishes, and
there is no statistical rule that says where to stop.

## 3. Can the data pick a benchmark?

| Fund | Modal best-fit benchmark | Win share | Decisive |
|---|---|--:|:--:|
{ch}

Bootstrapping which candidate fits best, a single one wins over 80% of resamples for
**{h['share_decisive']:.0%} of funds**. For the rest the winner changes from resample to
resample: "which benchmark" is then a judgement call, and the alpha should be reported as a
range.

## 4. Cherry-picking, measured

| Fund | Best fit | Its alpha | Most flattering | Its alpha | Gain |
|---|---|--:|---|--:|--:|
{pk}

Median gain from choosing the most flattering benchmark over the best-fitting one:
**{h['cherry_pick_gain']:.2%} a year**. That comes from a search over {h['n_benchmarks']}
candidates, and no conventionally reported t-statistic accounts for the search.

## 5. Encompassing tests

Both benchmarks came out significant in **{h['share_both_needed']:.0%}** of pairwise tests —
meaning neither *alone* is the benchmark, and the single-index alpha for those funds was never a
well-defined quantity.

## 6. The control — a fund whose true alpha is exactly zero

A simulated fund loading on two correlated factors, with **no alpha at all** by construction:

| Factor correlation | Alpha, correct benchmark | Alpha, one factor only | Error |
|---|--:|--:|--:|
{dm}

Benchmarked correctly, the measured alpha is essentially zero. Benchmarked against only one of
its two factors, it reaches **{h['false_alpha']:.2%} a year of pure fiction**.

Note the direction: the damage is **largest when the two factors are least correlated**, which
inverts the usual intuition that similar benchmarks are interchangeable. A benchmark that
correlates 0.9 with the right one is nearly harmless; one at 0.3 is not.

## 7. How often is the fiction significant?

| | Share with \\|t\\| > 2 |
|---|--:|
| Correct benchmark | {h['true_sig_rate']:.0%} |
| **Wrong benchmark** | **{h['false_sig_rate']:.0%}** |

On funds with a genuinely zero alpha. A bias is a nuisance; a *significant* bias is a machine for
manufacturing publishable findings.

## Caveats

- **These "funds" are index products with known tilts**, not active managers. That is deliberate
  — it makes mis-specification attributable — but a real active fund's exposures drift, which
  makes the problem worse rather than better.
- **Benchmarks here are investable ETFs**, not academic factor portfolios. Fama-French factors
  are long-short and would give different loadings; the qualitative conclusion is unchanged and
  the levels are not.
- **Newey-West with five lags.** Longer lags widen the standard errors further and would raise
  the spread-over-SE ratio, so the reported figure is conservative.
- **The candidate set is a choice too.** Nine benchmarks were selected as "defensible"; a wider
  set would produce a wider spread, and a narrower one a narrower spread. There is no neutral
  ground here, which is itself the finding.
- **No survivorship or fee adjustment.** Both matter for judging managers and neither affects
  the specification-sensitivity measured here.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1012-benchmark-choice-and-alpha](../README.md). Not investment advice.*
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
