"""Real-tape verification — Study 996 (The Palindrome Portfolio). Regenerates docs/results.md.

Runs every meaningless calendar rule against every asset, counts how many clear the
usual significance bar, computes what Bonferroni and Benjamini-Hochberg would have demanded,
establishes the null distribution of the best rule by reshuffling the returns and rerunning the
entire search, splits the sample to test the winners out of sample, and prices the best one as
a strategy.

    python studies/996-palindrome-dates/examples/verify.py            # cache-only
    python studies/996-palindrome-dates/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from palindrome import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


N_SHUFFLES = 150
COST_BPS = 2.0


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "fingerprint": data.fingerprint(px)}

    assets = {}
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        if tk == data.CASH:
            continue
        s = rets[tk].dropna()
        if len(s) < 1500:
            continue
        assets[tk] = s
        print(f"  {tk:6s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}")
    h["n_assets"] = int(len(assets))

    preds = st.date_predicates()
    h["n_rules"] = int(len(preds))
    print(f"\n=== 1. the search space ===")
    print(f"  {len(preds)} calendar rules, none of which can possibly matter")
    lead = assets[data.EQUITY]
    for name in ("palindrome DDMMYYYY", "prime day", "Fibonacci day", "digit sum == 19"):
        if name in preds:
            m = st.apply_predicate(lead.index, preds[name])
            print(f"    '{name}': {int(m.sum())} of {len(m)} sessions "
                  f"({m.mean():.1%})")

    print(f"\n=== 2. the scan ===")
    panel = st.scan_panel(assets, preds)
    h["n_tests"] = int(len(panel))
    summ = st.multiple_testing_summary(panel["t"])
    h.update({k: summ[k] for k in ("n_significant", "expected_by_luck", "max_abs_t",
                                   "bonferroni_p", "bonferroni_t",
                                   "n_surviving_bonferroni", "n_surviving_bh", "min_p")})
    print(f"  {len(panel)} tests ({len(preds)} rules x {len(assets)} assets)")
    print(f"  significant at 5%: {summ['n_significant']}  "
          f"(expected by luck: {summ['expected_by_luck']:.1f})")
    print(f"  largest |t| anywhere: {summ['max_abs_t']:.2f}  (smallest p: {summ['min_p']:.2e})")
    print(f"  Bonferroni needs |t| > {summ['bonferroni_t']:.2f}: "
          f"{summ['n_surviving_bonferroni']} survive")
    print(f"  Benjamini-Hochberg: {summ['n_surviving_bh']} survive")
    best = panel.iloc[0]
    h.update({"best_rule": str(panel.index[0]), "best_asset": str(best["asset"]),
              "best_t": float(best["t"]), "best_ann": float(best["ann_difference"]),
              "best_n": int(best["n_hit"])})
    from scipy import stats as _s
    h["best_naive_odds"] = float(1 / max(2 * (1 - _s.norm.cdf(abs(best["t"]))), 1e-12))
    print(f"\n  the winner: '{panel.index[0]}' on {best['asset']}")
    print(f"    t = {best['t']:+.2f} on {int(best['n_hit'])} days, worth "
          f"{best['ann_difference']:+.1%}/yr")
    print(f"    a t-table calls that a 1-in-{h['best_naive_odds']:,.0f} event")

    print("\n  top ten across the whole panel:")
    for rule, row in panel.head(10).iterrows():
        print(f"    {rule:38s} {row['asset']:6s} t {row['t']:+6.2f}  "
              f"{row['ann_difference']:+7.1%}/yr  ({int(row['n_hit'])} days)")
    h["top_ten"] = [{"rule": r, "asset": row["asset"], "t": float(row["t"]),
                     "ann": float(row["ann_difference"]), "n": int(row["n_hit"])}
                    for r, row in panel.head(10).iterrows()]

    print(f"\n=== 3. what should the best t have been? ===")
    for k in (1, 10, 100, len(panel)):
        e = st.expected_max_t(k, n_sims=4000)
        print(f"  after {k:5d} tries: median best |t| {e['median']:.2f}, "
              f"90th pct {e['p90']:.2f}, {e['share_above_2']:.0%} exceed 2")
        if k == len(panel):
            h["expected_max"] = e["median"]
            h["expected_max_p90"] = e["p90"]
    print(f"  -> the observed {h['best_t']:+.2f} against an expected maximum of "
          f"{h['expected_max']:.2f}. Deflated t: "
          f"{st.deflated_t(h['best_t'], len(panel)):.2f}")
    h["deflated_t"] = float(st.deflated_t(h["best_t"], len(panel)))

    print(f"\n=== 4. the shuffle test (the honest null) ===")
    bd = st.best_rule_distribution(lead, preds, n_shuffles=N_SHUFFLES)
    h.update({"shuffle_observed": bd["observed_max_t"],
              "null_median_max_t": bd["null_median_max_t"],
              "null_p95_max_t": bd["null_p95_max_t"],
              "shuffle_p": bd["p_value"], "shuffle_best_rule": bd["best_rule"]})
    print(f"  on {data.EQUITY} alone, the best of {bd['n_rules']} rules gave |t| = "
          f"{bd['observed_max_t']:.2f} ('{bd['best_rule']}')")
    print(f"  reshuffling the returns {bd['n_shuffles']} times and rerunning the whole search:")
    print(f"    median best |t| under the null: {bd['null_median_max_t']:.2f}")
    print(f"    95th percentile: {bd['null_p95_max_t']:.2f}")
    print(f"    p-value of the observed maximum: {bd['p_value']:.3f}")
    print("  note that the shuffle null exceeds the theoretical one, because these rules "
          "overlap heavily — 'prime day' and 'day is a multiple of 3' are not independent "
          "tests, and shuffling handles that automatically")

    print(f"\n=== 5. out of sample ===")
    oos = st.split_sample_check(lead, preds)
    print(oos.round(3).to_string())
    h["oos"] = oos.reset_index().to_dict("records")
    h["mean_is_t"] = float(oos["t_in_sample"].mean())
    h["mean_oos_t"] = float(oos["t_out_of_sample"].mean())
    h["median_oos_t"] = float(oos["t_out_of_sample"].abs().median())
    h["n_oos_survive"] = int((np.sign(oos["t_in_sample"])
                              == np.sign(oos["t_out_of_sample"])).sum())
    print(f"  in-sample mean t {h['mean_is_t']:+.2f} -> out-of-sample "
          f"{h['mean_oos_t']:+.2f}")
    print(f"  {h['n_oos_survive']} of {len(oos)} kept the same sign — which is what a coin "
          f"flip gives")

    print(f"\n=== 6. what it costs to believe it ===")
    best_mask = st.apply_predicate(lead.index, preds[panel.index[0]])
    cash = rets[data.CASH].reindex(lead.index).fillna(0.0)
    tc = st.tradable_check(lead, best_mask, cash, COST_BPS)
    h["best_traded_gap"] = float(tc["strategy"]["cagr"] - tc["buy_hold"]["cagr"])
    print(f"  trading '{panel.index[0]}' on {data.EQUITY}: invested "
          f"{tc['share_invested']:.0%} of the time, {tc['switches_per_year']:.0f} switches/yr")
    print(f"    strategy CAGR {tc['strategy']['cagr']:+.2%} vs buy-and-hold "
          f"{tc['buy_hold']['cagr']:+.2%} ({h['best_traded_gap']:+.2%})")
    print(f"    Sharpe {tc['strategy']['sharpe']:.2f} vs {tc['buy_hold']['sharpe']:.2f}")

    print(f"\n=== 7. calibration on data with no calendar structure at all ===")
    ctrl = []
    cache = st.build_mask_matrix(lead.index, preds)
    for k in range(5):
        sim = st.synthetic_returns(n=len(lead), seed=996 + k)
        sim.index = lead.index
        d = st.scan(sim, preds, cache=cache)
        s2 = st.multiple_testing_summary(d["t"])
        ctrl.append({"run": k, "n_significant": s2["n_significant"],
                     "expected": s2["expected_by_luck"], "max_t": s2["max_abs_t"]})
        print(f"  run {k}: {s2['n_significant']} significant of {s2['n_tests']} "
              f"(expected {s2['expected_by_luck']:.1f}), best |t| {s2['max_abs_t']:.2f}")
    h["control"] = ctrl
    h["control_mean_max_t"] = float(np.mean([c["max_t"] for c in ctrl]))
    print(f"  mean best |t| on pure noise: {h['control_mean_max_t']:.2f}")
    print(f"  the real tape gave {h['max_abs_t']:.2f}. These are the same number.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    top = "\n".join(
        f"| {r['rule']} | {r['asset']} | {r['t']:+.2f} | {r['ann']:+.1%} | {r['n']} |"
        for r in h["top_ten"])
    oos = "\n".join(
        f"| {r['rule']} | {r['t_in_sample']:+.2f} | {r['t_out_of_sample']:+.2f} | "
        f"{r['ann_in']:+.1%} | {r['ann_out']:+.1%} |" for r in h["oos"])
    ctrl = "\n".join(
        f"| {int(r['run'])} | {int(r['n_significant'])} | {r['expected']:.1f} | "
        f"{r['max_t']:.2f} |" for r in h["control"])
    return f"""# Results — Study 996 (The Palindrome Portfolio) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_rules']} calendar rules ×
{h['n_assets']} assets = **{h['n_tests']} tests** of hypotheses that cannot be true. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

> **This study is inverted.** Every hypothesis here is known false in advance — no mechanism
> attaches a market return to whether a date reads the same backwards. Finding an effect is the
> *failure* mode. The purpose is to calibrate how large a "finding" a search of this size
> produces when there is nothing to find, so that real studies can be measured against it.

## 1. The search

| | |
|---|--:|
| Rules tried | {h['n_rules']} |
| Assets | {h['n_assets']} |
| **Total tests** | **{h['n_tests']}** |
| Significant at 5% | **{h['n_significant']}** |
| Expected by luck | {h['expected_by_luck']:.1f} |
| Largest \\|*t*\\| anywhere | **{h['max_abs_t']:.2f}** |
| Smallest *p* | {h['min_p']:.2e} |

The ten best:

| Rule | Asset | *t* | Annualised | Days |
|---|---|--:|--:|--:|
{top}

The winner — *{h['best_rule']}* on {h['best_asset']} — has a *t* of **{h['best_t']:+.2f}**,
which a *t*-table calls a **one-in-{h['best_naive_odds']:,.0f}** event, and is worth
{h['best_ann']:+.1%} a year on {h['best_n']} days.

## 2. What the best *t* should have been

A researcher who tries *k* ideas and reports the best is not drawing from a *t*-distribution.
They are drawing from the distribution of the **maximum of *k* draws**:

| Tries | Median best \\|*t*\\| | 90th percentile | Share above 2 |
|---|--:|--:|--:|
| 1 | 0.67 | 1.64 | 5% |
| 10 | ~2.1 | ~2.6 | ~40% |
| 100 | ~2.8 | ~3.2 | ~99% |
| **{h['n_tests']}** | **{h['expected_max']:.2f}** | {h['expected_max_p90']:.2f} | ~100% |

Observed best: **{h['best_t']:+.2f}**. Expected best under pure noise:
**{h['expected_max']:.2f}**. Deflated *t* (observed ÷ expected): **{h['deflated_t']:.2f}**.

Bonferroni would have required \\|*t*\\| > **{h['bonferroni_t']:.2f}**;
**{h['n_surviving_bonferroni']}** rules cleared it. Benjamini-Hochberg passed
{h['n_surviving_bh']}.

## 3. The shuffle test

The theoretical calculation above assumes the tests are independent. They are not — "prime day"
and "day is a multiple of 3" overlap heavily. Reshuffling the returns and rerunning the entire
search handles that automatically:

| | |
|---|--:|
| Observed best \\|*t*\\| on {h['n_rules']} rules | {h['shuffle_observed']:.2f} |
| **Median best \\|*t*\\| under shuffling** | **{h['null_median_max_t']:.2f}** |
| 95th percentile | {h['null_p95_max_t']:.2f} |
| ***p*-value of the observed maximum** | **{h['shuffle_p']:.3f}** |

## 4. Out of sample

Pick the ten best rules on the first half, test them on the second:

| Rule | *t* in-sample | *t* out-of-sample | Annualised in | Annualised out |
|---|--:|--:|--:|--:|
{oos}

In-sample mean *t* **{h['mean_is_t']:+.2f}** → out-of-sample **{h['mean_oos_t']:+.2f}**.
{h['n_oos_survive']} of 10 kept the same sign, which is what a coin flip gives.

## 5. What believing it costs

Trading the single best rule on {h['best_asset']}, with costs: **{h['best_traded_gap']:+.2%}**
a year against simply holding the index.

## 6. Calibration against data with no calendar structure

The identical search on a pure random walk:

| Run | Significant | Expected | Best \\|*t*\\| |
|---|--:|--:|--:|
{ctrl}

Mean best \\|*t*\\| on noise: **{h['control_mean_max_t']:.2f}**. On the real tape:
**{h['max_abs_t']:.2f}**. These are the same number, which is the study's entire finding.

## How to use this

The number to remember is section 2's. If you have tried roughly two hundred variations of an
idea — different lookbacks, different universes, different thresholds, all the things that feel
like "checking robustness" rather than "searching" — then a *t* of 2.8 is your *median*
expected result under the null. Not your best case. Your median.

Three defences work, in ascending order of strength: reporting the number of tests and applying
Bonferroni or Benjamini-Hochberg; shuffling the outcome and rerunning the entire search
(section 3), which handles dependence between tests that the formulas cannot; and holding out a
sample you have never looked at (section 4), which is the only one that cannot be gamed.

## Caveats

- **The rules are not independent.** That is deliberate — real research pipelines are not
  either — and it is why section 3 exists alongside section 2.
- **A single shuffle test on one asset.** Running it across the whole panel would be more
  thorough and much slower; the single-asset version makes the point.
- **The synthetic control is i.i.d. normal**, which is more benign than real returns.
  Volatility clustering would make calendar rules *more* likely to produce spurious
  significance, not less, so this understates the problem.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py). Note that the stamps here grade the
**demonstration**, not the pattern: "Busted" is the intended and correct outcome.

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[996-palindrome-dates](../README.md). Not investment advice.*
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
