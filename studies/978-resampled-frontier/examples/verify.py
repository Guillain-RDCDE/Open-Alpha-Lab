"""Real-tape verification — Study 978 (The Resampled Frontier). Regenerates docs/results.md.

Runs Michaud resampling against plain optimisation, default shrinkage and 1/N on two
panels and two objectives, out of sample with costs; measures how far the resampled portfolio
sits from each competitor; and scores all four against a *known* mean and covariance in
simulation, where the utility gap can actually be computed.

    python studies/978-resampled-frontier/examples/verify.py            # cache-only
    python studies/978-resampled-frontier/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from resampled import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 504
STEP = 63
COST_BPS = 5.0
N_RESAMPLES = 60


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    panels = {"multi-asset": [c for c in data.MULTI if rets[c].notna().sum() > 3000],
              "sectors": [c for c in data.SECTORS if rets[c].notna().sum() > 1500]}
    h: dict = {"as_of": data.AS_OF, "window": WINDOW, "step": STEP,
               "n_resamples": N_RESAMPLES, "fingerprint": data.fingerprint(px),
               "n_assets": len(panels["multi-asset"])}

    print(f"as-of {data.AS_OF}   window {WINDOW}d every {STEP}d   {N_RESAMPLES} resamples per "
          f"rebalance   fp {data.fingerprint(px)}")
    for tag, cols in panels.items():
        print(f"  {tag:12s} {len(cols):2d} assets")

    results = {}
    for tag, cols in panels.items():
        sub = rets[cols].dropna(how="any")
        for obj in st.OBJECTIVES:
            print(f"\n=== {tag} / {st.OBJECTIVE_LABEL[obj]} "
                  f"({len(sub):,} sessions) ===")
            wf = st.walk_forward(sub, obj, window=WINDOW, step=STEP, cost_bps=COST_BPS,
                                 n_resamples=N_RESAMPLES)
            s = st.summarise(wf)
            print("  method                                    return     vol   Sharpe  "
                  "turnover   max w   eff.N   held")
            for m, row in s.iterrows():
                print(f"  {st.METHOD_LABEL[m]:40s} {row['mean_ret']:+8.2%} "
                      f"{row['realised_vol']:7.2%} {row['sharpe']:8.2f} {row['turnover']:9.2f} "
                      f"{row['max_weight']:7.1%} {row['effective_n']:7.1f} {row['n_held']:6.1f}")
            pairs = {o: st.paired_test(wf, "resampled", o) for o in ("plain", "shrunk", "equal")}
            for o, p in pairs.items():
                print(f"    resampled vs {st.METHOD_LABEL[o]:38s} return diff {p['diff']:+.3%}  "
                      f"t {p['t']:+5.2f}  wins {p['win_rate']:.0%} of {p['n']}")
            results[f"{tag}|{obj}"] = {
                "summary": {m: dict(v) for m, v in s.to_dict("index").items()},
                "pairs": pairs, "n_sessions": int(len(sub))}

    h["results"] = results
    head = results[f"multi-asset|max_sharpe"]["summary"]
    h.update({
        "ret_resampled": float(head["resampled"]["mean_ret"]),
        "vol_resampled": float(head["resampled"]["realised_vol"]),
        "sharpe_resampled": float(head["resampled"]["sharpe"]),
        "ret_plain": float(head["plain"]["mean_ret"]),
        "sharpe_plain": float(head["plain"]["sharpe"]),
        "ret_shrunk": float(head["shrunk"]["mean_ret"]),
        "sharpe_shrunk": float(head["shrunk"]["sharpe"]),
        "n_held_resampled": float(head["resampled"]["n_held"]),
        "n_held_plain": float(head["plain"]["n_held"]),
        "max_weight_resampled": float(head["resampled"]["max_weight"]),
        "max_weight_plain": float(head["plain"]["max_weight"]),
        "t_vs_plain": float(results["multi-asset|max_sharpe"]["pairs"]["plain"]["t"]),
        "t_vs_shrunk": float(results["multi-asset|max_sharpe"]["pairs"]["shrunk"]["t"]),
    })

    print("\n=== how far is the resampled portfolio from each competitor? ===")
    sub = rets[panels["multi-asset"]].dropna(how="any")
    X = sub.iloc[-WINDOW:].to_numpy()
    gaps = {}
    for obj in st.OBJECTIVES:
        w_res = st.resampled_weights(X, obj, n_resamples=200, seed=978)
        w = {m: st.weights_for(m, X, obj) for m in ("plain", "shrunk", "equal")}
        gaps[obj] = {m: st.weight_distance(w_res, v) for m, v in w.items()}
        print(f"  {st.OBJECTIVE_LABEL[obj]:20s} " +
              "  ".join(f"vs {st.METHOD_LABEL[m].split(' (')[0]}: {v:.1%}"
                        for m, v in gaps[obj].items()))
    h["weight_gaps"] = gaps
    h["weight_gap_vs_plain"] = float(gaps["max_sharpe"]["plain"])
    h["weight_gap_vs_shrunk"] = float(gaps["max_sharpe"]["shrunk"])

    print("\n=== parametric versus non-parametric resampling ===")
    for obj in st.OBJECTIVES:
        a = st.resampled_weights(X, obj, n_resamples=200, parametric=True, seed=978)
        b = st.resampled_weights(X, obj, n_resamples=200, parametric=False, seed=978)
        print(f"  {st.OBJECTIVE_LABEL[obj]:20s} the two draws differ by "
              f"{st.weight_distance(a, b):.1%} of the book")
        h[f"param_vs_nonparam_{obj}"] = float(st.weight_distance(a, b))

    print("\n=== against a KNOWN truth (simulation) ===")
    mu_true = sub.mean().to_numpy()
    cov_true = np.cov(sub.to_numpy(), rowvar=False, ddof=1)
    truth_out = {}
    for obj in st.OBJECTIVES:
        exp = st.truth_experiment(mu_true, cov_true, obj, n_obs=WINDOW, n_trials=30,
                                  n_resamples=N_RESAMPLES, seed=978)
        g = exp.groupby("method").agg(utility_gap=("utility_gap", "mean"),
                                      distance=("distance_to_optimal", "mean"))
        truth_out[obj] = {m: dict(v) for m, v in g.to_dict("index").items()}
        print(f"  {st.OBJECTIVE_LABEL[obj]}:")
        for m in st.METHODS:
            print(f"    {st.METHOD_LABEL[m]:40s} utility gap {g.loc[m, 'utility_gap']:.5f}  "
                  f"distance to the optimal weights {g.loc[m, 'distance']:.1%}")
    h["truth"] = truth_out
    for m in st.METHODS:
        h[f"utility_gap_{m}"] = float(truth_out["max_sharpe"][m]["utility_gap"])

    print("\n=== how many resamples are enough? ===")
    conv = []
    for nb in (10, 25, 50, 100, 200, 400):
        a = st.resampled_weights(X, "max_sharpe", n_resamples=nb, seed=1)
        b = st.resampled_weights(X, "max_sharpe", n_resamples=nb, seed=2)
        conv.append({"n_resamples": nb, "seed_to_seed_distance": st.weight_distance(a, b)})
        print(f"  {nb:4d} draws: two seeds land {st.weight_distance(a, b):.2%} of the book apart")
    h["convergence"] = conv

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    def block(key):
        s = h["results"][key]["summary"]
        return "\n".join(
            f"| {st.METHOD_LABEL[m]} | {r['mean_ret']:+.2%} | {r['realised_vol']:.2%} | "
            f"{r['sharpe']:+.2f} | {r['turnover']:.2f} | {r['max_weight']:.1%} | "
            f"{r['effective_n']:.1f} | {r['n_held']:.1f} |" for m, r in s.items())
    def pairs(key):
        return "\n".join(
            f"| resampled − {st.METHOD_LABEL[o]} | {p['diff']:+.3%} | {p['t']:+.2f} | "
            f"{p['win_rate']:.0%} |" for o, p in h["results"][key]["pairs"].items())
    truth = "\n".join(
        f"| {st.OBJECTIVE_LABEL[obj]} | {st.METHOD_LABEL[m]} | {d['utility_gap']:.5f} | "
        f"{d['distance']:.1%} |"
        for obj, per in h["truth"].items() for m, d in per.items())
    conv = "\n".join(f"| {r['n_resamples']} | {r['seed_to_seed_distance']:.2%} |"
                     for r in h["convergence"])
    gaps = "\n".join(
        f"| {st.OBJECTIVE_LABEL[obj]} | " +
        " | ".join(f"{g[m]:.1%}" for m in ("plain", "shrunk", "equal")) + " |"
        for obj, g in h["weight_gaps"].items())
    return f"""# Results — Study 978 (The Resampled Frontier) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Michaud resampling with
**{h['n_resamples']} draws per rebalance**, against plain optimisation, a default shrinkage
(Ledoit-Wolf covariance plus 50% shrinkage of the means toward their average) and 1/N. Two
objectives, two panels, long-only throughout, rolling **{h['window']}-day** window every
**{h['step']}** sessions, 5 bps a rebalance. As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`.*

## Multi-asset panel, maximum Sharpe — the headline case

| Method | Return | Volatility | Sharpe | Turnover | Max weight | Effective N | Positions held |
|---|--:|--:|--:|--:|--:|--:|--:|
{block('multi-asset|max_sharpe')}

| Paired comparison | Return difference | *t* | Resampled wins |
|---|--:|--:|--:|
{pairs('multi-asset|max_sharpe')}

## Multi-asset panel, minimum variance

| Method | Return | Volatility | Sharpe | Turnover | Max weight | Effective N | Positions held |
|---|--:|--:|--:|--:|--:|--:|--:|
{block('multi-asset|min_var')}

| Paired comparison | Return difference | *t* | Resampled wins |
|---|--:|--:|--:|
{pairs('multi-asset|min_var')}

## Sector panel, maximum Sharpe

| Method | Return | Volatility | Sharpe | Turnover | Max weight | Effective N | Positions held |
|---|--:|--:|--:|--:|--:|--:|--:|
{block('sectors|max_sharpe')}

## Where the resampled portfolio actually sits

Share of the book separating the resampled weights from each competitor:

| Objective | vs plain | vs shrinkage | vs 1/N |
|---|--:|--:|--:|
{gaps}

This is the study's central number. Resampling is **{h['weight_gap_vs_plain']:.0%}** away from
the single-shot optimiser it is fixing and **{h['weight_gap_vs_shrunk']:.0%}** away from a
one-line shrinkage — it is much closer to the cheap fix than to the thing it fixes, which is
Scherer's (2002) critique made numerical.

## Against a known truth

The one place "better portfolio" is falsifiable: draw samples from a *known* mean and
covariance, build each method's weights from the sample, and score them on the truth.

| Objective | Method | Utility gap | Distance to the optimal weights |
|---|---|--:|--:|
{truth}

## How many resamples are enough?

Two different random seeds, same data:

| Draws | Distance between the two answers |
|---|--:|
{conv}

Parametric and non-parametric resampling differ by
{h['param_vs_nonparam_max_sharpe']:.1%} of the book on maximum Sharpe and
{h['param_vs_nonparam_min_var']:.1%} on minimum variance — a choice that is rarely stated in
papers that use the method.

## Caveats

- **Long-only.** Michaud's construction is long-only and the constraint does a great deal of
  the diversifying on its own (Jagannathan & Ma 2003) — visible here in how close every method
  gets to 1/N on the sector panel.
- **The utility gap is measured under the estimator's own assumptions.** The truth experiment
  draws Gaussian samples from the sample moments; a fat-tailed or regime-switching truth would
  be less kind to every method, including this one.
- **No transaction-cost optimisation.** Resampling's smoother weights are worth more than this
  study credits when turnover is expensive; at 5 bps it barely registers.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[978-resampled-frontier](../README.md). Not investment advice.*
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
