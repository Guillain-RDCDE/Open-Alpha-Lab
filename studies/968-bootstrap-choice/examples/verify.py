"""Real-tape verification — Study 968 (Which Bootstrap). Regenerates docs/results.md.

Measures the empirical coverage of nominal 95% intervals for the mean and the Sharpe
ratio under four resampling schemes and an analytic alternative, in four simulated worlds
(i.i.d., volatility clustering, autocorrelation, both), sweeps the block length, and then shows
how much the choice moves the intervals published on four real tapes.

    python studies/968-bootstrap-choice/examples/verify.py            # cache-only
    python studies/968-bootstrap-choice/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from boot_choice import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


# Coverage is a Monte Carlo over a Monte Carlo, so the budget is stated rather than maximised:
# 150 replications resolve a coverage difference of about 4 points, which is the size of the
# effect this study is looking for, and 300 bootstrap draws are enough for a percentile
# interval whose own noise is small next to that.
N_REPS = 150
N_BOOT = 300
WORLDS = {
    "iid": dict(ar1=0.0, signal_strength=0.0),
    "garch": dict(ar1=0.0, signal_strength=1.0),
    "ar1": dict(ar1=0.25, signal_strength=0.0),
    "both": dict(ar1=0.25, signal_strength=1.0),
}
WORLD_LABEL = {"iid": "i.i.d. Student-t", "garch": "volatility clustering, no AR",
               "ar1": "AR(1) = 0.25, constant vol", "both": "clustering + AR(1)"}


def report() -> dict:
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "n_reps": N_REPS, "n_boot": N_BOOT}
    print(f"coverage study: {N_REPS} replications x {N_BOOT} bootstrap draws per interval")

    # ------------------------------------------------------------------ coverage
    cover = {}
    for stat, truth_key in (("mean", "mu_daily"), ("sharpe", "sharpe_ann")):
        print(f"\n=== nominal 95% coverage of the {stat.upper()} interval ===")
        print("  world                        " +
              "  ".join(f"{m[:9]:>9s}" for m in list(st.METHODS) + ["analytic"]))
        cover[stat] = {}
        for w, kw in WORLDS.items():
            def sampler(seed, kw=kw):
                return data.synthetic_returns(n_years=5, seed=seed, **kw)
            tbl = st.coverage_experiment(sampler, truth_key, stat, n_reps=N_REPS,
                                         n_boot=N_BOOT, seed=968)
            cover[stat][w] = {m: dict(tbl.loc[m]) for m in tbl.index}
            print(f"  {WORLD_LABEL[w]:28s} " +
                  "  ".join(f"{tbl.loc[m, 'coverage']:9.1%}" for m in
                            list(st.METHODS) + ["analytic"]))
        print("  widths (same order, mean interval width):")
        for w in WORLDS:
            print(f"  {WORLD_LABEL[w]:28s} " +
                  "  ".join(f"{cover[stat][w][m]['mean_width']:9.4f}" for m in
                            list(st.METHODS) + ["analytic"]))
    h["coverage"] = cover

    spreads = []
    for stat in st.STATS:
        for w in WORLDS:
            vals = [cover[stat][w][m]["coverage"] for m in st.METHODS]
            spreads.append(max(vals) - min(vals))
    h["max_coverage_spread"] = float(max(spreads))
    sharpe_cov = [cover["sharpe"][w][m]["coverage"] for w in WORLDS for m in st.METHODS]
    h["worst_coverage_sharpe"] = float(min(sharpe_cov))
    h["best_coverage_sharpe"] = float(max(sharpe_cov))
    h["iid_coverage_ar1"] = float(cover["mean"]["ar1"]["iid"]["coverage"])

    # the least-bad default: the method whose WORST coverage across all worlds is closest to 95%
    worst_by_method = {m: min(cover[s][w][m]["coverage"] for s in st.STATS for w in WORLDS)
                       for m in st.METHODS}
    best_m = max(worst_by_method, key=worst_by_method.get)
    h["best_method"] = st.METHOD_LABEL[best_m]
    h["best_method_key"] = best_m
    h["worst_by_method"] = {m: float(v) for m, v in worst_by_method.items()}
    h["best_worst_case_gap"] = float(0.95 - worst_by_method[best_m])
    print(f"\n  worst-case coverage by method (over both statistics and all four worlds):")
    for m, v in sorted(worst_by_method.items(), key=lambda kv: -kv[1]):
        print(f"    {st.METHOD_LABEL[m]:38s} {v:6.1%}")
    print(f"  least-bad default: {st.METHOD_LABEL[best_m]} "
          f"({h['best_worst_case_gap']:.1%} below nominal at its worst)")

    # --------------------------------------------------------------- block sweep
    print("\n=== block length sweep (circular block, mean, clustering + AR(1) world) ===")
    def sampler_both(seed):
        return data.synthetic_returns(n_years=5, seed=seed, **WORLDS["both"])
    sw = st.block_sweep(sampler_both, "mu_daily", "mean", n_reps=100, n_boot=N_BOOT)
    for b, row in sw.iterrows():
        print(f"  block {b:3d}: coverage {row['coverage']:6.1%}  mean width {row['mean_width']:.5f}")
    h["block_sweep"] = {int(k): dict(v) for k, v in sw.to_dict("index").items()}
    n_typ = 5 * st.TRADING_DAYS
    print(f"  the n^(1/3) rule would pick block = {st.default_block(n_typ)} for n = {n_typ}")
    h["rule_block"] = int(st.default_block(n_typ))

    # ------------------------------------------------------------- the real tape
    px = data.load_prices()
    print(f"\n=== the real tapes: dependence profile (as-of {data.AS_OF}, fp "
          f"{data.fingerprint(px)}) ===")
    prof, intervals = {}, {}
    for tk in data.TICKERS:
        r = px[tk].dropna().pct_change().dropna()
        p = st.dependence_profile(r)
        prof[tk] = p
        print(f"  {tk:8s} n={p['n']:6,}  AR(1) {p['ar1']:+.3f}  AR(5) {p['ar5']:+.3f}  "
              f"|r| AR(1) {p['abs_ar1']:+.3f}  excess kurtosis {p['kurtosis']:6.1f}  "
              f"skew {p['skew']:+.2f}")
    h["profiles"] = prof
    h["fingerprint"] = data.fingerprint(px)

    print("\n=== the same Sharpe, five intervals ===")
    for tk in data.TICKERS:
        r = px[tk].dropna().pct_change().dropna()
        tbl = st.real_tape_intervals(r, "sharpe", n_boot=4000)
        intervals[tk] = {m: dict(v) for m, v in tbl.to_dict("index").items()}
        print(f"  {tk}: point Sharpe {tbl['point'].iloc[0]:+.3f}")
        for m, row in tbl.iterrows():
            flag = "  <- crosses zero" if row["ci_low"] < 0 < row["ci_high"] else ""
            print(f"    {m:38s} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]  "
                  f"width {row['width']:.3f}{flag}")
    h["intervals"] = intervals
    widths = {tk: [v["width"] for v in intervals[tk].values()] for tk in data.TICKERS}
    h["max_real_width_ratio"] = float(max(max(w) / min(w) - 1 for w in widths.values()))
    spy = intervals["SPY"][h["best_method_key"]]
    h["spy_ci_low"], h["spy_ci_high"] = float(spy["ci_low"]), float(spy["ci_high"])
    print(f"\n  widest / narrowest interval on the same data: "
          f"{1 + h['max_real_width_ratio']:.2f}x")

    print("\n=== does the choice ever change a conclusion? ===")
    flips = []
    for tk in data.TICKERS:
        signs = {m: (v["ci_low"] > 0) for m, v in intervals[tk].items()}
        if len(set(signs.values())) > 1:
            flips.append(tk)
            print(f"  {tk}: 'Sharpe clears zero' is method-dependent -> " +
                  ", ".join(f"{m}:{'yes' if s else 'no'}" for m, s in signs.items()))
    if not flips:
        print("  no tape here flips its 'clears zero' verdict between methods "
              "(the samples are long; on a 3-year backtest they would)")
    h["flips"] = flips

    print("\n=== the same question on a SHORT sample (3 years of SPY) ===")
    r3 = px["SPY"].dropna().pct_change().dropna().iloc[-756:]
    t3 = st.real_tape_intervals(r3, "sharpe", n_boot=4000)
    for m, row in t3.iterrows():
        flag = "  <- crosses zero" if row["ci_low"] < 0 < row["ci_high"] else ""
        print(f"  {m:38s} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]{flag}")
    h["short_sample"] = {m: dict(v) for m, v in t3.to_dict("index").items()}
    h["short_flip"] = bool(len({v["ci_low"] > 0 for v in h["short_sample"].values()}) > 1)

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    cols = list(st.METHODS) + ["analytic"]
    head = " | ".join(st.METHOD_LABEL.get(m, "Analytic (Lo)") for m in cols)
    dash = "|".join(["--:"] * len(cols))
    def block(stat):
        return "\n".join(
            f"| {WORLD_LABEL[w]} | " +
            " | ".join(f"{h['coverage'][stat][w][m]['coverage']:.1%}" for m in cols) + " |"
            for w in WORLDS)
    sweep = "\n".join(f"| {b} | {r['coverage']:.1%} | {r['mean_width']:.5f} |"
                      for b, r in h["block_sweep"].items())
    prof = "\n".join(
        f"| {tk} | {p['n']:,} | {p['ar1']:+.3f} | {p['abs_ar1']:+.3f} | {p['kurtosis']:.1f} |"
        for tk, p in h["profiles"].items())
    ivals = "\n".join(
        f"| {tk} | {m} | {r['point']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
        f"{r['width']:.3f} |"
        for tk in h["tickers"] for m, r in h["intervals"][tk].items())
    short = "\n".join(
        f"| {m} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['width']:.3f} |"
        for m, r in h["short_sample"].items())
    worst = "\n".join(f"| {st.METHOD_LABEL[m]} | {c:.1%} |"
                      for m, c in sorted(h["worst_by_method"].items(), key=lambda kv: -kv[1]))
    return f"""# Results — Study 968 (Which Bootstrap): coverage, not taste

*Generated by [`examples/verify.py`](../examples/verify.py). Coverage is measured by
simulation — {h['n_reps']} independent five-year samples per world, {h['n_boot']} bootstrap
draws per interval — because a confidence interval is a claim about repeated sampling and
cannot be checked on one history. Real tapes (as-of **{h['as_of']}**, fingerprint
`{h['fingerprint']}`) are used only for the second question: how much the choice moves an
interval a desk would publish.*

## Coverage of a nominal 95% interval — the MEAN

| World | {head} |
|---|{dash}|
{block('mean')}

## Coverage of a nominal 95% interval — the SHARPE RATIO

| World | {head} |
|---|{dash}|
{block('sharpe')}

Worst-case coverage by method, taken across both statistics and all four worlds:

| Method | Worst coverage |
|---|--:|
{worst}

The least-bad default is **{h['best_method']}**, whose worst case is
{h['best_worst_case_gap']:.1%} below nominal.

## Block length (circular block, mean, clustering + AR(1))

| Block | Coverage | Mean width |
|---|--:|--:|
{sweep}

The `n^(1/3)` rule of thumb would choose **{h['rule_block']}** at this sample size. Coverage
improves with block length up to a point and then the intervals simply get wide; there is no
free lunch, only a bias-variance trade in the resampling itself.

## The real tapes

| Ticker | n | AR(1) of returns | AR(1) of \\|returns\\| | Excess kurtosis |
|---|--:|--:|--:|--:|
{prof}

Note the pattern that decides everything: returns are barely autocorrelated, but **absolute**
returns strongly are. That is volatility clustering without serial correlation — the case where
an i.i.d. bootstrap of the *mean* is nearly fine and an i.i.d. bootstrap of the *Sharpe* is not.

### The same Sharpe ratio, five intervals

| Ticker | Method | Sharpe | 95% interval | Width |
|---|---|--:|---|--:|
{ivals}

Widest over narrowest interval on identical data: **{1 + h['max_real_width_ratio']:.2f}×**.

### And on a short sample (last 756 sessions of SPY)

| Method | 95% interval | Width |
|---|---|--:|
{short}

Conclusion flips between methods on this short sample: **{h['short_flip']}**. Long samples
forgive a bad bootstrap; the three-year backtests these intervals usually decorate do not.

## Caveats

- **Percentile intervals only.** BCa and studentised bootstraps are better and slower; their
  absence here is a scope choice, not a claim that they would not help.
- **One block-length rule.** Politis & White (2004) give a data-driven optimal block length;
  the `n^(1/3)` rule used as a default is deliberately crude, which is why the sweep is run.
- **The simulated worlds are stylised.** They carry the three features that matter — serial
  correlation, volatility clustering, fat tails — but not regime shifts or jumps.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study [968-bootstrap-choice](../README.md).
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
