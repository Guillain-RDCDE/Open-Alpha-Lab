"""Real-tape verification — Study 975 (Shrink the Matrix). Regenerates docs/results.md.

Runs four covariance estimators through a rolling out-of-sample minimum-variance
backtest on two cross-sections — eleven sectors and forty single names — reports the
in-sample optimism, condition number, turnover and concentration of each, tests the volatility
differences pairwise, and validates every estimator against a known covariance matrix in
simulation.

    python studies/975-covariance-shrinkage/examples/verify.py            # cache-only
    python studies/975-covariance-shrinkage/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from shrinkage import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 252
STEP = 63
COST_BPS = 5.0


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    sectors = [s for s in data.SECTORS if rets[s].notna().sum() > 1500]
    names = [s for s in data.NAMES if rets[s].notna().sum() > 3000]
    h: dict = {"as_of": data.AS_OF, "window": WINDOW, "step": STEP,
               "n_sectors": len(sectors), "n_names": len(names),
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   window {WINDOW}d, re-estimated every {STEP}d   "
          f"fingerprint {data.fingerprint(px)}")
    print(f"  narrow cross-section: {len(sectors)} sector ETFs -> "
          f"{len(sectors) * (len(sectors) + 1) // 2} covariance parameters, "
          f"{WINDOW / (len(sectors) * (len(sectors) + 1) / 2):.1f} rows per parameter")
    print(f"  wide   cross-section: {len(names)} single names -> "
          f"{len(names) * (len(names) + 1) // 2} parameters, "
          f"{WINDOW / (len(names) * (len(names) + 1) / 2):.2f} rows per parameter")
    h["n_params_wide"] = int(len(names) * (len(names) + 1) // 2)
    h["n_params_narrow"] = int(len(sectors) * (len(sectors) + 1) // 2)
    h["sector_rows_per_param"] = float(WINDOW / h["n_params_narrow"])
    h["name_rows_per_param"] = float(WINDOW / h["n_params_wide"])

    results = {}
    for tag, cols in (("sectors", sectors), ("names", names)):
        print(f"\n=== {tag}: rolling minimum-variance, {COST_BPS:.0f} bps a rebalance ===")
        sub = rets[cols].dropna(how="any")
        print(f"  {len(sub):,} common sessions {sub.index[0].date()} -> {sub.index[-1].date()}")
        wf = st.walk_forward(sub, window=WINDOW, step=STEP, cost_bps=COST_BPS)
        s = st.summarise(wf)
        print("  estimator                            delta   promised  realised  optimism  "
              "turnover  max w   shorts   cond.")
        for m, row in s.iterrows():
            print(f"  {st.ESTIMATOR_LABEL[m]:36s} {row['delta']:5.2f}  {row['promised_vol']:8.2%}  "
                  f"{row['realised_vol']:8.2%}  {row['optimism']:+8.1%}  {row['turnover']:8.2f}  "
                  f"{row['max_weight']:5.1%}  {row['short_weight']:6.1%}  {row['condition']:8.0f}")
        best = s.drop(index=["sample"])["realised_vol"].idxmin()
        pair = st.paired_vol_test(wf, "sample", best)
        print(f"  best non-sample estimator: {st.ESTIMATOR_LABEL[best]}")
        print(f"  paired test vs the sample matrix: mean volatility difference "
              f"{pair['diff']:+.3%}, t {pair['t']:+.2f}, wins {pair['win_rate']:.0%} of "
              f"{pair['n']} rebalances")
        results[tag] = {"summary": {m: dict(v) for m, v in s.to_dict("index").items()},
                        "best": best, "paired": pair,
                        "n_sessions": int(len(sub)), "n_rebalances": int(pair["n"])}

    h["results"] = results
    wide = results["names"]["summary"]
    best = results["names"]["best"]
    h["best_method"] = st.ESTIMATOR_LABEL[best]
    h["best_method_key"] = best
    h["wide_optimism_sample"] = float(-wide["sample"]["optimism"])
    h["wide_promised_sample"] = float(wide["sample"]["promised_vol"])
    h["wide_realised_sample"] = float(wide["sample"]["realised_vol"])
    h["wide_realised_best"] = float(wide[best]["realised_vol"])
    h["wide_vol_saving"] = float(1 - wide[best]["realised_vol"] / wide["sample"]["realised_vol"])
    h["wide_paired_t"] = float(results["names"]["paired"]["t"])
    h["wide_win_rate"] = float(results["names"]["paired"]["win_rate"])
    h["n_rebalances"] = int(results["names"]["n_rebalances"])
    h["wide_condition_sample"] = float(wide["sample"]["condition"])
    h["wide_condition_best"] = float(wide[best]["condition"])
    h["turnover_best"] = float(wide[best]["turnover"])
    h["turnover_sample"] = float(wide["sample"]["turnover"])
    h["max_weight_best"] = float(wide[best]["max_weight"])
    h["max_weight_sample"] = float(wide["sample"]["max_weight"])
    h["sector_optimism_sample"] = float(-results["sectors"]["summary"]["sample"]["optimism"])

    print("\n=== window sensitivity (wide cross-section) ===")
    sens = []
    sub = rets[names].dropna(how="any")
    for w in (126, 252, 504, 756):
        wf = st.walk_forward(sub, window=w, step=STEP, cost_bps=COST_BPS)
        s = st.summarise(wf)
        b = s.drop(index=["sample"])["realised_vol"].idxmin()
        sens.append({"window": w, "rows_per_param": w / h["n_params_wide"],
                     "sample_vol": float(s.loc["sample", "realised_vol"]),
                     "best_vol": float(s.loc[b, "realised_vol"]),
                     "best": b, "delta": float(s.loc[b, "delta"])})
        print(f"  window {w:4d}d ({w / h['n_params_wide']:.2f} rows/param): sample "
              f"{s.loc['sample', 'realised_vol']:.2%} vs {st.ESTIMATOR_LABEL[b]} "
              f"{s.loc[b, 'realised_vol']:.2%}  (delta {s.loc[b, 'delta']:.2f})")
    h["window_sensitivity"] = sens

    print("\n=== long-only: does the constraint do the shrinkage for you? ===")
    lo = st.walk_forward(sub, window=WINDOW, step=STEP, long_only=True, cost_bps=COST_BPS)
    slo = st.summarise(lo)
    for m, row in slo.iterrows():
        print(f"  {st.ESTIMATOR_LABEL[m]:36s} realised {row['realised_vol']:.2%}  "
              f"max weight {row['max_weight']:.1%}")
    h["long_only"] = {m: dict(v) for m, v in slo.to_dict("index").items()}
    h["long_only_gap"] = float(slo.loc["sample", "realised_vol"] -
                               slo.drop(index=["sample"])["realised_vol"].min())

    print("\n=== simulation: every estimator against a KNOWN covariance matrix ===")
    rng = np.random.default_rng(975)
    sim_rows = []
    for n_obs in (60, 120, 252, 1000):
        n_assets, rho, vol = 40, 0.35, 0.02
        f = rng.normal(0, vol, n_obs)
        beta = np.full(n_assets, np.sqrt(rho))
        X = np.outer(f, beta) + rng.normal(0, vol * np.sqrt(1 - rho), (n_obs, n_assets))
        truth = (vol ** 2) * (rho * np.ones((n_assets, n_assets)) +
                              (1 - rho) * np.eye(n_assets))
        row = {"n_obs": n_obs}
        for m in st.ESTIMATORS:
            C, delta = st.estimate(X, m)
            row[m] = st.frobenius_error(C, truth)
            if m == "constant_corr":
                row["delta"] = delta
        sim_rows.append(row)
        print(f"  T={n_obs:5d} (N=40): " +
              "  ".join(f"{m}={row[m]:.3f}" for m in st.ESTIMATORS) +
              f"   delta={row['delta']:.2f}")
    h["simulation"] = sim_rows

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    def block(tag):
        s = h["results"][tag]["summary"]
        return "\n".join(
            f"| {st.ESTIMATOR_LABEL[m]} | {r['delta']:.2f} | {r['promised_vol']:.2%} | "
            f"**{r['realised_vol']:.2%}** | {r['optimism']:+.1%} | {r['turnover']:.2f} | "
            f"{r['max_weight']:.1%} | {r['short_weight']:.1%} | {r['condition']:,.0f} |"
            for m, r in s.items())
    sens = "\n".join(
        f"| {r['window']}d | {r['rows_per_param']:.2f} | {r['sample_vol']:.2%} | "
        f"{r['best_vol']:.2%} | {st.ESTIMATOR_LABEL[r['best']]} | {r['delta']:.2f} |"
        for r in h["window_sensitivity"])
    sim = "\n".join(
        "| " + str(r["n_obs"]) + " | " +
        " | ".join(f"{r[m]:.3f}" for m in st.ESTIMATORS) + f" | {r['delta']:.2f} |"
        for r in h["simulation"])
    lo = "\n".join(f"| {st.ESTIMATOR_LABEL[m]} | {r['realised_vol']:.2%} | {r['max_weight']:.1%} |"
                   for m, r in h["long_only"].items())
    heads = " | ".join(st.ESTIMATOR_LABEL[m].replace("Ledoit-Wolf -> ", "LW ")
                       for m in st.ESTIMATORS)
    return f"""# Results — Study 975 (Shrink the Matrix) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Four covariance estimators, a
rolling **{h['window']}-day** window re-estimated every **{h['step']}** sessions, one day of
execution lag and 5 bps a rebalance, scored by the realised volatility of the
minimum-variance portfolio each one builds. Two cross-sections: **{h['n_sectors']} sector
ETFs** ({h['n_params_narrow']} parameters, {h['sector_rows_per_param']:.1f} rows each) and
**{h['n_names']} single names** ({h['n_params_wide']} parameters,
{h['name_rows_per_param']:.2f} rows each). As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`.*

## The narrow cross-section — eleven sectors

| Estimator | Shrinkage δ | Promised vol | Realised vol | Optimism | Turnover | Max weight | Shorts | Condition |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{block('sectors')}

## The wide cross-section — forty names

| Estimator | Shrinkage δ | Promised vol | Realised vol | Optimism | Turnover | Max weight | Shorts | Condition |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{block('names')}

Paired test against the sample matrix: mean volatility difference
{h['results']['names']['paired']['diff']:+.3%}, *t* = **{h['wide_paired_t']:+.2f}** across
{h['n_rebalances']} rebalances, winning **{h['wide_win_rate']:.0%}** of them.

*Optimism* is the promise minus the delivery. A negative number means the optimiser told you it
had built a quieter portfolio than it had.

## Where the line is: window sensitivity

| Window | Rows per parameter | Sample matrix | Best shrunk | Which | δ |
|---|--:|--:|--:|---|--:|
{sens}

## Does a long-only constraint do the same job?

| Estimator | Realised vol | Max weight |
|---|--:|--:|
{lo}

Constraining weights to be positive is itself a form of shrinkage (Jagannathan & Ma 2003), and
the table shows how much of the benefit it captures on its own: the spread between the sample
matrix and the best estimator narrows to {h['long_only_gap']:.2%}.

## Simulation, where the truth is known

Relative Frobenius distance to the true covariance matrix, N = 40 assets:

| Observations | {heads} | δ |
|---|--:|--:|--:|--:|--:|
{sim}

## Caveats

- **No delistings.** The forty names are survivors, which if anything *understates* the
  estimation problem: a real cross-section has entries and exits and even less usable history.
- **Minimum variance only.** Shrinkage matters most where the inverse matrix is used most; a
  risk-parity or equal-weight book barely notices, which is part of why those approaches are
  popular (studies **171** and **976**).
- **Ledoit-Wolf, not the nonlinear version.** Ledoit & Wolf's later nonlinear shrinkage
  dominates the linear estimator tested here; it is heavier to implement and left as a fork.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[975-covariance-shrinkage](../README.md). Not investment advice.*
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
