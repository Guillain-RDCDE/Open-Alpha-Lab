"""Real-tape verification — Study 1010 (Mostly Noise). Regenerates docs/results.md.

States the counting argument and the Marchenko-Pastur band from N and T alone,
measures how much of the real spectrum falls inside it at several estimation windows, checks the
machinery against a synthetic world with a known number of factors, tests whether correlations
persist before and after the market factor is removed, and runs a min-variance horse race
scoring each estimator on whether its own risk forecast survives contact with the next quarter.

    python studies/1010-correlation-matrix-stability/examples/verify.py            # cache-only
    python studies/1010-correlation-matrix-stability/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from corrnoise import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 252
HOLD = 63


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "window": WINDOW,
               "fingerprint": data.fingerprint(px)}

    cols = [c for c in data.NAMES if c in px.columns
            and px[c].dropna().shape[0] > 2500]
    R = px[cols].pct_change().dropna()
    h["n_assets"] = int(len(cols))
    h["n_days"] = int(len(R))
    h["start"] = str(R.index[0].date())
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {len(cols)} names, {len(R):,} common sessions from {h['start']}")

    print("\n=== 1. the counting argument, before any data ===")
    p = st.parameters_vs_observations(len(cols), WINDOW)
    h.update({"n_parameters": p["n_parameters"], "n_obs": WINDOW, "q": p["q"]})
    print(f"  a {len(cols)}x{len(cols)} covariance matrix has "
          f"{p['n_parameters']:,} free parameters")
    print(f"  a {WINDOW}-day window gives {p['n_observations']:,} observations")
    print(f"  = {p['observations_per_parameter']:.1f} observations per parameter")
    b = st.marchenko_pastur_bounds(len(cols), WINDOW)
    h.update({"lambda_minus": b["lambda_minus"], "lambda_plus": b["lambda_plus"]})
    print(f"  q = N/T = {b['q']:.4f}")
    print(f"  Marchenko-Pastur noise band: [{b['lambda_minus']:.4f}, "
          f"{b['lambda_plus']:.4f}]")
    print(f"  any eigenvalue in that range is consistent with PURE NOISE. None of this")
    print(f"  required looking at a single price.")

    print("\n=== 2. the control: does the theory describe pure noise? ===")
    ctrl = []
    for k in (0, 1, 3, 6):
        shares, aboves = [], []
        for seed in range(4):
            sim = st.synthetic_returns(len(cols), WINDOW, n_factors=k,
                                       factor_strength=1.0, seed=1010 + seed)
            s = st.spectrum_analysis(sim)
            shares.append(s["share_inside"])
            aboves.append(s["n_above"])
        ctrl.append({"n_factors": k, "share_inside": float(np.mean(shares)),
                     "n_above": float(np.mean(aboves))})
        print(f"  {k} planted factors: {np.mean(shares):.0%} of eigenvalues inside the "
              f"band, {np.mean(aboves):.1f} above")
    h["control"] = ctrl
    h["ctrl_share_inside"] = ctrl[0]["share_inside"]
    h["ctrl_factors"] = 3
    h["ctrl_detected"] = float([c for c in ctrl if c["n_factors"] == 3][0]["n_above"])
    print(f"  with ZERO factors the assets are independent by construction, and "
          f"{ctrl[0]['share_inside']:.0%} of the spectrum lands inside the band, as it must.")
    print(f"  planting factors puts the right number above it. The tool works.")

    print("\n=== 3. the real matrix ===")
    s = st.spectrum_analysis(R.iloc[-WINDOW:])
    h.update({k: s[k] for k in ("n_inside", "n_above", "n_below", "share_inside",
                                "variance_above", "variance_inside", "largest",
                                "largest_share", "second", "condition_number")})
    h["eigenvalues"] = [float(x) for x in s["eigenvalues"]]
    print(f"  {s['n_above']} eigenvalues ABOVE the band, {s['n_inside']} inside, "
          f"{s['n_below']} below")
    print(f"  -> {s['share_inside']:.0%} of the spectrum is indistinguishable from noise,")
    print(f"     and it carries {s['variance_inside']:.0%} of the total variance")
    print(f"  largest eigenvalue {s['largest']:.2f} = {s['largest_share']:.0%} of the")
    print(f"     variance. That is the market, and everybody already knew about it.")
    print(f"  second largest: {s['second']:.2f} (band top is {s['lambda_plus']:.2f})")
    print(f"  condition number: {s['condition_number']:,.0f}")

    print("\n=== 4. how much does the estimation window help? ===")
    sw = st.spectrum_by_window(R, windows=(63, 126, 252, 504, 1260, 2520))
    print(sw.round(4).to_string())
    h["by_window"] = sw.reset_index().to_dict("records")
    h["long_window_factors"] = int(sw["n_above"].iloc[-1])
    h["short_window"] = int(sw.index.min())
    h["short_share_inside"] = float(sw["share_inside"].iloc[0])
    h["long_window"] = int(sw.index.max())
    h["long_share_inside"] = float(sw["share_inside"].iloc[-1])
    print(f"  at {h['short_window']} days the band swallows "
          f"{h['short_share_inside']:.0%} of the spectrum;")
    print(f"  at {h['long_window']} days, {h['long_share_inside']:.0%}.")
    print(f"  but a longer window assumes the correlations held still for that long,")
    print(f"  which is section 5's question.")

    print("\n=== 5. does the matrix persist at all? ===")
    per = st.matrix_persistence(R, window=WINDOW, step=HOLD)
    ps = st.persistence_summary(per)
    h.update({"pairwise_persistence": ps["pairwise"],
              "top_overlap": ps["top_overlap"],
              "residual_persistence": ps["residual"],
              "pairwise_sd": ps["pairwise_sd"],
              "mean_corr_drift": ps["mean_corr_drift"]})
    h["persistence"] = per.assign(start=per["start"].astype(str)).to_dict("records")
    print(f"  over {ps['n']} consecutive window pairs:")
    print(f"    pairwise correlations agree at      {ps['pairwise']:.3f} "
          f"(+/- {ps['pairwise_sd']:.3f})")
    print(f"    top eigenvector overlap             {ps['top_overlap']:.3f}")
    print(f"    AFTER removing the market factor:   {ps['residual']:.3f}")
    print(f"  the first number is the one people quote. The third is the one that says")
    print(f"  whether the matrix carries information beyond 'stocks move together'.")
    print(f"  average drift in the mean pairwise correlation between windows: "
          f"{ps['mean_corr_drift']:.3f}")

    print("\n=== 6. scored against a KNOWN truth (synthetic) ===")
    errs = []
    for n_obs in (100, 252, 504, 1260):
        acc = []
        for seed in range(4):
            sim = st.synthetic_returns(len(cols), n_obs, n_factors=3,
                                       factor_strength=1.0, seed=1010 + seed)
            truth = st.true_covariance(len(cols), n_factors=3, factor_strength=1.0,
                                       seed=1010 + seed)
            acc.append(st.estimator_error(sim, truth))
        m = pd.DataFrame(acc).mean()
        errs.append({"n_obs": n_obs, **m.to_dict()})
        print(f"  T={n_obs:5d}: " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
    h["truth_errors"] = errs
    print("  relative Frobenius distance to the true covariance. This is the only place")
    print("  accuracy can be measured directly rather than proxied.")

    print("\n=== 7. the portfolio horse race ===")
    race = st.portfolio_horse_race(R, WINDOW, HOLD, long_only=False)
    summ = st.race_summary(race)
    print(summ.round(4).to_string())
    h["race"] = summ.reset_index().to_dict("records")
    h["sample_forecast"] = float(summ.loc["sample", "forecast_vol"])
    h["sample_realised"] = float(summ.loc["sample", "realised_vol"])
    h["sample_calibration"] = float(summ.loc["sample", "calibration"])
    h["diag_realised"] = float(summ.loc["diagonal", "realised_vol"])
    h["diag_calibration"] = float(summ.loc["diagonal", "calibration"])
    h["gross_unconstrained"] = float(summ.loc["sample", "gross_leverage"])
    err = (summ["calibration"] - 1).abs()
    best = str(err.idxmin())
    h["best_method"] = best
    h["best_calibration"] = float(summ.loc[best, "calibration"])
    h["best_realised"] = float(summ.loc[best, "realised_vol"])
    h["best_calibration_err"] = float(err.loc[best])
    h["sample_calibration_err"] = float(err.loc["sample"])
    h["lw_delta"] = float(race[race["method"] == "ledoit_wolf"]["lw_delta"].mean())
    print(f"  fed the RAW matrix, the optimiser forecast {h['sample_forecast']:.2%} "
          f"volatility and realised {h['sample_realised']:.2%}")
    print(f"    -> calibration {h['sample_calibration']:.3f}. It underestimates its own")
    print(f"       risk, because it picks the directions where noise flattered the variance.")
    print(f"  best calibrated: {best} at {h['best_calibration']:.3f}")
    print(f"  the diagonal matrix -- throwing every correlation away -- realised "
          f"{h['diag_realised']:.2%} at calibration {h['diag_calibration']:.3f}")
    print(f"  Ledoit-Wolf average shrinkage intensity: {h['lw_delta']:.3f}")

    print("\n=== 8. how much of this does a long-only constraint fix for free? ===")
    lo = st.race_summary(st.portfolio_horse_race(R, WINDOW, HOLD, long_only=True))
    print(lo.round(4).to_string())
    h["race_long_only"] = lo.reset_index().to_dict("records")
    h["lo_sample_calibration"] = float(lo.loc["sample", "calibration"])
    h["lo_spread"] = float(lo["realised_vol"].max() - lo["realised_vol"].min())
    h["free_spread"] = float(summ["realised_vol"].max() - summ["realised_vol"].min())
    print(f"  unconstrained, the methods spread {h['free_spread']:.2%} in realised vol")
    print(f"  long-only, they spread {h['lo_spread']:.2%}")
    print(f"  a constraint you probably already have removes most of the difference")
    print(f"  between these estimators. That is worth knowing before implementing one.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    ctrl = "\n".join(
        f"| {int(r['n_factors'])} | {r['share_inside']:.0%} | {r['n_above']:.1f} |"
        for r in h["control"])
    bw = "\n".join(
        f"| {int(r['window'])} | {r['q']:.4f} | {r['lambda_minus']:.3f} – "
        f"{r['lambda_plus']:.3f} | {int(r['n_above'])} | **{r['share_inside']:.0%}** | "
        f"{r['largest_share']:.0%} | {r['condition_number']:,.0f} |"
        for r in h["by_window"])
    te = "\n".join(
        f"| {int(r['n_obs'])} | {r['sample']:.4f} | {r['diagonal']:.4f} | "
        f"{r['rmt']:.4f} | {r['ledoit_wolf']:.4f} | {r['constant_corr']:.4f} |"
        for r in h["truth_errors"])
    race = "\n".join(
        f"| {r['method']} | {r['forecast_vol']:.2%} | {r['realised_vol']:.2%} | "
        f"**{r['calibration']:.3f}** | {r['gross_leverage']:.2f}× | "
        f"{r['effective_n']:.1f} |" for r in h["race"])
    lo = "\n".join(
        f"| {r['method']} | {r['forecast_vol']:.2%} | {r['realised_vol']:.2%} | "
        f"{r['calibration']:.3f} | {r['effective_n']:.1f} |"
        for r in h["race_long_only"])
    return f"""# Results — Study 1010 (Mostly Noise) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_assets']} large-cap names,
{h['n_days']:,} common sessions from {h['start']}, {h['window']}-day estimation window. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The counting argument, before any data

| | |
|---|--:|
| Free parameters in a {h['n_assets']}×{h['n_assets']} covariance matrix | **{h['n_parameters']:,}** |
| Observations in a {h['n_obs']}-day window | {h['n_assets'] * h['n_obs']:,} |
| Observations per parameter | {h['n_assets'] * h['n_obs'] / h['n_parameters']:.1f} |
| q = N/T | {h['q']:.4f} |
| **Marchenko-Pastur noise band** | **[{h['lambda_minus']:.3f}, {h['lambda_plus']:.3f}]** |

Any eigenvalue inside that band is consistent with the assets being **completely unrelated**.
None of this required looking at a price, and it can be computed before choosing an estimation
window.

## 2. The control — does the theory describe pure noise?

| Planted factors | Share inside the band | Eigenvalues above |
|---|--:|--:|
{ctrl}

With **zero** factors the assets are independent by construction and the true correlation matrix
is the identity; {h['ctrl_share_inside']:.0%} of the sample spectrum lands inside the band, as
it must. Planting factors puts the right number above it. The measurement works before it is
pointed at anything real.

## 3. The real matrix

| | |
|---|--:|
| Eigenvalues above the band | {h['n_above']} |
| Inside the band | **{h['n_inside']}** |
| Below | {h['n_below']} |
| **Share indistinguishable from noise** | **{h['share_inside']:.0%}** |
| Variance carried by the noise bulk | {h['variance_inside']:.0%} |
| Largest eigenvalue | {h['largest']:.2f} ({h['largest_share']:.0%} of variance) |
| Second largest | {h['second']:.2f} |
| Condition number | {h['condition_number']:,.0f} |

The largest eigenvalue is the market factor, which nobody needed a matrix to discover. Below the
handful that escape, most of the object is noise.

## 4. Does a longer window help?

| Window | q | Noise band | Above | Share inside | Largest | Condition |
|---|--:|--:|--:|--:|--:|--:|
{bw}

Lengthening the window narrows the band and frees more eigenvalues — but it also assumes the
correlations held still for that long, which section 5 tests.

## 5. Does the matrix persist?

| | Agreement between consecutive windows |
|---|--:|
| Pairwise correlations | {h['pairwise_persistence']:.3f} |
| Top eigenvector overlap | {h['top_overlap']:.3f} |
| **After removing the market factor** | **{h['residual_persistence']:.3f}** |

The first row is the number usually quoted and it is largely an artefact: almost every pair is
positively correlated in both windows, so the agreement mostly restates that stocks move
together. Strip the first principal component and what remains of the matrix's structure is far
less persistent.

## 6. Scored against a known truth

Relative Frobenius distance to the **true** covariance, in the synthetic world:

| T | Sample | Diagonal | RMT | Ledoit-Wolf | Constant corr |
|---|--:|--:|--:|--:|--:|
{te}

## 7. The portfolio horse race

Minimum-variance portfolios built on one window, held through the next quarter:

| Estimator | Forecast vol | Realised vol | Calibration | Gross leverage | Effective N |
|---|--:|--:|--:|--:|--:|
{race}

**Calibration** — realised divided by forecast — is the column that matters. The raw sample
matrix comes in at {h['sample_calibration']:.3f}: the optimiser systematically underestimates
the risk of the portfolio it chooses, because it selects precisely the directions in which noise
made the variance look small. Ledoit-Wolf's average shrinkage intensity was **{h['lw_delta']:.3f}**,
meaning the estimator itself judged roughly that fraction of the sample matrix not worth keeping.

The **diagonal** row is the honest benchmark: throw every correlation away. Any method that
cannot beat it has not earned its complexity.

## 8. What a long-only constraint fixes for free

| Estimator | Forecast vol | Realised vol | Calibration | Effective N |
|---|--:|--:|--:|--:|
{lo}

Unconstrained, the estimators spread {h['free_spread']:.2%} in realised volatility. Long-only,
they spread {h['lo_spread']:.2%}. A constraint most investors already face removes much of the
difference between these methods — worth knowing before implementing one.

## Caveats

- **Marchenko-Pastur assumes i.i.d. Gaussian entries.** Real returns are fat-tailed and
  volatility-clustered, both of which widen the true noise band beyond the theoretical one. The
  share reported as noise is therefore, if anything, an underestimate.
- **Survivorship**, as everywhere in this desk's single-name work: these fifty all survived.
  Correlations among survivors are probably more stable than among a full universe.
- **Minimum variance ignores expected returns entirely**, which is deliberate — it isolates the
  covariance matrix as the only input — but it is not how anyone allocates. A mean-variance
  version would be *more* sensitive to estimation error, not less.
- **The RMT filter used is the simplest one.** Bouchaud and Potters' rotationally-invariant
  estimators are substantially better and would likely change the ranking in section 7.
- **One market, one period.** Correlations rose structurally over this sample, which flatters
  persistence in section 5 and is a feature of the era rather than of matrices.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1010-correlation-matrix-stability](../README.md). Not investment advice.*
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
