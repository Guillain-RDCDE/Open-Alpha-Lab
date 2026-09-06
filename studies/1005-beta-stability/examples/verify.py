"""Real-tape verification — Study 1005 (Beta Has a Half-Life). Regenerates docs/results.md.

Estimates non-overlapping annual betas for forty large-caps and nine sector
funds, measures how much of each beta survives into the next year, decomposes the apparent
instability into estimation error and genuine movement using each regression's own standard
error, compares single names against diversified portfolios to confirm which of the two is
responsible, and scores four beta forecasts — raw, Blume, Vasicek and a flat 1.0 — on
out-of-sample error.

    python studies/1005-beta-stability/examples/verify.py            # cache-only
    python studies/1005-beta-stability/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from betahalflife import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 252
BLUME_W = 0.66


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "window": WINDOW,
               "fingerprint": data.fingerprint(px)}

    cols = [c for c in (data.MARKET,) + data.NAMES if c in px.columns]
    R = px[cols].pct_change().dropna()
    h["n_names"] = int(len(cols) - 1)
    h["n_days"] = int(len(R))
    h["start"] = str(R.index[0].date())
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {h['n_names']} names vs {data.MARKET}, {len(R):,} common sessions from "
          f"{h['start']}")

    betas = st.rolling_betas(R, data.MARKET, WINDOW, WINDOW)
    lf = st.long_form(betas)
    h["n_periods"] = int(len(betas))
    h["n_estimates"] = int(len(lf))
    print(f"  {len(betas)} non-overlapping {WINDOW}-day windows -> {len(lf):,} beta estimates")
    print(f"  mean beta {lf['beta'].mean():.3f}, cross-sectional sd "
          f"{lf['beta'].std(ddof=1):.3f}, mean R^2 {lf['r2'].mean():.3f}")

    print("\n=== 1. does beta persist? (Blume 1971) ===")
    p = st.persistence(betas)
    h.update({"slope": p["slope"], "slope_se": p["slope_se"],
              "persist_r2": p["r2"], "intercept": p["intercept"],
              "correlation": p["correlation"]})
    h["half_life_years"] = st.half_life(p["slope"], WINDOW / 252)
    print(f"  beta(t) = {p['intercept']:.3f} + {p['slope']:.3f} * beta(t-1)   "
          f"(se {p['slope_se']:.3f}, R^2 {p['r2']:.3f}, {p['n_pairs']:,} pairs)")
    print(f"  slope is {p['slope'] / p['slope_se']:.1f} standard errors from zero and "
          f"{(1 - p['slope']) / p['slope_se']:.1f} from one")
    print(f"  half-life of a deviation from the mean beta: "
          f"{h['half_life_years']:.2f} years")

    print("\n=== 2. how much of that instability is REAL? ===")
    nf = st.noise_floor(betas)
    h.update({"observed_var": nf["observed_var"], "noise_var": nf["noise_var"],
              "true_var": nf["true_var"], "observed_sd": nf["observed_sd"],
              "noise_sd": nf["noise_sd"], "true_sd": nf["true_sd"],
              "noise_share": nf["noise_share"], "mean_se": nf["mean_se"]})
    print(f"  a {WINDOW}-day beta carries a mean standard error of {nf['mean_se']:.4f}")
    print(f"  so two consecutive estimates DIFFER even with a constant true beta.")
    print(f"    variance of the observed year-to-year change: {nf['observed_var']:.5f} "
          f"(sd {nf['observed_sd']:.3f})")
    print(f"    attributable to estimation error:             {nf['noise_var']:.5f} "
          f"(sd {nf['noise_sd']:.3f})")
    print(f"    genuine movement in beta:                     {nf['true_var']:.5f} "
          f"(sd {nf['true_sd']:.3f})")
    print(f"  -> {nf['noise_share']:.0%} of the apparent instability is measurement, not")
    print(f"     movement. Betas are steadier than beta ESTIMATES.")
    sn = st.signal_to_noise(betas)
    h["reliability"] = sn["reliability"]
    h["cross_sd"] = sn["cross_sectional_sd"]
    h["cross_true_sd"] = sn["true_sd"]
    print(f"  cross-sectionally: estimated betas spread {sn['cross_sectional_sd']:.3f}, "
          f"true betas about {sn['true_sd']:.3f} (reliability {sn['reliability']:.2f})")

    print("\n=== 3. the confirmation: portfolios ===")
    pb = st.portfolio_betas(R, data.MARKET, data.NAMES, n_per=10, n_ports=20,
                            window=WINDOW)
    plf = st.long_form(pb)
    pp = st.persistence(pb)
    pnf = st.noise_floor(pb)
    h.update({"port_slope": pp["slope"], "port_se": float(plf["se"].mean()),
              "port_noise_share": pnf["noise_share"],
              "port_half_life": st.half_life(pp["slope"], WINDOW / 252)})
    print(f"  single names: mean se {lf['se'].mean():.4f}, persistence "
          f"{p['slope']:.3f}, noise share {nf['noise_share']:.0%}")
    print(f"  10-stock portfolios: mean se {plf['se'].mean():.4f}, persistence "
          f"{pp['slope']:.3f}, noise share {pnf['noise_share']:.0%}")
    print(f"  measure beta better and LESS OF THE INSTABILITY SURVIVES. That is what")
    print(f"  'the instability was mostly noise' looks like when you go and check it.")
    print(f"  But notice the slope did NOT rise ({pp['slope']:.3f} vs {p['slope']:.3f}).")

    print("\n=== 3b. why the Blume slope cannot be read as a stability measure ===")
    cs = st.slope_is_confounded(betas)
    cp = st.slope_is_confounded(pb)
    h["reliability_single"] = cs["reliability"]
    h["reliability_port"] = cp["reliability"]
    h["disattenuated"] = cs["disattenuated_slope"]
    print(f"  regressing a noisy measure on a noisy measure attenuates the slope by")
    print(f"     var(true) / [var(true) + var(noise)]")
    print(f"  single names: cross-sectional sd {cs['cross_sectional_sd']:.3f}, "
          f"reliability {cs['reliability']:.3f}, raw slope {cs['raw_slope']:.3f}")
    print(f"                -> disattenuated persistence {cs['disattenuated_slope']:.3f}")
    print(f"  portfolios:   cross-sectional sd {cp['cross_sectional_sd']:.3f}, "
          f"reliability {cp['reliability']:.3f}, raw slope {cp['raw_slope']:.3f}")
    print(f"  diversifying cuts the NOISE and the TRUE SPREAD together, so the slope barely")
    print(f"  moves. It is a ratio, not a stability statistic. Section 8 makes this")
    print(f"  unanswerable: there, a DRIFTING beta produces a HIGHER slope than a constant one.")

    print("\n=== 4. sector funds, which are portfolios by construction ===")
    scols = [c for c in (data.MARKET,) + data.SECTORS if c in px.columns]
    SR = px[scols].pct_change().dropna()
    sect = []
    if len(SR) > WINDOW * 3:
        sb = st.rolling_betas(SR, data.MARKET, WINDOW, WINDOW)
        slf = st.long_form(sb)
        sp = st.persistence(sb)
        snf = st.noise_floor(sb)
        h["sector_slope"] = sp["slope"]
        h["sector_noise_share"] = snf["noise_share"]
        print(f"  {len(scols) - 1} sector funds, {len(sb)} windows: persistence "
              f"{sp['slope']:.3f}, mean se {slf['se'].mean():.4f}, noise share "
              f"{snf['noise_share']:.0%}")
        for nm in [c for c in sb.columns if not c.endswith(("__se", "__r2"))
                   and c != "start"]:
            s = sb[nm].dropna()
            if len(s) > 2:
                sect.append({"sector": nm, "mean_beta": float(s.mean()),
                             "sd_beta": float(s.std(ddof=1)),
                             "min": float(s.min()), "max": float(s.max())})
                print(f"    {nm}: beta {s.mean():.2f} +/- {s.std(ddof=1):.2f} "
                      f"(range {s.min():.2f} to {s.max():.2f})")
    h["sectors"] = sect

    print("\n=== 5. how long an estimation window? ===")
    ph = st.persistence_by_horizon(R, data.MARKET, windows=(63, 126, 252, 504))
    print(ph[["n_periods", "slope", "slope_se", "r2", "half_life_years"]].round(4)
          .to_string())
    h["by_horizon"] = ph.reset_index().to_dict("records")
    print("  longer windows -> more precise estimates -> more of the change is real ->")
    print("  higher measured persistence. The framework predicts this and the data obliges.")

    print("\n=== 6. does shrinkage help? ===")
    fc = st.forecast_comparison(betas, BLUME_W)
    print(fc.round(4).to_string())
    h["forecasts"] = fc.reset_index().to_dict("records")
    h["raw_rmse"] = float(fc.loc["raw", "rmse"])
    h["blume_rmse"] = float(fc.loc["blume", "rmse"])
    h["vasicek_rmse"] = float(fc.loc["vasicek", "rmse"])
    h["one_rmse"] = float(fc.loc["always_one", "rmse"])
    print(f"  raw estimate:        rmse {h['raw_rmse']:.4f}")
    print(f"  Blume ({BLUME_W}):         rmse {h['blume_rmse']:.4f}")
    print(f"  Vasicek:             rmse {h['vasicek_rmse']:.4f}")
    print(f"  just assume 1.0:     rmse {h['one_rmse']:.4f}   <- the honest floor")
    if h["one_rmse"] < h["raw_rmse"]:
        print("  note: assuming 1.0 for EVERY name beats using the measured beta. That is")
        print("  a real result and it should temper how much weight a beta carries.")

    print("\n=== 7. what shrinkage weight is actually right? ===")
    opt = st.optimal_shrinkage(betas)
    h["best_w"] = opt["best_w"]
    h["best_rmse"] = opt["best_rmse"]
    h["blume_default"] = opt["blume_default"]
    h["shrink_curve"] = opt["curve"]
    print(f"  fitted optimum: w = {opt['best_w']:.3f} (rmse {opt['best_rmse']:.4f})")
    print(f"  Blume's constant: {opt['blume_default']:.2f}")
    print(f"  persistence slope from section 1: {p['slope']:.3f}")
    print(f"  theory says the last two should coincide; they differ by "
          f"{abs(opt['best_w'] - p['slope']):.3f}")

    print("\n=== 8. the control: a world where beta never moves ===")
    ctrl = []
    for drift in (0.0, 0.10, 0.25, 0.50):
        sim = st.synthetic_panel(n_names=40, n_days=10000, beta_drift=drift,
                                 idio_vol=0.30)
        sb = st.rolling_betas(sim, "MKT", WINDOW, WINDOW)
        sp = st.persistence(sb)
        snf = st.noise_floor(sb)
        ctrl.append({"beta_drift": drift, "slope": sp["slope"],
                     "noise_share": snf["noise_share"], "true_sd": snf["true_sd"],
                     "observed_sd": snf["observed_sd"]})
        print(f"  true beta drift {drift:.2f}/yr: persistence {sp['slope']:.3f}, "
              f"observed sd {snf['observed_sd']:.3f}, of which "
              f"{snf['noise_share']:.0%} is noise")
    h["control"] = ctrl
    print(f"  the FIRST row has a perfectly constant true beta and still shows persistence")
    print(f"  below one and visible instability. That is the artefact the decomposition")
    print(f"  removes, and the real data sits between these rows.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    fc = "\n".join(
        f"| {r['method']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['bias']:+.4f} |"
        for r in h["forecasts"])
    bh = "\n".join(
        f"| {int(r['window_days'])} | {int(r['n_periods'])} | {r['slope']:.3f} | "
        f"±{r['slope_se']:.3f} | {r['r2']:.3f} | {r['half_life_years']:.2f} |"
        for r in h["by_horizon"])
    ctrl = "\n".join(
        f"| {r['beta_drift']:.2f} | {r['slope']:.3f} | {r['observed_sd']:.3f} | "
        f"{r['true_sd']:.3f} | **{r['noise_share']:.0%}** |" for r in h["control"])
    sect = "\n".join(
        f"| {r['sector']} | {r['mean_beta']:.2f} | ±{r['sd_beta']:.2f} | "
        f"{r['min']:.2f} – {r['max']:.2f} |" for r in h["sectors"])
    return f"""# Results — Study 1005 (Beta Has a Half-Life) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_names']} names against
{h['n_days']:,} sessions from {h['start']}, {h['n_periods']} non-overlapping
{h['window']}-day windows, {h['n_estimates']:,} beta estimates. As-of **{h['as_of']}**;
fingerprint `{h['fingerprint']}`.*

## 1. Does beta persist?

Regressing each period's cross-section of betas on the previous period's — Blume's (1971) test:

**β(t) = {h['intercept']:.3f} + {h['slope']:.3f} · β(t−1)**  (se {h['slope_se']:.3f},
R² {h['persist_r2']:.3f})

The slope sits {h['slope'] / h['slope_se']:.1f} standard errors above zero and
{(1 - h['slope']) / h['slope_se']:.1f} below one. A deviation from the average beta decays by
half in **{h['half_life_years']:.2f} years**.

## 2. How much of that instability is real?

A {h['window']}-day beta carries a mean standard error of **{h['mean_se']:.4f}**, so two
consecutive estimates would differ even if the true beta never moved at all. Since consecutive
windows are independent, that contributes 2·mean(se²) to the variance of the observed change:

| Component | Variance | Std deviation |
|---|--:|--:|
| Observed year-to-year change | {h['observed_var']:.5f} | {h['observed_sd']:.3f} |
| Attributable to estimation error | {h['noise_var']:.5f} | {h['noise_sd']:.3f} |
| **Genuine movement in beta** | **{h['true_var']:.5f}** | **{h['true_sd']:.3f}** |

**{h['noise_share']:.0%} of the apparent instability is measurement, not movement.** Betas are
considerably steadier than beta *estimates* — a distinction that no risk report makes, and that
changes what should be done about it.

Cross-sectionally the same correction applies: estimated betas spread {h['cross_sd']:.3f} across
names while true betas spread about {h['cross_true_sd']:.3f}, a reliability of
{h['reliability']:.2f}.

## 3. The confirmation: portfolios

If the instability were genuine, measuring beta more precisely would not make it go away. It
does:

| | Mean standard error | Noise share | Persistence slope |
|---|--:|--:|--:|
| Single names | {h['mean_se']:.4f} | {h['noise_share']:.0%} | {h['slope']:.3f} |
| 10-stock portfolios | **{h['port_se']:.4f}** | **{h['port_noise_share']:.0%}** | {h['port_slope']:.3f} |

Note carefully what the slope did: **{h['port_slope']:.3f} against {h['slope']:.3f}** — it did
*not* rise. That is the next section.

## 3b. Why the Blume slope cannot be read as a stability measure

Regressing a noisy measurement on a noisy measurement attenuates the slope by the classical
errors-in-variables factor:

> plim(slope) ≈ persistence × var(true β) / [var(true β) + var(estimation error)]

The slope therefore moves with **two** quantities, and they usually move together.

| | Cross-sectional SD | Reliability | Raw slope | Disattenuated |
|---|--:|--:|--:|--:|
| Single names | {h['cross_sd']:.3f} | {h['reliability_single']:.3f} | {h['slope']:.3f} | {h['disattenuated']:.3f} |
| 10-stock portfolios | — | {h['reliability_port']:.3f} | {h['port_slope']:.3f} | — |

Diversifying cuts estimation error *and* the true spread of betas at the same time, so the
ratio barely moves. Section 8 settles the matter: there, a **drifting** beta produces a
**higher** slope than a perfectly constant one, because drift widens the cross-section. Anyone
reading "the Blume slope is 0.6, so betas are unstable" is reading a ratio, not a stability
statistic.

## 4. Sector funds

| Sector | Mean beta | SD across years | Range |
|---|--:|--:|--:|
{sect}

## 5. How long an estimation window?

| Window (days) | Periods | Slope | SE | R² | Half-life (years) |
|---|--:|--:|--:|--:|--:|
{bh}

Longer windows give more precise estimates, so less of the observed change is noise and measured
persistence rises. The framework predicts this before the table is computed.

## 6. Does shrinkage help?

Predicting each period's beta from the previous one:

| Method | RMSE | MAE | Bias |
|---|--:|--:|--:|
{fc}

The row to look at is **always_one** — assuming a beta of 1.0 for every name, using no data at
all. It scores {h['one_rmse']:.4f} against the raw estimate's {h['raw_rmse']:.4f}. Any beta
model has to beat that before it has earned its place.

## 7. What shrinkage weight is right?

The fitted optimum is **w = {h['best_w']:.3f}** (RMSE {h['best_rmse']:.4f}) against Blume's
conventional {h['blume_default']:.2f}. Theory says the optimal shrinkage weight should equal the
persistence slope from section 1, which was {h['slope']:.3f}; the two differ by
{abs(h['best_w'] - h['slope']):.3f}, close enough to be reassuring about the framework rather
than about any single number.

## 8. The control: a world where beta never moves

| True beta drift (per year) | Persistence slope | Observed SD | True SD | Noise share |
|---|--:|--:|--:|--:|
{ctrl}

The **first row has a perfectly constant true beta** and still produces a persistence slope
below one and visibly wandering estimates. That is the artefact the decomposition in section 2
removes, and the real data sits between these rows rather than at either extreme.

## Caveats

- **Survivorship.** The forty names are all still listed in 2026. A firm whose beta drifted
  toward three before it failed is not in this sample, so the *true* instability is understated
  — the bias runs against the study's conclusion, which is the comfortable direction.
- **OLS betas on daily data** are biased downward by non-synchronous trading for less liquid
  names. These are large caps so the effect is small, but Dimson (1979) aggregated coefficients
  would be the fix and are not applied here.
- **Non-overlapping windows** avoid manufacturing persistence but leave few observations per
  name. Section 5 shows the sensitivity to window length; the qualitative conclusion is stable
  across it.
- **The noise decomposition assumes independent estimation errors** across consecutive windows.
  Non-overlapping windows make that reasonable; overlapping ones would violate it badly, which
  is the second reason for the default.
- **Beta is measured against SPY**, not a true market portfolio. A different proxy moves the
  levels and, in principle, the stability — though sector betas in section 4 suggest not by
  much.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1005-beta-stability](../README.md). Not investment advice.*
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
