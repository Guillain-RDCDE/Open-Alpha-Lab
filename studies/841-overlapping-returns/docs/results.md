# Results — Study 841 (Overlapping-Returns Inflation): the long-horizon predictive-regression trap

*Generated from [`overlapping_returns/`](../overlapping_returns/) on a **deterministic, offline
synthetic world** (seeds 841…, `rho = 0.95`, `delta = −0.9`, 600 monthly rows). This is a
research-method demo, so the world is built on purpose: the **null** has a persistent predictor with
**zero forecasting power** (`beta = 0`), so any long-horizon t-stat or R² above the nominal 5% level
is, by construction, an artefact of the return overlap; the **positive control** plants a genuine
one-period edge (`beta = 0.005`). Real free data can never certify "zero predictability", so there is
no real-tape stamp — the limitation is named on the SIGNAL axis and the study is capped at `NONE`.
Null-world fingerprint `4111f0ae3f09` (600 monthly rows, 1970-01-31 → 2019-12-31).
Run stamped as-of 2026-06-30. Fingerprint `4111f0ae3f09`.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Does overlap inflate inference?" `CONFIRMED`

We forecast the **cumulative return over the next `h` months** from a predictor known today, sampled
**monthly** — the workhorse "long-horizon predictive regression" behind decades of claims that
valuation ratios forecast returns. Consecutive observations overlap by `h−1` months, so the
regression residual is a moving average of order `h−1`; ordinary-least-squares standard errors, which
assume the residuals are serially uncorrelated, are therefore **wrong**. On a world with **no
predictability at all**, the naive OLS t-statistic and R² are grossly inflated, and the distortion
**grows with the horizon**.

- **The naive test rejects a true null far too often, worse and worse as `h` grows.** At `h = 1`
  (no overlap) the naive 5% test rejects **6.2%** of the time — correctly sized. By `h = 12` it
  rejects **56.0%**, and by `h = 24` it rejects **65.6%** — a test that is supposed to fire 5% of the
  time under the null fires two-thirds of the time. The mean |t| under the null climbs from **0.83**
  (its honest value ≈ 0.8) to **3.54** at `h = 24`.
- **The R² is manufactured too.** The mean naive R² under the *pure null* rises from **0.2%** at
  `h = 1` to **3.1%** at `h = 24` — an "explanatory power" that is entirely an overlap artefact.
- **The corrections restore honest inference.** The **Hodrick (1992) "1B"** standard error — which
  moves the summation onto the regressor so the moments are built from non-overlapping one-period
  returns — is well-sized at **every** horizon (5.8–6.2%). **Newey-West** (Bartlett, `lags = h−1`)
  helps enormously (66% → 19% at `h = 24`) but remains somewhat over-sized for this persistent
  regressor and finite sample — the well-known finite-sample weakness of HAC here.

So `NONE` on the signal axis (a synthetic-only method demo — the long-horizon "predictability" is a
pure artefact, and there is no real edge to detect), `MIRAGE` on tradability (an inflated t-stat and
R² are illusions you cannot harvest — the apparent edge evaporates under a correct standard error),
and `CONFIRMED` on the myth-check (yes, overlapping long-horizon returns genuinely and severely
inflate the naive t and R², monotonically in `h`).

## Data stamp

- **Null world** (`beta = 0`, `rho = 0.95`, `delta = −0.9`, monthly return vol ≈ 4.3%): 600 rows,
  1970-01-31 → 2019-12-31, fingerprint `4111f0ae3f09`, seed 841.
- **Positive-control world** (`beta = 0.005`, same generator): a genuine one-period edge (per-period
  R² ≈ 1.6%) — the machinery must still *detect* it under the corrected standard errors.
- **Monte Carlo**: 2,000 independent worlds per horizon (seeds 841…2840), Newey-West `lags = h−1`,
  two-sided 5% critical value 1.96.

## The headline — the naive test's size explodes with the horizon; Hodrick holds

Rejection rate of the two-sided 5% test **under the null** (`beta = 0`) — this is the test's *size*,
which should be ≈ 0.05 for an honest test (2,000 sims per horizon):

| Horizon `h` (months) | Naive OLS | Newey-West (`h−1`) | Hodrick 1B | mean \|t\| naive | mean naive R² |
|---|--:|--:|--:|--:|--:|
| **1** (no overlap) | 0.062 | 0.064 | 0.062 | 0.83 | 0.002 |
| **3** | 0.265 | 0.121 | 0.058 | 1.41 | 0.005 |
| **6** | 0.429 | 0.137 | 0.060 | 1.97 | 0.010 |
| **12** | **0.560** | 0.157 | 0.060 | 2.69 | 0.018 |
| **24** | **0.656** | 0.191 | **0.060** | 3.54 | 0.031 |

The naive OLS test is honestly sized only at `h = 1`, where there is no overlap. From there it
degrades monotonically to a **65.6%** rejection rate of a *true* null. The Hodrick 1B standard error
is essentially exact at every horizon (0.058–0.062); Newey-West removes most, but not all, of the
distortion.

## One world, up close — a 4.8-sigma "discovery" that isn't there

A single null world (seed 841) makes the trap vivid — the same regression, three standard errors:

| Horizon `h` | slope | naive R² | naive t | Newey-West t | Hodrick 1B t |
|---|--:|--:|--:|--:|--:|
| 1 | +0.0042 | 0.7% | +2.01 | +1.95 | +1.94 |
| 6 | +0.0225 | 3.6% | +4.73 | +2.37 | +1.88 |
| **12** | +0.0294 | **3.9%** | **+4.84** | +1.79 | +1.29 |
| 24 | +0.0501 | 6.8% | **+6.47** | +1.62 | +1.27 |

At `h = 12` the naive regression prints **t = +4.84** — a "5-sigma discovery" of predictability that
**does not exist** (this world has `beta = 0`). The Hodrick t is **+1.29** — correctly insignificant.
This is precisely how a driftless valuation ratio can look like a stunning long-horizon forecaster.

## The positive control — the corrections detect a REAL edge (power), they don't just kill the null

The same machinery on worlds with a **genuinely planted** edge (`beta = 0.005`), 2,000 sims — the
rejection rate is now the *power* to detect real predictability:

| Horizon `h` | Naive OLS | Newey-West | Hodrick 1B |
|---|--:|--:|--:|
| 1 | 0.944 | 0.946 | 0.943 |
| 6 | 0.998 | 0.980 | 0.892 |
| 12 | 0.999 | 0.967 | 0.815 |
| 24 | 0.997 | 0.937 | **0.626** |

Both corrected tests have **high power** to find a real edge (Hodrick 0.94 → 0.63 as the correction
grows appropriately conservative at long horizons; Newey-West 0.95 → 0.94). The corrections are not
numb — they *reward genuine predictability* while refusing to be fooled by overlap. (The naive test's
apparent 0.99 "power" is contaminated by its 66% size — it rejects everything, edge or not, so its
rejections are uninformative.)

## Why the verdict is what it is

1. **There is no real edge to detect.** On a world *built* with `beta = 0`, the entire long-horizon
   "predictability" — a naive R² of 3% and t-stats above 6 — is an artefact of the overlap. A
   synthetic-only method demo can never certify a real edge, so **Signal `NONE`**.
2. **Nothing to trade.** An inflated t-statistic and R² are statistical illusions; the apparent
   forecast has zero out-of-sample value and vanishes the instant you use a correct standard error.
   **Tradability `MIRAGE`.**
3. **The pitfall is real and severe.** Overlapping long-horizon returns inflate the naive 5% test to
   a **66%** rejection rate under the null and manufacture a 3% R² from nothing, monotonically in
   `h`; the Hodrick 1B correction restores honest size and the control proves it still detects a real
   edge. **`CONFIRMED`.**

## The honest takeaway

A long-horizon predictive regression is only as trustworthy as its standard error. Overlapping
monthly returns turn a driftless predictor into a t-stat above 6 and an R² of 3% — a "discovery" that
is entirely an artefact of the fact that adjacent 12-month returns share 11 months. The Hodrick
(1992) standard error (or Newey-West with enough lags) sees through it, and the synthetic control
shows the correction still banks a *genuine* edge. `NONE` × `MIRAGE`, pitfall `CONFIRMED`. This is a
method demo on a synthetic world by design — it can never earn `REAL`, which requires a robust
*t* ≥ 2 on a real tape.
