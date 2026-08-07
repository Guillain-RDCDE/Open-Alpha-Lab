# Results — Study 835 (Spurious Regression): trending series manufacture false significance

*Generated from [`spurious_regression/`](../spurious_regression/) on **deterministic, offline
synthetic worlds** (base seed 835). This is a research-method demo (Granger & Newbold 1974), so the
data is built on purpose: the two series in each pair are drawn **independent**, so any "significant"
relation the level regression prints is spurious by construction. A research-method demo cannot
certify "no relation" from real prices, so there is no real-tape stamp; the data-availability limit is
named on the SIGNAL axis and the study is capped at `NONE`. Simulation Fingerprint `73e2821b184c`
(base_seed 835, 5,000 pairs × 250 obs, driftless), as-of 2026-06-30.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Do trending series manufacture false significance?" `CONFIRMED`

We simulate 5,000 pairs of **independent** Gaussian random walks (each `I(1)`, no relation between
them) and regress `y` on `x` in **levels**. A correctly sized 5% *t*-test should reject "no relation"
about 5% of the time. The textbook OLS *t*-stat instead rejects **85.0%** of the time — a **17×
oversized** test — with a mean `|t|` of **8.99** and a mean R² of **0.24** (four pairs in ten clear
R² > 0.25), all from pure nonstationarity. **First-differencing** (regress `Δy` on `Δx`) collapses the
rejection rate back to **5.3%** and R² to **0.004** — the fix works. The same OLS on two *stationary*
series is correctly sized (**5.1%**), proving the inflation is a property of the **unit root**, not of
OLS. And the Engle-Granger **cointegration test** rejects no-cointegration on **5.0%** of the
independent walks (correctly: nothing there) but **100%** of a genuinely cointegrated pair — the
machinery tells a real long-run relation from a spurious one.

So `NONE` on the signal axis (a constructed null — there is nothing real to find, and no real tape a
method demo could stamp), `MIRAGE` on tradability (the spurious spread is itself a random walk, not
mean-reverting; a costed pairs trade earns no edge distinguishable from zero and bleeds costs), and
`CONFIRMED` on the myth-check (yes — regressing trending, nonstationary series manufactures grossly
inflated significance, and adding a shared trend makes it worse still).

## The headline — level OLS on two independent random walks over-rejects 17×

5,000 pairs, 250 observations each, driftless independent random walks:

| specification | reject \|t\|>1.96 | mean \|t\| | median \|t\| | mean R² | share R²>0.25 |
|---|--:|--:|--:|--:|--:|
| **levels** (`y` on `x`) | **0.850** | **8.99** | 7.06 | **0.241** | 0.398 |
| **first differences** (`Δy` on `Δx`) — the fix | **0.053** | 0.80 | 0.67 | 0.004 | 0.000 |

The nominal test size is **0.05**; the level regression rejects at **0.850** (Wilson 95% CI
**[0.840, 0.860]**) — a **17.0×** oversized test. First-differencing restores the correct size (0.053,
essentially nominal) and the R² vanishes (0.004). The "relation" was an artefact of the shared
`I(1)` structure, not a signal.

## Trending makes it worse — the claim in one line

Add a common deterministic drift (0.15/step) to both independent walks — the "trending series" case:

| specification | reject \|t\|>1.96 | mean \|t\| | mean R² | share R²>0.25 |
|---|--:|--:|--:|--:|
| **trending levels** | **0.981** | **28.48** | **0.662** | 0.899 |

With a shared trend the level regression rejects **98.1%** of the time, the mean `|t|` balloons to
**28.5**, and the mean R² is **0.66** — nine pairs in ten show R² > 0.25. Two unrelated series that
merely *drift together* look almost deterministically related. **Do trending series manufacture false
significance? Emphatically yes.**

## More data makes it *worse*, not better — the signature of the pitfall

Fresh independent-walk panels at each sample size (4,000 pairs each):

| n_obs | level reject | level mean \|t\| | level mean R² | diff reject |
|---|--:|--:|--:|--:|
| 50 | 0.679 | 3.99 | 0.243 | 0.059 |
| 125 | 0.787 | 6.20 | 0.236 | 0.052 |
| 250 | 0.847 | 8.99 | 0.241 | 0.052 |
| 500 | 0.895 | 12.81 | 0.241 | 0.050 |
| 1000 | 0.926 | 17.99 | 0.240 | 0.045 |

The spurious `|t|` scales with **√T**, so *more data makes the level test reject more* — the mean
`|t|` climbs from 4.0 (n=50) to 18.0 (n=1000) and the rejection rate marches toward 1. The differenced
regression stays pinned at ~5% throughout. This is the exact opposite of the usual "more data → sharper
inference" intuition, and it is why a big-`n`, high-`t`, high-R² regression on levels is *no* comfort.

## Specificity — the same OLS on stationary series is fine

Level OLS on two **independent stationary** (white-noise) series, 5,000 pairs, 250 obs:

| specification | reject \|t\|>1.96 | mean \|t\| | mean R² |
|---|--:|--:|--:|
| **stationary levels** | **0.051** | 0.80 | 0.004 |

Correctly sized at **5.1%**. The over-rejection is *not* a defect of OLS — it is a property of running
the regression on **nonstationary** (unit-root) data. Difference first, or work with stationary
variables, and the *t*-stat means what it says.

## The other fix — cointegration tells spurious from genuine (positive control)

Engle-Granger two-step cointegration test (`statsmodels.tsa.stattools.coint`), 300 pairs each,
reject the no-cointegration null at 5%:

| world | reject no-coint | median p-value | reading |
|---|--:|--:|---|
| **independent random walks** | **0.050** | 0.495 | correctly finds **nothing** |
| **genuinely cointegrated pair** | **1.000** | 0.000 | correctly finds the **real** relation |

On the independent walks the test fails to reject (~5%, the nominal size): no genuine long-run
relation, exactly right. On a pair sharing a common stochastic trend (`y − βx` stationary) it rejects
every time. The cointegration test is the discipline that separates a real relationship in levels from
a spurious one — the positive control that proves the machinery is unbiased. *(A faithful-engine /
power check only — never cited in support of a real-tape stamp, of which this synthetic study has
none.)*

## The timer — can you trade the spurious "relationship"? (no look-ahead)

A quant who trusts the level regression bets the spread `y − βx` mean-reverts. Using a **trailing**
hedge ratio and z-score (known at `t−1`, no look-ahead), contrarian on the residual, 3,000 pairs:

| one-way cost | gross/day | net/day | *t* (net) | Sharpe (net) | ~ann. |
|---|--:|--:|--:|--:|--:|
| **0 bps** | −27.46 | −27.46 | −1.23 | −1.43 | −69.2%/yr |
| **1 bp** | −27.46 | −27.99 | −1.25 | −1.45 | −70.5%/yr |
| **5 bps** | −27.46 | −29.59 | −1.33 | −1.54 | −74.6%/yr |

The gross edge is **indistinguishable from zero** (|*t*| = 1.23, well below 2): the spurious spread is
itself a random walk, so there is no mean reversion to harvest — the contrarian trade neither wins nor
loses in expectation, and every cost level only pushes it further underwater. **Mirage.** *(The
"bps/day" are PnL in normalised spread-σ units × 10⁴; the load-bearing numbers are the near-zero
gross *t* and the negative Sharpe.)*

## Why the verdict is what it is

1. **There is nothing real to find.** The two series are drawn independent; the level regression's
   "significance" is a manufactured artefact of the unit root, and no real tape exists that a
   method demo could stamp. **Signal `NONE`.**
2. **Nothing to trade.** The spurious spread is a random walk, not a mean-reverting one; a costed
   pairs trade earns no edge distinguishable from zero and loses net of any friction.
   **Tradability `MIRAGE`.**
3. **The myth is confirmed.** Regressing trending, nonstationary series manufactures grossly inflated
   *t*-stats and R² (85% false rejection on driftless walks, 98% with a shared trend), and the
   inflation *grows with the sample*. First-differencing and a cointegration test both see through it.
   **`CONFIRMED`.**

## The honest takeaway

A high *t*-stat and a big R² are worthless on nonstationary data. Two independent random walks — or,
worse, two series that merely trend together — will hand you a textbook "significant relation" 85–98%
of the time, and adding data only sharpens the illusion. The cure is old and simple: difference to
stationarity (or test for cointegration) *before* you believe a levels regression. This is a method
demo on synthetic worlds by design — it can never earn `REAL`, which requires a robust *t* ≥ 2 on a
real tape. Every number here is reproduced by [`examples/verify.py`](../examples/verify.py).
