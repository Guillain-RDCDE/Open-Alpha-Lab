# Study 1010 — Mostly Noise 🧱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how much of an estimated correlation matrix is signal? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | Most of it, and the amount is calculable before any data is collected. Estimating a 50×50 covariance matrix means fitting **1,275 parameters** from 252 observations per asset; at q = N/T = 0.198, Marchenko-Pastur puts the pure-noise eigenvalue band at [0.308, 2.089]. On the real matrix, **31 of 50 eigenvalues fall inside it** — 62% of the spectrum, carrying 48% of the total variance and indistinguishable from a matrix of random numbers. Only **5 escape upward**, and the largest of those is simply the market: it alone accounts for 20% of the variance. A further 14 fall *below* the band, which is not signal either — those are the near-degenerate directions that make the matrix ill-conditioned, and an optimiser is drawn to precisely them because a tiny variance estimate looks like a free lunch. Lengthening the estimation window does not fix this: at 2520 days the informative count is still only 4. More data narrows the band; it does not manufacture factors that were never there. The synthetic control confirms the machinery rather than the story — with **independent** assets by construction, 99% of the spectrum lands inside the band as it must, and planting 3 factors puts 3.0 eigenvalues above it. Persistence tells the same story from another direction: consecutive windows' pairwise correlations agree at 0.64, which sounds reassuring until the market factor is removed, after which the residual agreement is **0.62**. Almost all the apparent stability is the single fact that stocks move together. |
| **Tradability** — does cleaning it produce a portfolio that is actually better out of sample? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Cleaning helps, mostly by stopping the optimiser lying to itself. Fed the raw sample matrix, a minimum-variance portfolio forecast 9.2% volatility and realised 12.7% — a calibration ratio of **1.46**, because the optimiser selects precisely the directions where noise made the variance look smallest. ledoit_wolf brought that to 1.36 while realising 12.5%. The honest benchmark is the diagonal matrix — throwing every correlation away — which realised 13.4% at a calibration of 4.24; the cleaned estimators earn their place against it. Ledoit-Wolf shrank toward its target with an average intensity of **0.37**, i.e. the estimator itself concluded that roughly that fraction of the sample matrix was not worth keeping. Two practical readings. First, the estimation window is a real choice with a computable cost: at 63 days the band swallows 90% of the spectrum against 14% at 2520. Second, a long-only constraint removes much of the damage on its own — gross leverage falls from 3.0× to 1.0× — so a good deal of what cleaning buys is available for free to anyone who cannot short anyway. |

> **In one sentence:** 62% of a 50-asset correlation matrix estimated from 252 days is inside the Marchenko-Pastur noise band, and the optimiser's own risk forecast is out by a factor of 1.46 until you clean it.

## What we tested

A covariance matrix is the input every optimiser depends on and the one nobody
puts a standard error on. This study puts one on it, and most of the answer is arithmetic
available before any data is collected.

**The counting argument and the noise band.** Fifty assets means 1,275 free parameters.
Marchenko-Pastur (1967) gives the eigenvalue distribution of a *pure-noise* sample correlation
matrix exactly, as a function of q = N/T alone: λ± = (1 ± √q)². Every sample eigenvalue inside
that band is consistent with the assets being completely unrelated. The study measures how much
of a real equity spectrum falls inside it — most of it — and how that changes with the
estimation window.

**A control that calibrates the tool before it is used.** A synthetic world with a *known* number
of factors: at zero factors the assets are independent and the spectrum must sit inside the
band; planting factors must push the right number above it. Both hold.

**Persistence, with the obvious confound removed.** Consecutive windows' pairwise correlations
agree well, which sounds reassuring and mostly restates that stocks move together. Stripping the
first principal component collapses the agreement — the informative measurement, and one rarely
made.

**The practical test scores calibration, not just risk.** Minimum-variance portfolios from the
raw, RMT-cleaned, Ledoit-Wolf and *diagonal* matrices are compared on realised volatility against
their own forecast. An optimiser fed a noisy matrix underestimates the risk of the portfolio it
picks, because it selects the directions where noise flattered the variance. The diagonal
baseline — throwing every correlation away — is a hard benchmark, and a final section shows how
much of the benefit a long-only constraint delivers for free.
**Dedup:** distinct from **1005-beta-stability** (a single exposure's shelf life),
**1004-how-many-stocks** (portfolio size) and **1003-bitcoin-in-a-portfolio** (estimating a
mean); the subject here is the second moment as an object.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how much of the correlation matrix behind every risk model is actually measurable, and how to know in advance |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Marchenko-Pastur band against the empirical spectrum, a known-factor control, persistence with the market stripped out, and a calibration-scored estimator race |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`corrnoise/`](corrnoise/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
