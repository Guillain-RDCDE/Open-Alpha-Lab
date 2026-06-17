# Study 257 -- AAII-Sentiment

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The contrarian *direction* is right and stable across every sub-period (panic months +23.8%/yr vs euphoria +8.5%/yr next month), but the headline long-short HAC *t* = **+1.18** and the predictive-regression slope *t* = **-1.54** both fall short of the *t* >= 2 bar; R-squared ~2%. Sentiment is a curated, revised monthly snapshot, not point-in-time weekly. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A long/flat contrarian overlay nets **+9.3%/yr**, *below* buy-and-hold's **+12.5%/yr** -- sitting out euphoric months skips the meat of bull markets. The "edge" lives in a few ex-post-obvious rebounds (2009, 2020) and never pays for the missed upside. |
| **Vintage / survivorship** | ![Named](https://img.shields.io/badge/Vintage--biased-8b949e?style=flat-square) | The curated monthly AAII snapshot is not point-in-time; the live weekly edge a real trader could have captured is weaker still. |

> **In one sentence:** the AAII bull-bear survey leans the contrarian way -- panicked individual investors really are followed by stronger S&P months than euphoric ones -- but the effect never clears *t* >= 2, explains ~2% of next-month variance, and as a timing overlay loses outright to staying invested, making it a *weak, untradeable* folklore gauge.

## The claim

> *Is the AAII bull-bear survey a contrarian timing tool?*

## What we tested

Join the curated monthly AAII bull-bear table (1987-2026) to ^GSPC month-end
closes; at each month-end the observed spread predicts the S&P **price** return
earned the *following* month (one-month execution lag). We run three honest
tests: (a) a **regime sort** of next-month return by prior-spread tercile, (b) a
**predictive HAC regression** of next-month return on the standardised prior
spread, and (c) a **long/flat contrarian overlay** pinned head-to-head against
buy-and-hold net of costs. We add a sub-period breakdown and a deterministic
synthetic positive control that confirms the engine recovers a planted contrarian
edge (slope *t* = -2.99) and reads ~zero on the null (*t* = -0.20).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the survey in plain language, the right-direction regime chart, and why a directionally-true pattern still loses to buy-and-hold |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | synthetic positive control, regime HAC t-stats, predictive HAC regression, timing overlay vs buy-and-hold, sub-period decomposition |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`aaii_sentiment/`](aaii_sentiment/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
