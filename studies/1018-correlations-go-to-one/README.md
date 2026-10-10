# Study 1018 — Correlations Go to One 🔗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do correlations genuinely rise in crises, beyond the statistical artefact? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes. In a past-only high-volatility regime the 20 stocks' average pairwise correlation goes from **0.21 to 0.47**. The same procedure run on a constant-correlation GARCH fitted to the same tape reports **−0.001**, so the genuine rise is **+0.26** (block-bootstrap 95% CI [+0.15, +0.34]). It is positive under all six regime definitions. The artefact is real where the folklore measures it: selecting months by their own volatility manufactures +0.08 of a +0.32 gap. Correlations do **not** go to one: the highest 63-day reading in 33 years was 0.73. By the Forbes–Rigobon yardstick the whole rise is a louder market factor (betas unchanged), which is interdependence rather than contagion. It is still undiversifiable. Survivor panel of 20 large caps. |
| **Tradability** — does the genuine part hurt a diversified holder? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It hurts, but diversification does not fail. In crises an equal-weight 20-stock book ran **37.1%** volatility, against **26.0%** predicted from calm correlations with crisis volatilities plugged in. That is a **+43%** correlation shortfall (CI [+26%, +54%]). The diversification ratio fell from **2.14 to 1.44**, not to 1. All 10 named episodes under-forecast risk from pre-crisis correlations (median 1.31×). The lesson for a risk model is to stress correlations as well as volatilities. It is not a trade, so it cannot be Investable. |

> **In one sentence:** Correlations genuinely rise in crises but nowhere near one: 0.21 calm to 0.47 in high-volatility markets, with essentially none of it a statistical artefact when the regime is defined from past data. That makes a diversified stock book 43% riskier than volatility alone predicts, without ever making diversification fail.

## What we tested

*"In a crisis all correlations go to one — diversification fails exactly when you need it"*
(stated at full strength in Page & Panariello, *FAJ* 2018, "When Diversification Fails"; tail
evidence in Longin & Solnik 2001 and Ang & Chen 2002). The data are daily adjusted closes of 20
large US stocks from 1990 to 2022, with the S&P 500 as the market state variable (skfolio tapes,
SHA-pinned). We measure crisis-vs-calm average correlation under six regime definitions, four of
them past-only. The **Boyer–Gibson–Loretan / Forbes–Rigobon artefact** is priced exactly, by
running the identical procedure on a constant-correlation GARCH fitted to the same tape; we also
apply the FR adjustment. Exceedance correlations are compared with their Gaussian benchmark, and
we ask what is left for an equal-weight holder (realised crisis vol vs a "vol-only" prediction,
diversification ratio). A synthetic world plants a crisis-correlation jump against a
constant-correlation stochastic-volatility null.
**Dedup:** **578-cross-asset-correlation-regime** asks whether high correlation *predicts returns*;
**1010-correlation-matrix-stability** is about estimation noise in a correlation matrix;
**502-betting-against-correlation** is a cross-sectional sort; **974-diversification-saturation**
prices the *k*-th asset on average. None of them separates the conditioning artefact from a
genuine crisis rise, or measures what that rise costs a diversified holder.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why picking stormy days fools the ruler, what is left once you correct for it, and what it does to a 20-stock portfolio |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | BGL closed form, the constant-correlation GARCH null, block-bootstrap genuine gaps, Forbes–Rigobon vs constant beta, exceedance curves, diversification ratios, episodes, synthetic control |

Reproducible headline run (fingerprinted, as-of 2022-11-30): [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`goestoone/`](goestoone/). **Not investment advice**: research & education. See [LICENSE](../../LICENSE).*
