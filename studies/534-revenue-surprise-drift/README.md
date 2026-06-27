# Study 534 — Revenue-Surprise-Drift 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the revenue-surprise drift exist? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Sorted on the **standardized unexpected revenue (SUR)**, the top-minus-bottom drift **never clears *t* = 2** at any horizon (max **t = +0.59** at 1 day; the 5- and 60-day spreads are **negative**). The cross-section is non-monotone (the 2nd SUR quintile is the highest), the sign **flips with the bucket count**, and a within-quarter block placebo sits at **p = 0.582**. The faithful synthetic control (zero edge → t = 0.03, planted edge → t = 9.55) proves the detector works — there is simply no revenue drift to find on this **30-name survivor** basket. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of one-way costs × per-event turnover (+ borrow), every horizon is essentially zero-or-negative (best is **+0.02%** at 20 days). There is no gross edge to erode and nothing to deploy. |
| **"Adds info beyond EPS"?** (Jegadeesh-Livnat) | ![Busted](https://img.shields.io/badge/Adds_info_beyond_EPS%3F-Busted-8b949e?style=flat-square) | Match each revenue event to the same name's reported **EPS surprise** and run the SUR sort **within EPS-sign strata**: the long-short is **−0.30%** inside beats, **−0.67%** inside misses, **pooled −0.19% at t = −0.40**. The specific Jegadeesh-Livnat *incremental*-information claim does **not** replicate on this conservative large-cap survivor basket. |

> **In one sentence:** Jegadeesh-Livnat (2006) say the post-earnings drift follows the *revenue* surprise and adds information beyond EPS — but rebuilt honestly on a 592-event, 30-name large-cap survivor basket with standardized unexpected revenue from EDGAR, the drift **never clears t = 2** (max t = 0.59), flips sign across horizons and bucket counts, dies in a block placebo (p = 0.58), and shows **no** incremental drift once the EPS surprise's sign is controlled (pooled t = −0.40) — a clean None × Mirage on a conservative basket, with the EPS sibling [363](../363-pead-drift) as the contrast that *does* survive.

## What we tested

We rebuild Jegadeesh-Livnat's revenue-surprise drift as a clean event study on a fixed **30-name large-cap basket**: per name we pull every quarterly **revenue** figure from **EDGAR**'s frame-tagged calendar quarters (with the 10-Q/10-K filing date), form the **standardized unexpected revenue** (SUR = the seasonal-random-walk surprise `Rev_q − Rev_{q−4}` scaled by the trailing volatility of those seasonal differences), sort events into SUR quintiles, and measure the forward **1 / 5 / 20 / 60-day** drift of a top-minus-bottom long-short — entering the close **one day after the filing is public** (no look-ahead). The Signal axis tests the long-short against zero with a one-sample *t*, a 20,000-draw SUR-label placebo, and a within-quarter block placebo; Tradability charges one-way costs × per-event turnover plus short-leg borrow. The third axis tests Jegadeesh-Livnat's *incremental* claim directly — the SUR sort **within EPS-sign strata**, using the reported EPS surprise from study [363](../363-pead-drift). A deterministic synthetic control with a *planted* drift confirms the engine is faithful and well-powered (so the flat result is a real absence of signal, not a broken detector). Survivorship (the basket is names still trading in 2026) and the thinner EDGAR sample (2013→2026) are named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "revenue surprise" is, why sales might drift differently from earnings, and why — on big liquid names — the revenue drift simply isn't there, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | SUR quintile long-short, forward 1/5/20/60-day drift, a one-sample *t* + label-shuffle & block placebo nulls, costs × turnover, the within-EPS-strata incremental test, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`revenue_drift/`](revenue_drift/). Surprise = standardized unexpected revenue (SUR) from EDGAR frame-tagged quarterly revenue. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
