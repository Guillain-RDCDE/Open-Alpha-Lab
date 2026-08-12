# Study 877 — GDPNow Revisions 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the daily GDPNow revision predict forward SPY? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The predictive slope of 1-day forward SPY on the nowcast revision is **insignificant** (NW *t* = **−1.02**, *R²* = **0.043%**), **flips sign** under a one-day execution lag and across the two eras (*t* = +0.19 / −1.08), and sits inside a permutation placebo (*p* = 0.32). The only significant piece — big **up**-revisions preceding next-day **weakness** (−18.71 bps, *t* = −2.74) — is **wrong-signed vs the claim** *and* flips to +11.24 bps once you can't trade the release-day close. Big **down**-revisions are flat (*t* = +0.01), so "downward revisions precede weakness" fails outright. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A timer that holds SPY for one day after an up-revision earns Sharpe **0.07** at a generous 1 bp cost — a tenth of buy-and-hold's **0.80** on the same dates — and goes **negative** (−0.34) at 5 bps. Sitting out the drift waiting for up-revisions costs more than the revision times. |
| **Priced-in-already?** | ![Yes](https://img.shields.io/badge/Priced_in%3F-Yes-8b949e?style=flat-square) | GDPNow updates post *intraday, right after* the same releases the tape trades in the first minutes; the daily revision is a **coincident restatement** of absorbed news, not a forecast of tomorrow. |

> **In one sentence:** "trade SPY on the Atlanta Fed's daily GDPNow revision" sounds like a free real-time growth surprise, but on 2,042 genuine within-quarter revisions (2011–2026) the predictive slope is insignificant (*t* = −1.02, *R²* = 0.04%), sign-flips under a one-day lag and across eras, and the one significant number is *wrong-signed and fragile* — so the nowcast revision is real news but a **None** to predict and a **Mirage** to trade.

## What we tested

The **Atlanta Fed GDPNow** nowcast is revised almost daily as data arrives. We take its full daily forecast history from the public workbook (`TrackingDeepArchives` + `TrackingArchives`, **2,102** forecasts over **60** quarters, 2011–2026), form the **within-quarter day-over-day revision** (the real-time growth surprise), and regress the 1- and 5-trading-day forward **SPY** total-return on it — Newey-West *t*, *R²*, a top/bottom-decile conditional test, a two-era cut (split 2019), a 5,000-draw permutation placebo, and a costed up-revision timer. We deliberately act at the **release-day close** (the most generous execution) and show the result does not survive a one-day lag. A 20-seed synthetic control (planted revision→return edge, *t* = +6.65; null fires 0/20) confirms the machinery is unbiased.

**Dedup.** Distinct from [387-economic-surprise-index](../../387-economic-surprise-index/) (a Citi-style multi-series beat/miss composite vs a *trailing-average* consensus, sampled monthly), [384-ism-pmi-regime](../../384-ism-pmi-regime/) (a PMI *level/regime*), [268-sahm-rule](../../268-sahm-rule/) (an unemployment-rate recession *trigger*), and [385-jobless-claims-momentum](../../385-jobless-claims-momentum/) (momentum in a single labour series): this study is the **daily revision of one broad GDP nowcast**, a different input and cadence.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a nowcast revision is, why "up-revision = buy" feels right, and why a number the market already traded minutes earlier can't forecast tomorrow — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive HAC regression, the decile conditional, execution-lag robustness, the two-era cut, the permutation placebo, an up-revision-timer-vs-buy-and-hold race, and a 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gdpnow/`](gdpnow/). Nowcast = Atlanta Fed GDPNow top-line real-GDP estimate (public workbook); SPY is total-return, price-only. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
