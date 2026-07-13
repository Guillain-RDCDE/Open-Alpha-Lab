# Study 761 — Hotel-RevPAR 🏨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does RevPAR momentum *lead* hotel REITs? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The claimed positive lead is absent: the HAC *t* of forward returns on RevPAR YoY momentum is ~0 at 1–3 months (**−0.9 / −1.4**) and turns significantly **negative** at 6–12 months (**−2.3 / −2.8**) — a high RevPAR YoY *precedes lower* returns (12m: **+7.8%** after a boom vs **+16.5%** after a bust). The only significant relationship runs the **wrong way** for the claim. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | A long-when-RevPAR-growing overlay *does* top buy-and-hold in sample (net Sharpe **0.45 vs 0.31**; **9.0×** vs **4.2×**) — but on **8 switches over 28 years**, i.e. ~2 crash-dodge regime calls (2008–09, 2020), with **no positive predictive content** and **lagging execution**. Thin and event-driven, not a repeatable edge. |
| **Leading indicator?** | ![Busted](https://img.shields.io/badge/Leading_indicator%3F-Busted-8b949e?style=flat-square) | Lead-lag correlation peaks at **L = −6**: the hotel stock moves ~6 months *before* RevPAR turns. RevPAR is a coincident-to-lagging confirmation of the travel cycle, not the early read the folklore sells. |

> **In one sentence:** "ride the travel cycle — when hotel RevPAR momentum turns up, buy hotel REITs" gets the arrow backwards: on a cited RevPAR proxy vs HST and a lodging-REIT basket over 28 years, the equity *leads* the RevPAR print by ~6 months, a booming RevPAR YoY is a *late-cycle* tell that precedes **lower** forward returns (HAC *t* = −2.8), and the one overlay that beats buy-and-hold does so only by dodging two crashes on a handful of lagged regime calls.

## What we tested

Hotel **RevPAR** (Revenue Per Available Room) is the lodging industry's headline demand gauge, published monthly by **[STR / CoStar](https://str.com/)** and tracked on every hotel desk; the folklore is that *RevPAR momentum leads the sector* — accelerating travel demand that hotel REITs haven't fully priced. STR's monthly tape is proprietary, so we build a small, **clearly-labelled approximate reconstruction** of U.S. monthly RevPAR anchored to STR/CoStar-reported annual figures (COVID months set to the reported national path), align it to **HST** (the flagship lodging REIT) and an equal-weight lodging-REIT basket via yfinance with a strict one-month STR release lag, and ask — with a Welch *t*, a Newey-West HAC predictive regression, a placebo null, and a lead-lag cross-correlation — whether RevPAR YoY momentum *leads* the hotel tape or merely *echoes* what the stock already discounted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what RevPAR is, why "buy the travel upcycle" feels right, and why the hotel stock has already moved by the time the RevPAR headline prints — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | YoY momentum vs forward returns, a Welch *t* + HAC predictive regression + placebo null, a lead-lag cross-correlation, a timing-vs-buy-and-hold Sharpe race, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hotel_revpar/`](hotel_revpar/). RevPAR is an explicit **proxy** (an approximate monthly reconstruction anchored to STR/CoStar annual figures), not STR's licensed tape. Hotel prices are total-return (`auto_adjust`). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
