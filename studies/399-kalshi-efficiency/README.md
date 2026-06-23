# Study 399 — Kalshi-Efficiency 🗳️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real, harvestable mispricing? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A **methods demo, not a backtest** — Kalshi's resolved history isn't free, so the "real" tape is an explicit *illustrative* book whose favourite–longshot bias is **planted by construction**. That can never back a `REAL` stamp (which needs a robust *t* ≥ 2 on a **real** tape). The calibration engine is validated; **no real edge is claimed.** |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even a *planted* 4-cent favourite–longshot bias nets only **~2.4¢** after a 1¢ fee, and a realistic **2–3¢** of fee + bid/ask drops the *t* from 3.3 through the bar to **0.60**. EV per share is `outcome − price − fee`; the documented mispricing is exactly the size the exchange's costs eat. |
| **Free edge in the mispricing?** | ![Busted](https://img.shields.io/badge/Free_edge%3F-Busted-8b949e?style=flat-square) | The 92% win-rate that makes the pitch glow is **base rate × favourites-win**, not an edge — the control's calibrated `edge=0` book wins **88%** of the time yet nets *negative*. Same high-confidence / no-harvestable-edge shape as [Study 351](../351-btc-5m-polymarket-momentum/). |

> **In one sentence:** event-contract prices are probabilities, and a finite book of binaries *will* wiggle off the calibration diagonal — but the documented favourite–longshot bias is only a few cents, and on a transparent illustrative book (Kalshi history isn't free) that few-cent gap is statistically slippery and entirely eaten by the exchange's per-contract fee and spread, so the "edge in the mispricing" is a base-rate-and-fee mirage, not money.

## What we tested

Kalshi sells **binary event contracts**: each pays $1 on YES and $0 on NO, so the price in cents *is* the crowd's probability. The believer's pitch is **price vs frequency** — if some bucket of contracts is mispriced, buy the cheap longshots short / the rich favourites long and harvest the gap. Kalshi's resolved-contract history is **not free**, so we run a **transparent, clearly-labelled illustrative book** of 4,000 resolved binaries — a *methods demo*, not a live backtest — carrying a mild favourite–longshot tilt so the calibration curve has something to show. We draw the reliability curve, read the miscalibration off a Brier-score decomposition, build the long-favourite / short-longshot spread net of a one-way fee, and judge it with a Welch *t* and a within-bucket randomization null. A deterministic synthetic control plants a *known* edge: with `edge=0` the book is perfectly calibrated and the fee makes a fair book a pure cost (no false positive); a large edge lights up — proving the engine is faithful **and** that a high win-rate alone is never an edge. (Same family as [Study 351](../351-btc-5m-polymarket-momentum/) and the research-method demos 343–350.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an event contract is, why a price is a probability, what "calibrated" means, and why a 92%-win longshot/favourite book still isn't free money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the reliability curve, Murphy's Brier decomposition, the favourite/longshot spread net of fees, a Welch *t* + within-bucket randomization null, and a synthetic faithful-engine / fee-bites control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`kalshi_efficiency/`](kalshi_efficiency/). The "real" tape is an explicit **illustration** (a constructed event-contract book), not live Kalshi prices — a methods demo. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
