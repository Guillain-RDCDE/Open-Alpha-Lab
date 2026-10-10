# Study 1017 — Do Bull Markets Die of Old Age? 🐂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do older bulls end more often than chance? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The 11 completed US bulls since 1926 (±20% rule, total return) have Weibull shape **k = 1.10** (95% CI 0.72–2.00) — "ageing" on the textbook reading. But random walks with the tape's drift and vol, dated by the same rule, give a median **k = 1.22** (88% of them above 1): real bulls sit at **p = 0.72** against the random walk and **p = 0.24** against GARCH(1,1)-t. Bull age does not predict the next 12 months either (Newey-West **t = −0.99**, simulated p = 0.35). No dating rule (15%, 20%, 25%, asymmetric) gets below p = 0.24. |
| **Tradability** — does de-risking old bulls pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Cutting to 50% equity once a bull outlives the median known at the time (one-month lag, 10 bp one-way, bills as cash), 1940–2018: net excess Sharpe **0.49 vs 0.57** buy-and-hold (Δ −0.08, CI −0.16 to +0.01), $1 → **$773 vs $4,787** — worse even than a constant mix with the same average exposure ($1,079). Out of sample on the daily S&P 500 since 1990: Δ −0.09. |

> **In one sentence:** Bull markets only look like they die of old age because a ±20% ruler builds ageing into any series — random walks dated the same way age at least as fast, a bull's age does not forecast next year's return, and selling half of every old bull ended with 6.2× less money than buy-and-hold.

## What we tested

"This bull market is long in the tooth": the belief — and the business-cycle folklore it borrows
from — that the **older** a bull market is, the **more likely** it is to end (positive duration
dependence), so a prudent investor de-risks late in a long bull. We date every bull and bear on the
Fama-French monthly **total-return** market 1926–2018 with the conventional ±20% peak/trough rule
(plus 15%, 25% and an asymmetric filter in the spirit of Lunde & Timmermann 2004), fit a
right-censored Weibull hazard with cycle-bootstrap CIs, and — the decisive step — run thousands of
random-walk and GARCH(1,1)-t tapes with the tape's own drift and volatility through the **same**
rule, because a filter that needs a 20% move to start and end a bull manufactures ageing on its own.
Then a look-ahead-free predictive regression of next-12-month return and drawdown on bull age (HAC
plus a simulated null), and a de-risk-old-bulls rule with one lag, costs and bills, cross-checked on
the daily S&P 500 price index 1990–2022. Twelve bulls a century is the honest ceiling on power.
**Dedup:** distinct from **81-four-year-itch** (a fixed calendar cycle, not the age of the current
bull), **816-drawdown-duration** (cross-sectional time-underwater of single stocks) and
**333-recovery-speed** (which names lead out of a crash); none dates market-wide bulls or tests
duration dependence against a dating-rule null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | every bull since 1926, a fake market that "ages" too, whether an old bull warns of a bad year, and what selling old bulls cost |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | censored Weibull + bootstrap, life tables against random-walk and GARCH nulls through the same rule, rule sensitivity, real-time predictive regressions with a simulated null, the de-risk backtest and a planted-ageing control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`oldage/`](oldage/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
