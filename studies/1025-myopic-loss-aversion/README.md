# Study 1025 — Don't Look 🙈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the evaluation horizon flip a loss-averse investor from bonds to stocks, near one year? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The flip is real: with Tversky-Kahneman preferences a monthly evaluator prefers bonds (block-bootstrap p < 0.001) and a ten-year evaluator prefers stocks (p 0.044, only just). But the break-even horizon on 1926-2018 is **30 months**, with a 95% interval of **2.6 months to 9.4 years**, so this tape cannot pin Benartzi and Thaler's one year. And the one year turns out to be the *i.i.d.* answer: drawing the same months independently gives 11.6 months, while the real sequence of returns gives 30 (23 on their own 1926-1990 window). |
| **Tradability** — what does the myopic behaviour actually cost? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Looking less changes what you see, not what you earn. An investor who moves to bills after seeing a loss and back after a gain (one lag, 5 bp) gives up **11.5% a year checking daily** (HAC t 5.8) and 7.8% checking weekly. Checking monthly, quarterly or yearly, the cost is about 1% a year, not significant, and the switcher's excess Sharpe is no worse than buy-and-hold's. The advice protects against one expensive habit, reacting to daily noise. It adds no edge. |

> **In one sentence:** A loss-averse investor does come to prefer stocks once they look rarely enough, but the break-even is 30 months (2.5 years) with a 95% interval of 2.6 months to 113 months (9.4 years) — the famous one year is the i.i.d. answer — and looking less is a seatbelt, not an edge: it only pays against daily and weekly checking.

## What we tested

Benartzi & Thaler (1995, *Quarterly Journal of Economics*) explained the equity premium
with two human quirks. Losses hurt about twice as much as equal gains (loss aversion). People
also *count* their gains and losses often (myopia). Give an investor Tversky-Kahneman (1992)
preferences, let them look once a year, and they are indifferent between stocks and bonds. On
this reading the premium is the price of looking too often. Laboratory experiments (Thaler,
Tversky, Kahneman & Schwartz 1997; Gneezy & Potters 1997) found that subjects given less
frequent feedback took more risk. The folk version is advice: *check your portfolio less and
you'll stay invested.*

**Setup.** We rebuilt the calculation on frozen, SHA-pinned tapes. Stocks are the Fama-French
total-return market (1926-2018), bills are the one-month T-bill, and bonds are a 20-year par bond
built from Moody's AAA yields. The daily S&P 500 covers the day-to-week end of the curve, and
core CPI gives a real-terms check. Each horizon from one day to ten years gets a loss
probability and a cumulative-prospect-theory value. The break-even horizon comes with a circular
block bootstrap of the joint monthly returns, and is swept across λ, α, real versus nominal,
pre- versus post-1970, bond maturity and three estimators. A pre-registered myopic switcher
prices the behaviour the advice is meant to prevent.
**Dedup:** distinct from **1007-time-diversification** (how return dispersion scales with
horizon; no preferences), **151-stocks-for-long-run** (do stocks always win over the long run),
**1008-start-date-lottery** (start-date path dependence), **1002-best-days-missed** (missing
specific days) and **332-downside-beta** (a cross-sectional downside-risk premium). This study
covers *preferences over evaluation frequency* and the measured cost of acting on them.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how often red shows up on your screen, why it hurts, and what panicking about it actually costs |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | CPT on empirical horizon distributions, the break-even and its block-bootstrap interval, three estimators, λ/α/real/sub-period sweeps, the switcher with HAC and paired-bootstrap inference, the planted-premium control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dontlook/`](dontlook/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
