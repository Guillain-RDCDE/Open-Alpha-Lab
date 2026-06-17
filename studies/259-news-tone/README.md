# Study 259 -- News-Tone

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The headline next-day regression *t* = **+5.74** is a **hindsight artifact**: demeaning the forward return within each calendar month drives the slope to **exactly zero** (*t* = +0.00), and the only information-respecting version -- last month's tone, knowable in advance -- reads *t* = **+0.97**, Sharpe 0.19. No day-ahead predictability survives. The curated proxy can only *over*-state the effect. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The only "profitable" version (+35%/yr, *t* = 9.5) uses each month's mood *with hindsight* -- information no trader had in real time. Costs are trivial (the proxy flips a few times a year); it is **hindsight, not friction, that kills it**. The honest version is indistinguishable from buy-and-hold. |
| **Look-ahead / proxy bias** | ![Named](https://img.shields.io/badge/Hindsight--curated-8b949e?style=flat-square) | The tone proxy is a hand-curated, in-sample monthly read of known macro mood swings; all positive results are upper bounds, and even those dissolve under the within-month and prior-month placebos. |

> **In one sentence:** a curated daily news-tone proxy appears to forecast the next day's S&P 500 with a glittering *t* = 9.5 -- until you demean within the month (slope → exactly zero) or feed a trader only last month's tone (Sharpe 0.19), revealing the entire "edge" as the month-level drift a hindsight-labelled index already knew, making this a textbook **None / Mirage**.

## The claim

> *Does aggregate news tone move the next day's tape?*

## What we tested

We build a curated monthly **news-tone proxy** (a z-scored read of the macro
headline mood -- deeply negative in 2008 and March-2020, positive in the 2017
and 2021 melt-ups), forward-fill it to business days, and match it to ^GSPC
close-to-close returns. The honest question: does today's tone forecast
*tomorrow's* return? We report (a) the mechanical same-day correlation, (b) the
naive next-day sign-timing strategy that *looks* spectacular, and then the two
checks that demolish it: **within-month demeaning** (removes the month drift) and
**prior-month tone** (removes the hindsight). A turnover/borrow cost sweep and a
deterministic synthetic positive control (which confirms the engine reads ~zero
on noise and lights up on a planted link) round it out.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the seductive same-day chart, why the naive +35%/yr strategy is a warning sign, and the two plain-language tests that make the edge evaporate |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC predictive regressions, the within-month placebo, the lagged-tone tradability test, cost sweep, sub-periods, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`news_tone/`](news_tone/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
