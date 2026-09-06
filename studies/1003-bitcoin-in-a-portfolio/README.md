# Study 1003 — The 1% Allocation ₿

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — what allocation does bitcoin's own history actually imply? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Emphatically, in sample. Over 11.8 years to 2026-06-30, a 60/40 returned 9.31% a year at 10.9% volatility, Sharpe 0.85. The Sharpe-maximising bitcoin sleeve was **16.5%** — not 1%, not 2% — lifting the ratio to 1.18 on a 18.68% return, at the cost of a -31.2% maximum drawdown against the base portfolio's -21.7%. That number is what the historical record says, and the gap between it and every published recommendation is the whole subject of this study. One correction along the way: bitcoin trades 365 days a year and the rest of a portfolio does not, so everything here is aligned to the equity calendar; skipping that step changes bitcoin's annualised volatility by 20% and every ratio built on it. |
| **Tradability** — what expected return must you assume for the published 1-2% to be right? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | So why does nobody recommend 16%? Because nobody believes the realised mean. Bitcoin returned 51% a year over this sample; inverting the optimiser shows what each published allocation is really assuming. A **1% sleeve is optimal if you expect bitcoin to return -5.0% a year**. A 2% sleeve implies -2.0% — essentially nothing. A 5% sleeve implies +8.0%, about what equities are expected to do. These are not cautious readings of the data; they are the answers you get **after discarding it**, which is a defensible position and an entirely different one from what the accompanying backtests imply. The override is justified by the standard error: at 55% volatility, pinning bitcoin's expected return to ±2 percentage points takes **765 years**, and 12 years leaves a standard error of 16% — so the 51% is not knowledge. Out of sample the walk-forward allocator returned 15.11% against the base portfolio's 10.00%, with its chosen weight swinging from 4.7% to 20.0%. The honest form of a bitcoin recommendation states the expected return it assumes; the weight then follows, and can be argued with. |

> **In one sentence:** Bitcoin's own history says hold 16.5%, so the industry's 1-2% is not a reading of the data but an override of it — a 2% sleeve is what you get from assuming bitcoin returns -2.0% a year.

## What we tested

This study set out to show that the data cannot distinguish a 1% bitcoin
allocation from a 5% one. That hypothesis was **rejected**, and the rejection is the study.

Fed bitcoin's actual record on a properly aligned calendar, a Sharpe optimiser recommends a
large double-digit sleeve, and a block bootstrap ranks 5% above 1% in nearly every draw. The
published 1–2% allocations therefore cannot be defended as cautious readings of the evidence:
the evidence points firmly the other way.

**So the question inverts.** `implied_mean_for_weight` recentres bitcoin's returns onto each
candidate expected return — preserving volatility, correlation and path shape — and reports what
each recommended weight assumes. A 2% sleeve is the mean-variance answer if you expect bitcoin
to earn roughly **nothing**; a 1% sleeve assumes a *negative* expected return. Those are priors,
not inferences, and the study shows why holding one is defensible: at bitcoin's volatility the
standard error on its mean return is enormous, so the realised figure is not evidence about the
expected one. Admitting that uncertainty explicitly, by drawing the mean from N(realised, SE) and
re-optimising, still does not reach 1% — no treatment of *sampling* uncertainty does. Only a
different prior does.

**Two corrections that change the numbers.** Bitcoin trades 365 days a year and the rest of a
portfolio does not, so the whole panel is aligned to the equity calendar; and a sleeve left
unrebalanced grows, so a quoted "2% allocation" often was not one.
**Dedup:** distinct from **983-bitcoin-leads-equities** (lead-lag), **989-altcoin-downside-beta**
(downside capture) and **415-gold-allocation** (a different asset with a different problem);
the subject here is whether an allocation recommendation is estimable at all.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why bitcoin's own history recommends far more than 1%, and what the published small allocations are really assuming |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the optimiser inverted into implied expected returns, a block bootstrap, explicit mean uncertainty, the sample-size arithmetic, walk-forward allocation with costs, and a known-truth control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`onepercent/`](onepercent/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
