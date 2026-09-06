# Study 984 — A Dollar Off 💸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the price drop by the full dividend on the ex-day? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Across **1025 ex-dates** on 12 mega-cap payers (2005-01-06 → 2026-06-15), a dollar of dividend took **1.003** of a dollar out of the price once that day's market move is removed (HC1 *t* against 1.0 = **+0.14**; against zero +53.13). The four defensible summaries of the same events disagree by more than the effect anyone is arguing about: the mean per-event ratio is +1.13, the median 1.13, total-drop-over-total-dividend 1.133, the regression slope 1.003. That spread is not sloppiness — the per-event ratio divides a 1.4% daily move by a 0.85% dividend, so 29% of individual events land outside the range [0, 2] entirely. Elton and Gruber's 0.778 sits outside the bootstrap interval [1.052, 1.218]. |
| **Tradability** — is there a dividend-capture trade in the gap? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Buying at the cum close and selling at the ex close, keeping the dividend, earned **-11.9 bps** gross per event and **-15.9 bps** after 2 bps a side (*t* = -4.04, 45% of events profitable, dispersion 126 bps). The trade breaks even at -5.9 bps of round-trip cost, and at the 15% qualified-dividend rate a taxable holder nets -31.8 bps. The edge, if it is one, is smaller than the noise on a single event by a factor of 8. |

> **In one sentence:** A dollar of dividend removes 1.00 of a dollar from the price — but the four standard ways of computing that number span 0.13, which is larger than the effect the literature has spent fifty years arguing about.

## What we tested

A share about to pay a dollar must be worth a dollar more than the same share
after it pays, so on the ex-dividend morning the price should fall by the dividend. Elton and
Gruber measured it in 1970, found **77.8 cents**, and read the shortfall as a tax-clientele
effect — launching fifty years of papers.

This study measures it again on several hundred ex-dates across twelve mega-cap payers since 2005, using a
**dividend-unadjusted** tape (the desk's usual total-return closes would return 1.000 by
construction, which is why this study keeps its own cache). And it spends as much effort on
whether the number is *measurable* as on what it is. Three things compound: a quarterly dividend
is about 0.7% of the price while a single day's move is about 1.2%, so the per-event ratio
divides noise by something small and has tails heavy enough that a third of individual events
score outside [0, 2]; the market moves on the ex-day too, and the correction for it (a strictly
backward-looking beta) changes the answer materially; and the four defensible ways to summarise
the same events — mean of ratios, median, total-drop-over-total-dividend, and the regression
slope — **disagree by more than the effect the literature is arguing about**. A synthetic
generator with a *planted* drop fraction then settles which estimator to trust, and the
dividend-capture trade prices the whole thing in basis points against costs and tax.
**Dedup:** distinct from **222-dividend-capture-strategy** (a holding-period strategy, not the
ex-day price mechanics), **408-dividend-yield-factor** and **117-dividend-aristocrats** (yield as
a cross-sectional signal), **740-total-return-vs-price-return** (the effect of adjustment on
long-run performance) and **971-weekly-bar-alignment** (a different provider-mechanics
study).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the ex-day is, why the drop ratio is nearly impossible to average, and what the price really does when a dividend detaches |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | a corporate-actions audit, market-adjusted event construction, four estimators put against a planted truth, bootstrap intervals, cuts by yield/name/era, and the capture trade priced against costs and tax |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`exday/`](exday/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
