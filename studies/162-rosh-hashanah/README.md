# Study 162 -- Rosh-Hashanah

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Mean window return **−0.87 bps**, HAC *t* = **−0.02**; gap vs matched random −11.5 bps (*t* = −0.17). Remove 2008 and the mean **flips to +38.6 bps**. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Timing net mean −9.1 bps/window (*t* = −0.22); gross is already near zero; no break-even cost exists. |
| **Driven by 2008?** | ![Yes](https://img.shields.io/badge/Yes-8b949e?style=flat-square) | The 2008 Yom Kippur window fell during the Lehman collapse (−17.76% in eight days). It is the single observation that turns the 46-year mean negative. |

> **In one sentence:** "Sell Rosh Hashanah, buy Yom Kippur" is a one-observation story -- remove the Lehman-crisis window of 2008 and the S&P 500 actually rises during the High Holiday period; the matched-random baseline and the HAC t-stat both confirm there is no exploitable signal here.

## What we tested

The Wall Street adage "Sell Rosh Hashanah, buy Yom Kippur" claims that Jewish institutional
investors reduce risk before the solemn Days of Awe, depressing U.S. equities in the ~8-day
window between the two holidays. We hardcode the Hebrew calendar event table (1980--2026, from
Reingold & Dershowitz 2018) and test the S&P 500 (^GSPC) close-to-close return in the
[RH-eve close, YK close] window across 46 annual observations (1980--2025). We pin it against
(a) 40 matched random windows per year drawn from the same calendar month and (b) the
unconditional 8-trading-day baseline -- then apply a HAC Newey-West t-stat and an explicit
2008 leave-one-out sensitivity.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the adage, the bar chart showing 2008 dominance, the matched-random comparison, plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, subsamples, matched random, 2008 sensitivity, synthetic positive control, cost table |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`rosh_hashanah/`](rosh_hashanah/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
