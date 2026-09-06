# Study 1004 — How Many Stocks 🔢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how many holdings does it take to remove diversifiable risk? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | The textbook curve is real. Across 40 large-cap names over 16 years, the average equal-weighted portfolio's volatility falls from 27.9% with one stock to 17.0% with all of them, and **10 holdings** capture 90% of that reduction. Evans and Archer's fifteen is the right answer to the question they asked, and anyone repeating it is not making an arithmetic error. |
| **Tradability** — does the volatility answer survive when you ask about terminal wealth instead? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | They are answering the wrong question. Standard deviation describes the *average* portfolio; an investor holds *one*, for decades. Measured on the dispersion of terminal wealth across randomly drawn portfolios — what actually varies between one investor and another — 90% of the available benefit needs **32 holdings**, 3.1× the textbook number. With 10 stocks the 5th-to-95th percentile of outcomes still spans a factor of **2.6×**; reaching the textbook's implied comfort takes far more names. The mechanism is skew, not covariance: 0% of these names lost money outright over the period and the top decile produced 49% of the basket's total return, so a small portfolio probably misses the names that mattered. Its median terminal wealth sits 5% below the mean at 10 names and -1% below at 32. Volatility cannot see any of this, because averaging across draws is exactly the step that hides it. |

> **In one sentence:** Twenty stocks removes 90% of the *volatility* you can diversify away and leaves the spread of actual outcomes at 2.6× between the 5th and 95th percentile — the terminal-wealth criterion asks for 32 names, not 10.

## What we tested

Evans & Archer (1968) plotted portfolio standard deviation against the number
of holdings, saw it flatten around fifteen names, and created a textbook fact. This study
reproduces that curve exactly — it is real — and then plots two others on the same draws.

**Standard deviation describes the average portfolio; an investor holds one.** Measured on the
dispersion of *terminal wealth* across randomly drawn portfolios, which is what actually differs
between one investor and another, 90% of the available benefit takes roughly twice as many
holdings. Tracking error against the index gives a third answer again. Three criteria, three
numbers, and the familiar "twenty stocks" is not wrong so much as unlabelled.

**The mechanism is skew, not covariance.** Single-stock returns are right-skewed — a minority of
names carry the basket — so a small portfolio probably misses them and its *median* outcome lags
its *mean* outcome. Averaging across draws is precisely the step that hides this, which is why
no amount of care with the textbook statistic would have revealed it. A synthetic cross-section
with **independently tunable** correlation and return dispersion makes the claim falsifiable:
one knob moves the volatility curve, the other moves the wealth curve, and they are not the same
curve.
Survivorship is stated up front rather than footnoted: the basket is names still listed in 2026,
which inflates the level of every curve and the comparison between curves far less.
**Dedup:** distinct from **1010-correlation-matrix-stability** (estimating the covariance),
**631-equal-weight-vs-cap-weight** (weighting) and **1006-most-stocks-underperform-cash** (the
cross-section itself rather than portfolio size).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why 'twenty stocks is enough' answers a question about the average investor rather than about you |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | three diversification curves on identical draws, the skew mechanism, rebalancing versus buy-and-hold, and a two-knob identification test |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`howmany/`](howmany/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
