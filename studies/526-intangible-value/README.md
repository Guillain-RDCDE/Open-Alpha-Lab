# Study 526 — Intangible-Value

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Plain book-to-market understates value for intangible-heavy firms — does capitalising R&D + SG&A into an *adjusted* book sharpen the value sort enough to matter?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Intangible-adjusted-B/M long-short earns **+2.21%/yr**, HAC *t* = **+0.67** (n=220), fails the label-shuffle placebo (pctile 85.2, p = **0.305**), not robust across split fractions. WEAK not NONE only on the strength of the Fama-French / Lev-Srivastava literature. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of 10 bps turnover + 100 bps/yr borrow the spread is **+1.20%/yr at t = 0.37**, Sharpe **0.18**, max DD **−44%**. Both legs are ~13–16%/yr survivor large-caps; the spread is noise. |
| **Does the adjustment beat plain B/M?** | ![Confirmed](https://img.shields.io/badge/Beats_plain_B%2FM%3F-Confirmed-8b949e?style=flat-square) | Plain B/M is dead flat (**−0.16%/yr**, *t* = −0.05); the adjustment re-ranks the field and the **adjusted − plain** spread is **+2.38%/yr at HAC *t* = +2.93**. The mechanical Lev-Srivastava claim holds — even though it doesn't make value *print money* here. |

> **In one sentence:** capitalising R&D + SG&A measurably re-ranks the value sort and beats plain book-to-market head-to-head (*t* = 2.93), but on a 21-year large-cap survivor basket *neither* value sort earns a statistically significant or tradable premium — Weak signal, Mirage tradability, adjustment-beats-plain Confirmed.

## What we tested

Lev & Srivastava (2019) / Eisfeldt-Papanikolaou (2013) / Peters-Taylor (2017): build an
**intangible-adjusted book** = reported book equity + capitalised R&D (5-yr amortised *knowledge*
capital) + capitalised 30%-of-SG&A (3-yr amortised *organisation* capital). Each month, rank 40
large-caps by intangible-adjusted B/M; long the cheap tertile, short the expensive tertile, one
reporting lag + one execution lag, monthly rebalance. Race the adjusted-B/M long-short against the
**plain** B/M long-short and SPY, with a Newey-West HAC *t*, a label-shuffle placebo, costs +
short-borrow, and a fixed-seed synthetic positive control. Fundamentals from SEC EDGAR
companyfacts; monthly total returns / prices from yfinance, 2005-02 → 2026-05 (256 months). Universe
is survivorship-biased — we name it and treat absolute levels as upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why expensed intangibles break book-to-market in plain language, the adjustment recipe, synthetic positive control, real-panel value race, honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the perpetual-inventory capitalisation, plain-vs-adjusted head-to-head, HAC inference, label-shuffle placebo, split-fraction sweep, equity curve, costs + borrow, survivorship discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`intangible_value/`](intangible_value/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
