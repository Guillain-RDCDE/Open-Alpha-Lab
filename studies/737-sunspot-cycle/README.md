# Study 737 — Sunspot-Cycle ☀️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 11-year solar cycle drive stock returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A fitted 11-year wave explains **R² = 0.36%** of monthly returns (HAC *t* = +1.04 / +1.83, both sub-2); active-Sun months *trail* quiet ones (**−673 bps/yr**, placebo *p* = 0.095); the max−min turning-point contrast is Welch *t* = −1.39. The one \|*t*\| ≥ 2 (fwd-12m after solar **minima**, *t* = +3.33) is wrong-signed for the claim, n = 9, and only *p* = 0.054 vs a random calendar. |
| **Tradability** — could you time the market on the Sun? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A lag-honest "long the active half" timer earns **3.1%/yr** vs buy-and-hold's **6.2%/yr** (excess *t* = −2.65), a lower Sharpe too — it loses, significantly, before costs even bite. |
| **Jevons's sunspot cycle in the tape?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Ten solar cycles of the longest S&P tape, an exogenous known-in-advance calendar, every benefit of the doubt — and there is no 11-year clock in equity returns. |

> **In one sentence:** across ten solar cycles and 98 years of the S&P, the ~11-year sunspot cycle explains **0.36%** of monthly return variation, active-Sun months do slightly *worse* than quiet ones, a solar-clock timer halves buy-and-hold — and the study's single *t* > 2 (the year after a solar minimum) points the wrong way, rests on nine events, and dissolves against a random calendar; Jevons's solar market cycle is real astronomy attached to imaginary alpha.

## What we tested

In the 1870s W. S. Jevons tied the Sun's ~11-year sunspot cycle to commercial crises
(*sunspots → harvests → trade cycles → markets*); the harvest chain is long dead but the
headline — [**"the stock market runs on an 11-year solar clock"**](https://www.nature.com/articles/019033a0)
— keeps being reincarnated. We steelman its strongest form (the cycle is *exogenous and
known in advance*, so no reverse-causation escape hatch), hardcode solar cycles **16–25**
from the SILSO/NOAA record as a **labelled sunspot proxy**, and test the S&P 500 price
index back to **1927** three ways: an 11-year phase regression (HAC), a high-vs-low
activity regime split (block bootstrap + phase-shift placebo), and a forward-return event
study across the 19 independent solar turning points (with a random-calendar placebo) —
then put a lag-honest solar-clock timer through a cost sweep. **As-of 2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the century-long chart, the active-vs-quiet-Sun split, the shiny "buy the solar minimum" number and why it's a trap, the timer that just keeps you out of a rising market |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC phase regression, the block-bootstrap regime split and phase-shift placebo, the independent-turning-point forward-return study with random-calendar placebos, the lag-honest costed timer, the 20-seed synthetic null |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sunspot_cycle/`](sunspot_cycle/). The solar calendar is hardcoded from
WDC-SILSO / NOAA-SWPC; the monthly sunspot series is a **labelled cosine proxy** of the
smoothed number (not the raw SILSO file), and the S&P tape is **^GSPC price-only** (no
dividends — labelled as such, chosen for its ten-cycle length). **Not investment advice**
— research & education. See [LICENSE](../../LICENSE).*
