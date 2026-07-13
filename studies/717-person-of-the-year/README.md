# Study 717 — Person-of-the-Year 🗞️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the cover jinx the honoree's stock? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The pooled 12-month decline is real on the tape (**−135% CAR**, *t* = **−2.95**, placebo *p* = **0.03**, **4/4** honorees down) — but it's **absent at one month** (*t* = −0.18), rests on **four events**, and **vanishes to *t* = −0.00 once you control for the honoree's prior-year run-up** (corr = **−0.58**). Significant raw, fully explained by selection. |
| **Tradability** — can you short it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The shorts even *made money* (net **+42%** avg), so costs aren't the killer — but **~4 tradable business honorees in 25 years** is no capacity, the ex-ante squeeze risk is ruinous (**TSLA ran +25%** against the short first), and the "edge" is short-momentum **beta**, not magazine alpha. |
| **"Cover curse?"** | ![Misattributed](https://img.shields.io/badge/Cover_curse%3F-Misattributed-8b949e?style=flat-square) | The decline is real but it's **zenith mean-reversion**, not a magazine jinx: TIME crowns people at their peak, peaks revert, and the residual curse after removing prior run-up is **zero**. |

> **In one sentence:** every business Person-of-the-Year (Bezos'99, Gates'05, Musk'21, Trump'24) really did underperform the market over the next year — a *t* = −2.95, placebo-*p* = 0.03 "curse" — but it's built on four events, shows up only after month one, and dissolves to *t* = 0 once you account for the fact that TIME crowns stocks at their zenith, so it's selection-driven mean-reversion wearing a magazine's face, not a tradable jinx.

## What we tested

The **magazine-cover curse** says a triumphant cover marks the top — and TIME's
**[Person of the Year](https://time.com/person-of-the-year)**, revealed every mid-December,
is the most-watched cover there is. Real cover-effect data would need every business cover
ever printed, so we hardcode the transparent **census of the honorees who ran a public
company** (AMZN'99, MSFT'05, TSLA'21, DJT'24 — plus the untradable-at-the-time picks named
on the Signal axis) and run a textbook long-horizon **event study**: the **cumulative
abnormal return** (stock minus a `α + β·SPY` market model) over 1–12 months after the
announcement, a placebo null sized to four events, a prior-run-up control for selection, a
borrow-aware short, and a deterministic synthetic power check.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the cover "curse" looks real, why it's only four famous names, and how "crowned at the peak" fakes a jinx — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | market-model CARs by horizon, the placebo null, the prior-run-up regression that zeroes the residual, borrow-aware short economics, and a synthetic power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`person_of_the_year/`](person_of_the_year/). Honorees are an explicit **hardcoded, cited census** (the tradable business Persons of the Year). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
