# Study 754 — Beige-Book-Tone 📖

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a positive-tone Beige Book precede an equity drift up? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The *direction* is right at every horizon (5-day **+0.40%** vs base **+0.04%**, up-rate **68%** vs **64%**) — but **no horizon clears t ≥ 2** (best Welch *t* = **+1.44**, placebo *p* = **0.06**), and the dose-response HAC *t* grazes 2 at *only* the 1-day horizon before collapsing to **+0.14** at 3 days. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | "Long the 5-day window on every cheerful book" sits in cash **~90%** of the time and earns **+2.2%/yr** net vs buy-and-hold's **+14.2%**. Almost nothing to harvest. |
| **Anecdote leads the tape?** | ![Not supported](https://img.shields.io/badge/Anecdote_leads_the_tape%3F-Not_supported-8b949e?style=flat-square) | Drop the 2020 crash and the positive-minus-base gap halves to **+0.21pp** (*t* = **0.69**); corr(tone, *prior* 5-day return) = **−0.24**. Cheerful books cluster in expansions — that's the risk premium on a schedule, not a lead. |

> **In one sentence:** a positive-tone Beige Book really is followed by a hair more upside, in the right direction at every horizon, but the tilt is statistically insignificant (best *t* = 1.44), evaporates once the 2020 crash is removed, and a cheerful-book overlay in cash 90% of the year captures almost nothing — so the Fed's "mood ring" **describes** an economy the market already priced rather than **leading** the tape.

## What we tested

Fed-watching folklore reads the **Beige Book** — the Fed's anecdote-driven digest of regional business conditions, published eight times a year ~two weeks before each [FOMC meeting](https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm) — for its **tone**, and says a *cheerful* book is a green light for stocks in the days after. We score each release's [Loughran-McDonald](docs/references.md) net optimism, align it to daily SPY with a strict release-day-close entry (no look-ahead), and run an event study: forward 1/3/5/10-day drift split by tone sign, a Welch *t* and placebo null, a Newey-West dose-response regression, an ex-2020 regime test, and a costed overlay. (The tone here is a small, **clearly-labelled proxy** — a narrative-anchored reconstruction, not a live full-text scrape, which is the beat-7 extension; the release calendar and SPY are real.) A synthetic planted-edge control confirms the engine would light up on a real link.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "read the Fed's mood, front-run the market" is mostly cheerful books living in bull markets, and why buying every one of them earns crumbs — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event-window drift split by tone, a Welch *t* + placebo null, the Newey-West dose-response, the decisive ex-2020 regime confound, the costed overlay, and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`beige_book_tone/`](beige_book_tone/). Tone is a **labelled LM-net-tone proxy** (a hardcoded reconstruction, not a live full-text count), named as such; release dates and SPY are real. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
