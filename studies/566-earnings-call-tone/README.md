# Study 566 — Earnings-Call-Tone 🗣️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does the emotional tone of a company's earnings call — upbeat vs guarded — predict its stock afterwards?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does call tone forecast post-call drift? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The literature (Loughran-McDonald 2011; Price et al. 2012) says a **surprise-controlled** tone slope predicts drift — but there is **no free real tape** of scored transcripts × event-time CAR, and this desk grants `REAL` only to a robust *t* ≥ 2 on **real** data. So it is **capped at `WEAK`**. On the seeded panel the engine is faithful (controlled slope-*t* **+7.95**, placebo *p* **0.0005**, flat **−0.19** at the null) — a machinery proof, not market evidence. |
| **Tradability** — does the tone edge pay? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The naive tone long-short *looks* huge (+5.76%/event, net **+21.7%/yr**) — but that is PEAD in a linguistic costume. The *real* edge is the surprise-controlled slope, which needs vendor-grade transcript scoring and clean event-time CAR to isolate; quarterly turnover + a borrow on the guarded short leave a thin, `FRAGILE` residual. |
| **"Confounded by the number?"** | ![Named](https://img.shields.io/badge/Named-8b949e?style=flat-square) | Upbeat calls follow good quarters, so tone is entangled with the earnings surprise. In the **null** world (tone truly adds nothing) the *naive* slope-*t* still averages **+6.50** — a clean false positive — while the *controlled* slope-*t* is **−0.19**. Any tone→drift claim that skips the surprise control is measuring PEAD. |

> **In one sentence:** earnings-call tone as a linguistic cousin of PEAD is a credible, literature-backed effect and our engine banks it when it is planted (controlled slope-*t* +7.95, flat at the null) — but with **no free real tape** it is capped at `WEAK`, and its whole danger is confounding by the number: at the true null the *uncontrolled* tone slope manufactures a *t* of +6.5, so only a surprise-controlled read isolates the words, making this a `WEAK` × `FRAGILE`, confound-`NAMED` study.

## What we tested

The claim (Price, Doran, Peterson & Bliss 2012; Loughran & McDonald 2011): the **net tone** of an
earnings-call transcript — its (positive − negative) sentiment-word share under a finance lexicon —
forecasts the stock's **post-call drift** (the cumulative abnormal return *after* the call), a
linguistic cousin of post-earnings-announcement drift. Because scored transcripts joined to event-time
CAR are a **paid-vendor / hand-scored** product with no free feed, this study is **synthetic-only**: a
deterministic, seeded event panel (40 firms × 12 quarters = 480 calls) whose single knob `tone_beta`
plants a linguistic drift on the *residual* tone (the part orthogonal to the numeric surprise). We report
the **naive** tone→CAR slope, the honest **surprise-controlled** slope (the isolated linguistic edge), a
**label-shuffle placebo** null, the tradable tone long-short with costs + a short borrow, a four-window
sign-stability sweep, and a **seed-robust synthetic positive control** (25 seeds) that proves the engine
banks a planted edge and stays flat at the null. The `REAL` stamp is off the table by design (no real
tape → `WEAK`), and the confound is named openly. *Distinct from the numeric-surprise drifts
([363 PEAD-drift](../363-pead-drift/), [534 revenue-surprise-drift](../534-revenue-surprise-drift/)) — those
drift on the number; this drifts on the words — and from the aggregate-mood studies
([259 news-tone](../259-news-tone/), [392 glassdoor-sentiment](../392-glassdoor-sentiment/)).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "call tone" is, why the upbeat-minus-guarded chart looks like free money, and the plain-language reason it's mostly the earnings number in disguise |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the naive vs surprise-controlled tone slope, the false-positive at the null, the placebo, the long-short with costs & borrow, the four-window sweep, and the seed-robust synthetic positive control |

This is a **synthetic-only** study: the reproducible headline run (480-call panel, planted
`tone_beta = 0.020`, panel fp `3ec051a6458a`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); the offline machinery lives in
[`earnings_call_tone/`](earnings_call_tone/), and the real-tape fetch returns an empty frame by design
(named on the Signal axis).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`earnings_call_tone/`](earnings_call_tone/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
