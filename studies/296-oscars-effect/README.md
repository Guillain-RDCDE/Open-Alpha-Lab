# Study 296 — Oscars-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Event-day-0 abnormal return **-27.9 bps**, HAC t = **-1.35**, placebo p = **0.18**; no day in the -1..+3 window clears \|t\| = 2; the CAR wanders around zero. What little sign there is runs *opposite* to the folklore. n = 31 too small to detect anything below ~40 bps. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The "buy the day after the Oscars" rule loses money **gross** (-23.8 bps/trade) before a cent of cost; net of costs it is -25.8 bps. No vehicle, no edge. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The Oscar box-office bump is real and local to the winning film; the broad index registers nothing the next morning beyond ordinary daily noise. |

> **In one sentence:** the market does not care who wins Best Picture — the morning after the Academy Awards the S&P 500 does nothing distinguishable from a random day, and what tiny tilt exists is mildly *negative*, the opposite of the "cultural-event rally" story.

## What we tested

The Oscars-Effect is folklore by analogy: the ceremony is one of the most-watched live
broadcasts in the US, so "the market must react." We hardcode all 31 ceremony dates and
Best Picture winners (67th–97th, 1995–2025) in `data.py`, join them to the ^GSPC daily
price tape, and run a clean **event study** on the first full session after the broadcast
(one execution lag baked in — the ceremony evening itself is never used). We report the
event-day-0 abnormal return with a Newey-West HAC t-stat, the CAR profile over relative
days -1..+3, a tradable rule net of one-way costs versus buy-and-hold on the same exposed
days, and a 5,000-draw permutation/placebo control. A synthetic positive control confirms
the machinery detects a planted ~50 bps post-ceremony bump; the real tape shows none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the box-office bump vs the index, the morning-after chart, the verdict in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event-study AAR/CAR, HAC t-stats, the tradable rule net of costs, the permutation/placebo control, the n=31 power calculation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`oscars_effect/`](oscars_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
