# Study 732 — Tour-de-France-Effect 🚴

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The three-week Tour window is a mild *drag*, not a bump: raw EWQ **−0.35%** (*t* = −0.37), France-minus-Europe abnormal **−0.21%** (*t* = −0.78) — both slightly negative, hit rate 13/30 below a coin flip, and indistinguishable from an ordinary three weeks (placebo *p* = 0.167). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is no positive edge to trade — the seasonal is negative before costs and **−0.45%** net of 5 bps. "Buy French stocks during the Tour" is a systematic small loss: paying spread to rent the summer doldrums. |
| **A France-specific effect?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | France does not beat Europe during the Tour (it slightly lags; Welch *t* = −0.13). The mild July softness is ordinary pan-European summer beta — the folklore mis-attributes a region-wide, calendar-driven non-event to a bike race. |

> **In one sentence:** the "Grande Boucle bump" doesn't exist — the Tour runs through the
> middle of the "Sell in May" summer window, French equities drift *down* ~0.35% over the
> race, and once you net out Europe there is nothing France-specific left at all.

## What we tested

Financial-media colour every July claims French markets get a feel-good, summer-holiday
seasonal while the country follows the Tour de France — the calendar-window cousin of the
real sports-sentiment effect ([Edmans, García & Norli 2007](docs/references.md)). We
hardcode all 30 editions 1996→2025 (2020 COVID-shifted to Aug/Sep — a built-in "race vs
calendar" probe) with their Grand Départ and final-stage dates, and measure the window
return on `EWQ` (iShares MSCI France, total-return) both raw and *abnormal* vs the `VGK`
Europe benchmark — a one-sample *t* across the 30 independent editions, a random-window
placebo, a CAC-40 price-only cross-check, and a costed zero-look-ahead "trade it" timer
(the Tour dates are public a year ahead, so there is no information lag to argue about).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why July is the *worst* month to expect a bump, the drag that isn't a bump, and why the trade loses money |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery, the raw-vs-abnormal decomposition, the random-window placebo, the event anatomy, the 2020 race-vs-calendar probe, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tour_de_france_effect/`](tour_de_france_effect/). The Tour calendar is
hardcoded from Wikipedia; **price-only vs total-return labelled** (`^FCHI` is the
dividend-free CAC 40 price index, cross-check only; the abnormal test uses total-return
`EWQ`/`VGK`). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
