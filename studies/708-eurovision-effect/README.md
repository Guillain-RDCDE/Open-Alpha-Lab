# Study 708 — Eurovision-Effect 🎤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Only 1 of 4 primary cuts clears *t* ≥ 2: winner, 1-month AR **+1.63%**, *t* = **+2.07**, placebo *p* = **0.038** — but it fails at 1 week (*t* = −0.06), fails for hosting (*t* = −0.40), fails pooled (*t* = +1.37), and 5/12 leave-one-out draws push it below the bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Zero-look-ahead capture (enter the first close *after* the result) never clears *t* ≥ 2 net of costs: best case *t* = +1.78 at *p* = 0.058. A naive-looking host "sell-the-news" dip (*t* = −2.39) evaporates under its own placebo (*p* = 0.180). |
| **Winning beats hosting?** | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Winner beats host at both horizons; 1-month Welch *t* (winner − host) = **+1.95 to +1.99** — consistently the right sign, one hair short of certified. |

> **In one sentence:** the Eurovision "feel-good bump" shows up in exactly one of six
> tested cuts — barely, fragile to which single event you drop, gone once you charge
> costs and require a realistic entry — and half the countries that ever won or hosted
> (Ukraine three times over) never even had a tradable stock market to test.

## What we tested

Believers of the "Eurovision economy" story (a fixture of financial media every May)
claim the winning — or hosting — country's stock market gets a national-pride bump
around the Grand Final. We hardcode all 26 Eurovision editions 2000→2025 (2020
COVID-cancelled) with their winner and host countries, map each to a single-country
ETF where one exists (12 winner-country and 13 host-country events survive — the rest
have no tradable vehicle, most strikingly three-time winner Ukraine), and measure the
abnormal return vs the VGK Europe benchmark from the last close before the Saturday
final through 1 week and 1 month after, with a random-window placebo, a jackknife, and
a zero-look-ahead tradable-capture test.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the country-by-country tour, why half the winners have no market to test, the trade that almost-but-doesn't work |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery, the placebo, the jackknife fragility, the event anatomy, the winner-vs-host Welch split, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`eurovision_effect/`](eurovision_effect/). The Eurovision calendar is
hardcoded from Wikipedia; **survivorship named**: only countries with a still-listed
single-country ETF enter the sample, and two of those (Russia's ERUS, Portugal's PGAL)
were later delisted. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
