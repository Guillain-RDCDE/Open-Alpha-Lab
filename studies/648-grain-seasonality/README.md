# Study 648 — Grain-Seasonality 🌾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the old-crop/new-crop calendar show up in corn/wheat/soy? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Pooled spring (weather-scare) vs harvest spread across CORN/WEAT/SOYB, 2011→2026: **−0.17%/month, Welch *t* = −0.19**, block-bootstrap 95% CI **[−1.97%, +1.75%]**. Zero of 36 (grain × month) cells clear the Bonferroni-×36 bar (\|t\| ≥ 4.05); two of three grains (corn, soy) point the *wrong* direction from the claim; the best month for every single grain is **October** — inside the harvest window the claim says should be the low. |
| **Tradability** — can a holder capture it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | WEAT alone bleeds a **certified −90.3 bps/month (*t* = −5.09)** to its own weighted roll vs a roll-naive futures proxy — bigger than any spring/harvest point estimate in the study. The seasonal long/short timer's apparent Sharpe gain is a de-leveraging artifact of sitting in cash 5 months a year, not a discovered edge: net pooled Sharpe is still **−0.08**. |
| **"Beats a coin?"** | ![Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square) | The timer's active-leg hit rate, pooled across all three grains, is **100/216 = 46.3%** (Wilson [39.8%, 53.0%]) — nominally *below* even money, mean per active month *t* = **−0.09**. |

> **In one sentence:** the "spring planting-scare high, harvest-time low" story sounds like
> physics and has a real theory-of-storage mechanism behind it, but on the three grain ETFs
> retail can actually buy — CORN, WEAT, SOYB, 2011→2026 — no calendar month, no grain, and no
> pooled spring-vs-harvest spread clears the desk's own multiple-testing bar (best pooled *t* =
> −0.19), two of three grains point backwards, and the one ETF (WEAT) whose roll cost we can
> size loses more to its own contango every month than the entire seasonal is worth: **None,
> and a Mirage.**

## What we tested

The claim: US grains carry a calendar — a spring weather-risk premium (planting, pollination,
winter-wheat green-up) that bleeds out as harvest pressure arrives. We test the **tradable**
expression of it: CORN, WEAT and SOYB (Teucrium single-commodity ETFs, 2011→2026), not a
theoretical spot index. Per grain: a 12-cell one-sample HAC-*t* month table (36 cells across 3
grains, **Bonferroni**-corrected), Welch *t* of the best/worst month vs the rest, and a
spring-vs-harvest Welch *t* using each grain's own hardcoded USDA crop-progress window
(planting/weather-scare/harvest — facts, no network). Pooled across all three grains with a
circular block-bootstrap CI. A calendar-known seasonal long/short timer (no execution lag —
the USDA windows repeat every year) is charged 4 legs/yr × 10 bps × NAV and compared gross/net to
buy-and-hold; its active-leg hit rate is tested against a fair coin. The third axis names the
**ETF's own roll**: CORN/WEAT/SOYB vs the roll-naive ZC=F/ZW=F/ZS=F front-month splice (a
spot-price proxy, never tradable as a continuous series) sizes how much of any calendar edge the
ETF's weighted-roll mechanics already give back. **Dedup:**
[307-coffee-seasonality](../307-coffee-seasonality/) (frost/harvest, a different mechanism),
[308-cocoa-squeeze](../308-cocoa-squeeze/) (a squeeze, not a calendar),
[309-oj-frost](../309-oj-frost/) (tail risk, not a smooth seasonal),
[226-crude-seasonality](../226-crude-seasonality/) /
[227-natgas-winter](../227-natgas-winter/) (energy, no storage-cycle mechanism),
[639-gasoline-rvp-seasonality](../639-gasoline-rvp-seasonality/) (a *statutory* deadline, not an
agronomic one — and it's `REAL`) and [651-sugar-seasonality](../651-sugar-seasonality/) (a
different crop, different crush cycle) never test the corn/wheat/soy planting-to-harvest
calendar. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why grain traders swear by "sell the harvest, buy the scare" — and why the three ETFs anyone can actually buy show none of it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 36-cell Bonferroni table, the per-grain and pooled Welch/HAC splits, the block-bootstrap CI, the ETF-vs-futures roll-drag test, the timer's costs, and the coin-flip hit-rate test |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`grain_seasonality/`](grain_seasonality/). CORN/WEAT/SOYB are single, continuously-listed
ETFs (no survivorship panel); the ETF's own roll mechanics are named and sized on the third axis.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
