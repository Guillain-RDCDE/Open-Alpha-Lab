# Study 650 — Heating-Oil-Seasonality 🛢️❄️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does HO=F earn a heating-season premium? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No calendar month clears the Bonferroni-12 bar. The **autumn-build** window (Sep–Nov) is **negative and wrong-signed**: −0.94% vs +1.51% off-season, Welch *t* = **−1.71** — the single closest-to-significant number in the study, pointing *against* the folklore. The **winter-draw** window (Dec–Feb) is unremarkable (*t* = +0.18), and the pooled heating window (Sep–Feb) **under-performs** the off-season it's supposed to beat (*t* = −0.89). |
| **Tradability** — does a seasonal timer beat buy-and-hold? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A timer built exactly as the folklore prescribes (long Sep–Feb, cash otherwise) throws away roughly four-fifths of buy-and-hold's Sharpe **before a single basis point of cost** (0.07 vs 0.27) — sitting in cash skips June, HO=F's actual best month (+3.1%). The one real retail vehicle for this trade, UHN, lost to the raw futures splice in 8 of its 10 seasons and **was wound down in 2018** — it cannot be bought today at any price. |
| **"Heating oil rallies into winter" — the trader's rule of thumb?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Busted, not merely weak: the autumn half of the claim runs **backward** on 26 years of tape, the winter half is noise, and the ETF built to sell retail this exact trade no longer exists. |

> **In one sentence:** the "buy heating oil ahead of winter" story fails on its own terms —
> the autumn-build half of the claim is negative and closer to significant than anything else in
> the study, but pointing the *wrong way* (Welch *t* = −1.71), no month survives Bonferroni
> correction across 26 years of HO=F, a by-the-book seasonal timer loses most of buy-and-hold's
> Sharpe before costs even apply, and the one ETF (UHN) that ever let retail express the trade
> was discontinued in 2018 — **None, and a Mirage**.

## What we tested

The claim, in its own two stages: distillate demand should pull heating oil futures up through
the **autumn build** (Sep–Nov, the market pricing the coming winter ahead of time) and hold the
rally through the **winter draw** (Dec–Feb, physical inventories fall as furnaces run — a real,
EIA-documented seasonal in the *physical* market). We test the calendar directly on **HO=F**
(NY Harbor ULSD futures, yfinance, 2000-09 → 2026-06, 309 months): per-month one-sample *t*'s
with a Bonferroni-12 bar, Welch *t*'s for autumn-build and winter-draw each against an off-season
control (Mar–Aug), and a seasonal timer (long Sep–Feb, T-bill otherwise) raced against
buy-and-hold with one-way costs charged per switch. HO=F's Yahoo chain is a **spliced continuous
front-month series — not back-adjusted** — so the roll/contango cost a real futures holder pays
is already baked into the tape by construction, not modeled separately. The third axis pairs
**UHN**, the actual retail ETF for this trade (2008-04-10 → 2018-09-11, then wound down — it is
not buyable today), against the HO=F splice over the identical Sep–Feb windows to ask whether the
wrapper eats even more of whatever's left. A 20-seed synthetic world with a planted heating-season
premium proves the machinery (never cited as evidence). **Dedup:**
[227-natgas-winter](../227-natgas-winter/) (the parallel winter-demand claim for natural gas —
also wrong-signed), [639-gasoline-rvp-seasonality](../639-gasoline-rvp-seasonality/) (a
*statutory* spring/autumn calendar on the gasoline-crude spread — real on the spread, mirage on
the roll), [306-crack-spread](../306-crack-spread/) (does the crack level *predict* refiners —
a different mechanism entirely) and [226-crude-seasonality](../226-crude-seasonality/) (crude's
*spring* seasonal, a different commodity and season) never test HO=F's own autumn/winter
calendar. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "cold weather = higher heating-oil prices" sounds airtight, and why the tape says the opposite in the exact month it should be truest |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Bonferroni month table, the autumn-build/winter-draw Welch splits, the timer race with costs, the UHN-vs-splice paired gap, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md) (fingerprint `5e77c79ac501`).

---

*Engine: [`heating_oil_seasonality/`](heating_oil_seasonality/). HO=F is a spliced continuous
futures chain (no survivorship); UHN's discontinuation (2018) is named explicitly on the third
axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
