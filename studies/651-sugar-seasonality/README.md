# Study 651 — Sugar-Seasonality 🍬

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Brazil/India crush calendar show up in raw sugar? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Pre-harvest-tight (Jan–Mar) vs crush-glut (Apr–Jul) spread on CANE, 2011→2026: **+0.63%/month, Welch *t* = +0.46**, block-bootstrap 95% CI **[−1.89%, +2.98%]**. Zero of 12 calendar-month cells clear the Bonferroni bar (\|t\| ≥ 3.47) — not one even clears the far weaker nominal \|t\| ≥ 2 line. |
| **Tradability** — can a holder capture it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | CANE's own roll drags a nominal (uncertified, *t* = −0.80) **−20.1 bps/month** against a long-tight bet, and the seasonal timer's apparent Sharpe gain (+0.14 net vs buy-and-hold's −0.29) is a cash-drag artifact of sitting flat 5 months a year on an ETF that has lost money on average. |
| **"Beats a coin?"** | ![Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square) | The timer's active-leg hit rate is **61/104 = 58.7%** (Wilson [49.0%, 67.6%]) — nominally above even money, but the interval straddles 50% and the mean per active month is statistically zero, *t* = **+0.60**. |

> **In one sentence:** the "Brazil crushes cane April–November, India crushes it October–April, so
> sugar should be tight and expensive into the Northern-Hemisphere winter and cheap once the
> Brazilian harvest floods the market" story has real agronomics behind it, but on the one ETF a
> retail account can actually buy — CANE, 2011→2026 — no calendar month, no tight-vs-crush spread,
> and no seasonal timer clears the desk's own bar (best pooled *t* = 0.46, best/worst single month
> *t* = 1.36/−1.35, active-leg hit rate indistinguishable from a coin): **None, and a Mirage.**

## What we tested

The claim: raw sugar (ICE No.11) has a harvest calendar driven by the world's two largest cane
suppliers — Brazil's Center-South crush (**April → November**) and India's cane-crushing season
(**October → April**) — with old-crop stocks scarcest, and prices supposedly firmest, in the
Northern-Hemisphere winter just before Brazil's new crush ramps up, giving that premium back as
the crush floods the market every spring. We test the **tradable** expression of it: **CANE**
(Teucrium Sugar Fund, 2011→2026), not a theoretical spot index. A 12-cell one-sample HAC-*t* month
table (**Bonferroni**-corrected), Welch *t* of the best/worst month vs the rest, and a
tight-vs-crush Welch *t* using a hardcoded crush calendar (Brazil/India crush windows and the
claimed pre-harvest-tight window — facts, no network) with a circular block-bootstrap CI. A
calendar-known seasonal long/short timer (no execution lag — the crush calendar repeats every
year) is charged 4 legs/yr × 10 bps × NAV and compared gross/net to buy-and-hold; its active-leg
hit rate is tested against a fair coin. The third axis names **CANE's own roll**: CANE vs the
roll-naive SB=F front-month splice (a spot-price proxy, never tradable as a continuous series)
sizes how much of any calendar edge the ETF's weighted-roll mechanics already give back. **Dedup:**
[307-coffee-seasonality](../307-coffee-seasonality/) (a frost tail-event, not a smooth planting
calendar), [308-cocoa-squeeze](../308-cocoa-squeeze/) (a supply-shock squeeze, not a recurring
calendar) and [648-grain-seasonality](../648-grain-seasonality/) (the same *shape* of test — old-
crop/new-crop, Bonferroni, block-bootstrap, roll-drag, calendar timer — but corn/wheat/soybeans, a
different crop family with its own planting/pollination mechanism, not cane) never test the raw-
sugar Brazil/India crush calendar. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why sugar traders swear by "buy the winter tight, sell the Brazilian harvest" — and why the one ETF anyone can actually buy shows none of it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 12-cell Bonferroni table, the tight-vs-crush Welch/HAC split, the block-bootstrap CI, the ETF-vs-futures roll-drag test, the timer's costs, and the coin-flip hit-rate test |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sugar_seasonality/`](sugar_seasonality/). CANE is a single, continuously-listed ETF (no
survivorship panel); its own roll mechanics are named and sized on the third axis. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
