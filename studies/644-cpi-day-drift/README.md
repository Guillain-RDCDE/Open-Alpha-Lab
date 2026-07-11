# Study 644 — CPI-Day-Drift 📊🌡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do stocks/bonds move systematically on CPI mornings? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | **Real on one leg, none on the rest.** TLT's realized high-low range is genuinely elevated on CPI days — Welch *t* = **+2.70**, one-sided random-calendar placebo **p = 0.00025** over 353 releases since 1997 — bonds get louder on the print, mechanically sound (CPI directly re-prices rate expectations). Everything else is a statistical zero: SPY return *t* = **−0.01**, TLT return *t* = **+0.67**, SPY's own range *t* = **+0.82** (placebo *p* = 0.18) — all fail the **t ≥ 2** bar. |
| **Tradability** — can you harvest it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No directional edge exists to bank: a naive "own SPY only on CPI day" timer starts from a **+3.73 bps vs +3.80 bps** non-edge and turns net **−6.27 bps/event** (≈ **−0.75%/yr**) the moment 5 bps costs are charged. The one certified effect — TLT's elevated range — has no drift to ride; harvesting pure "loudness" needs an options/vol instrument we don't test, and the FOMC precedent (637) says that premium tends to be pre-priced away anyway. |
| **"CPI day = the biggest day of the month" (post-2021)?** | ![Busted](https://img.shields.io/badge/Biggest_day%3F-Busted-8b949e?style=flat-square) | Every metric drifts the "more true since 2022" way (best: TLT range, 8.2% → 15.1% of months), but **none of the four pre/post differences clears *t* = 2** (best case *t* = +1.31), and even the strongest post-2022 rate is a small minority of months — nowhere near literally "the" biggest day. |

> **In one sentence:** the CPI print does something real to the bond market — TLT swings harder on release mornings (*t* = 2.70, placebo *p* = 0.00025) — but neither SPY nor TLT shows a bankable *direction*, no naive timer survives costs, and the trading-desk truism that "CPI day is the market's biggest day of the month" is directionally plausible but statistically **unconfirmed** even in the post-2022 hiking-cycle era — Mixed signal, a Mirage to trade, and a Busted myth.

## What we tested

We hardcode **353 actual CPI release dates 1997-01-14 → 2026-06-10** (the identical,
source-verified BLS calendar used by sibling study
[602-macro-announcement-premium](../602-macro-announcement-premium/)) and split daily SPY and
TLT close-to-close returns — plus each asset's high-low realized range — into CPI days vs all
other days: Welch *t* (points/log where relevant), a Newey-West dummy-regression *t*, Wilson
hit rates, and 20-seed × 1,000-draw random-calendar placebos (two-sided for return, one-sided
for range). An event window [−3..+3] checks pre-release drift and post-release persistence
(neither certifies). A regime split at **2022-01-01** — the Fed's Dec-2021 hawkish pivot, chosen
from the FOMC calendar, not fit to the data — tests whether CPI's market salience structurally
increased, as a **difference**, not eyeballed. The third axis operationalizes the "biggest day
of the month" folklore directly: for every month, is the CPI session the single largest-move day,
against the honest chance baseline (≈4.8%)? A two-knob synthetic control (independent planted
return-shift and volatility-multiplier) proves the machinery detects either effect without
manufacturing the other. **Dedup:**
[602-macro-announcement-premium](../602-macro-announcement-premium/) (the pooled
FOMC+CPI+NFP equity premium — this study isolates CPI alone, plus the bond-side and
realized-range legs 602 doesn't test), [643-payrolls-day-effect](../643-payrolls-day-effect/)
(its sibling — Nonfarm Payrolls mornings, same protocol, different print) and
[637-fomc-vol-crush](../637-fomc-vol-crush/) (the FOMC *decision* afternoon, a policy call, not
a pre-market data release — and implied vol, not realized range). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why CPI morning became a market-moving event, why bonds react and stocks don't, and why "the biggest day of the month" is a story that outran the data |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits on return and range for SPY and TLT, the event-window anatomy, the regime-split difference test, the "biggest day" hit-rate test with Wilson intervals, the naive-timer cost sweep, and the two-knob synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cpi_day_drift/`](cpi_day_drift/). The CPI calendar is hardcoded from the BLS
archived-news-release index (identical table to study 602); SPY/TLT are index-tracking ETFs, no
survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
