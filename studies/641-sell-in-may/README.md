# Study 641 — Sell in May 🍂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does Nov→Apr really trounce May→Oct? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The deep, **price-only** ^GSPC tape (1950–2026) clears the bar — Welch **t = +2.78**, Newey-West **t = +2.96**, year-block bootstrap **t = +3.16** — but the **dividend-inclusive** tapes that describe a real portfolio (^SP500TR 1988–, SPY 1993–) sit at **t = 1.1–1.3**, uncertified. Direction is consistent everywhere; magnitude and significance are not. |
| **Tradability** — does a Nov→Apr timer beat buy-and-hold? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No. On every dividend-inclusive tape the timer **trails buy-and-hold on CAGR by ~2.2–2.3 pts/yr and on excess-of-cash Sharpe too** (0.49–0.52 vs 0.52–0.54) — because May→Oct pays a **positive** ~+3.8%/half-year, not zero. Lower drawdown is just less exposure. |
| **Driven by a handful of bad Septembers/Octobers?** | ![Confirmed](https://img.shields.io/badge/Bad_autumns%3F-Confirmed-8b949e?style=flat-square) | Dropping the 5 worst Septembers + 5 worst Octobers (10 of 917 months, 1.1% of the sample) cuts the winter-summer gap by **35%** and drops the deep-tape Welch *t* **below the certification bar** (2.78 → 1.88). |

> **In one sentence:** "sell in May" is real on the deep price-only tape (76 years, *t* ≈ 2.8–3.2) but rides heavily on ten famous crash autumns, can't be certified on the modern dividend-inclusive sample (*t* ≈ 1.1–1.3), and even where it's real, a cash-in-summer timer **loses to buy-and-hold on both return and risk-adjusted terms** because summer pays positive money, not nothing.

## What we tested

**"Sell in May and go away"** (the Halloween indicator; Bouman & Jacobsen 2002, *AER*) — U.S.
equities earn almost all their return Nov→Apr, so an investor should hold stocks in winter and
sit in cash May→Oct. We test the winter/summer monthly split on ^GSPC (1950–2026, price-only,
matching the deep-history literature), SPY (1993–) and ^SP500TR (1988–, genuine total return),
with a Welch/HAC split, a **year-block bootstrap** and **sign test** paired one point per
Halloween year, a decomposition of how much of the effect is a handful of crash Septembers/
Octobers, and a cost-charged **Halloween timer** (long Nov→Apr, cash May→Oct via ^IRX) raced
against buy-and-hold on **excess-of-cash Sharpe**, not raw CAGR. **Dedup:**
[55-summer-lull](../55-summer-lull/) tests the same claim and reaches a compatible WEAK/MIRAGE
verdict; this study adds the year-block bootstrap, the price-only-vs-total-return split (they
disagree), the "bad autumns" decomposition, and an excess-of-cash timer race. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "sell in May" is half-true, why the other half is a handful of famous crashes, and why sitting in cash still loses you money |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the year-block bootstrap, the price-only-vs-total-return disagreement, the crash-autumn decomposition, and the excess-of-cash timer backtest |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sell_in_may/`](sell_in_may/). ^GSPC/SPY/^SP500TR are indices/index-trackers (no
survivorship); ^IRX supplies the cash leg. **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
