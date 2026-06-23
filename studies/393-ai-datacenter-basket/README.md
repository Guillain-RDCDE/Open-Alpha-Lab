# Study 393 — AI-Datacenter-Basket 🖥️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the datacenter/power basket genuinely out-earn the market? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | On the tape, **yes**: the equal-weight eight beat SPY by **+47%/yr**, HAC **t = 3.19** (n = 52), and still **t = 3.12** vs QQQ and **t = 2.37** vs the equal-weight field — with **5 of 8** names clearing t = 2 individually. But the eight are **named because they won** (look-ahead), over **one short ~4-year regime**, and the synthetic null shows selection alone manufactures a spread. Real recent spread, hindsight-selected. |
| **Tradability** — could you have captured it forward? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs are trivial (an 8-name hold; net CAGR unchanged). The mirage is that **you could not have named these eight in 2019.** The ex-post "pick the 8 winners" rule reproduces **121%** of the spread, the basket sits at the **100th percentile** of 2,000 random baskets, and even a *blind* eight from the field beat SPY by **+24%/yr**. The +57%/yr is a number only hindsight can buy. |
| **Ride the build-out by buying the basket?** | ![Busted](https://img.shields.io/badge/Ride_the_build--out%3F-Busted-8b949e?style=flat-square) | The +57%/yr decomposes into ≈**+24%/yr** (the candidate field was hot — survivorship) + ≈**+33%/yr** (hand-picking its winners). Neither is forward-specifiable. It's [Study 355 (Mag-7)](../355-magnificent-seven/) in a datacenter costume — the same selection trick. |

> **In one sentence:** the AI build-out is real and the eight-name "datacenter-and-power" basket genuinely beat the market and even the Nasdaq on the AI-era tape (HAC *t* ≈ 3) — but it is the tape that *named* the basket: strip the hindsight and the +57%/yr decomposes into a survivorship-hot candidate field (≈+24%/yr) plus hand-picking its winners (≈+33%/yr), neither of which a 2019 investor could have specified, and a synthetic null reproduces +14%/yr of "edge" from selection on a no-edge tape.

## What we tested

The viral **"buy the build-out"** trade: hold the AI **datacenter-and-power basket** — NVDA, VRT, ETN, CEG, VST, SMCI, ANET, DELL (chips, cooling/electrical gear, merchant-power utilities, AI servers, switches) — equal-weight, monthly-rebalanced, and ride the capex wave instead of stock-picking the one AI winner. Over **52 months (2022-02 → 2026-05)** of yfinance monthly total returns we race the eight against **SPY** (the market) and **QQQ** (the tech tape), and against an **equal-weight basket of the same 30-name datacenter/power candidate field** (the theme-beta control). The spread is large and significant (`t ≈ 3`) — so the question is *why*. We decompose it three ways: against the market, against the field, and against the **ex-post "pick the 8 winners in hindsight"** rule (the basket's selection rule made explicit) and a **2,000-draw random-basket distribution**. A deterministic synthetic null (no name has any edge) shows look-ahead selection manufactures a large spread *from nothing*. Look-ahead / survivorship is named on the **Signal** axis. (Same selection-artefact family as [Study 355](../355-magnificent-seven/) and [Study 356](../356-glp1-basket/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the basket really did win, why you couldn't have named the eight in 2019, and why "pick the winners" is a trick — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC *t* of the spread vs SPY/QQQ/field, the single-name decomposition, the ex-post-selection placebo, the random-basket distribution, the survivorship-vs-selection split, and a synthetic null that manufactures the spread from zero edge |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ai_datacenter_basket/`](ai_datacenter_basket/). The field & membership are an explicit **current-membership projection** (look-ahead + survivorship), named on the Signal axis. **Not investment advice** — research & education; concentrated thematic risk is real. See [LICENSE](../../LICENSE).*
