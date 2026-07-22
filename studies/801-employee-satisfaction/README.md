# Study 801 — Employee Satisfaction 😊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does owning the "100 Best Companies to Work For" earn alpha? *(survivor-picked basket)* | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Our 17-name perennial-member basket out-returns the market **+20.2% vs +15.0%/yr** and — the one favourable result — beats *random* survivor baskets (placebo **p = 0.013**), so it isn't pure survivorship. **But** the *risk-adjusted* claim Edmans made fails the bar: CAPM **α = +2.82%/yr** at Newey-West **t = +1.10** (β = **1.15** eats most of the raw gap), excess-over-market NW **t = +1.90** (< 2), win-rate Wilson floor on 50.0%. Hand-picked **survivors** + market-only control = literature-supported, **tape-unproven**. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs are trivial (2.4%/mo turnover → **0.01–0.03%/yr** drag), but there's no certified alpha to bank: the ~5pp/yr of outperformance is **β = 1.15** — leverage you can buy on SPY directly — and the residual is **t = 1.10** and non-persistent. Beta wearing an intangible's clothes. |
| **Does the edge persist?** | ![Busted](https://img.shields.io/badge/Persist%3F-Busted-8b949e?style=flat-square) | The alpha lives entirely in **2016-2021** (+8.1%/yr, **t = 2.77**) and turns **negative** in **2021-2026** (−2.3%/yr, t = −0.59). One five-year window, then gone — the fingerprint of a lucky stretch or a published anomaly that decayed. |

> **In one sentence:** a hand-picked basket of perennial "Best Companies to Work For" beats the
> market by ~5pp/yr and even out-does random baskets of large-cap survivors (placebo *p* = 0.013),
> but once you charge it for the **β = 1.15** it's carrying, the risk-adjusted alpha is only
> +2.82%/yr at *t* = 1.10 — insignificant, front-loaded into 2016-2021 and negative since — so the
> honest read on a survivor-picked, market-only-controlled basket is **weak signal, no paycheck.**

## What we tested

Edmans (2011) found a portfolio of Fortune's **"100 Best Companies to Work For"** earned a
four-factor alpha of ~3.5%/yr — employee satisfaction as an intangible the market underprices. We
build a **modest, hand-coded, cited** basket of **17 publicly-listed perennial members** (Marriott,
Cisco, Salesforce, Amex, Aflac, Adobe, NVIDIA, Hilton, Accenture, Alphabet…), equal-weight it,
rebalance monthly, and race its **total return** against SPY, 2016-2026 (126 months). The headline
is a **market-model (CAPM) alpha** with a Newey-West *t*; we grade it against a **random
survivor-basket placebo** (the honest survivorship yardstick), a first/second-half persistence
split, a costed long-only timer, and a 20-seed synthetic control. **We are brutally honest about
the two caps:** the basket is hand-picked from *today's* known survivors (survivorship **named on
the Signal axis** — Whole Foods, Ultimate Software and Nordstrom left the tape and are excluded),
and we control only for the market (CAPM), not Edmans' four factors, so a positive alpha here is an
**upper bound**. **Dedup:** [392-glassdoor-sentiment](../392-glassdoor-sentiment/) tests the
*crowd-rating* construct (a different signal), [526-intangible-value](../526-intangible-value/)
tests intangibles *broadly*, and [751-fortune-500-inclusion](../751-fortune-500-inclusion/) tests a
*different* Fortune list (size-ranked) as an *event study* — this is the **best-employer** list held
as a **basket**. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "great places to work" beat the market on paper, why most of that is just higher market exposure, why picking known survivors stacks the deck, and why the edge vanished after 2021 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the CAPM-alpha NW/OLS split, the excess-over-market series, the random survivor-basket placebo, the first/second-half persistence cut, the costed timer, and a 20-seed synthetic power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`employee_satisfaction/`](employee_satisfaction/). The basket is an explicit **hand-coded, cited, survivor-selected** list of perennial "100 Best Companies to Work For" members (survivorship named on the Signal axis; delisted/privatised perennials excluded). CAPM (market-only) alpha, not 4-factor — an upper bound. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
