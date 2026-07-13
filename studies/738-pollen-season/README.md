# Study 738 — Pollen-Season 🤧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do allergy-brand owners beat the market through spring pollen season? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Mean window abnormal return **+2.03%**, one-sample *t* = **+1.06** (n=30); no cut clears \|*t*\| ≥ 2 (vs staples 1.65, spin-off-free core 1.06), the bootstrap CI **[−1.71%, +5.63%]** straddles zero, hit-rate Wilson [45.5%, 78.1%] includes a coin flip. |
| **Tradability** — does the seasonal survive costs? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Long-basket/short-SPY earns a gross **+2.03%/yr** but at *t* = 1.06 it is undistinguishable from zero *before* costs; net of costs + short-leg borrow it fades to +1.70%/yr (*t* = 0.89). No edge to size. |
| **A tradable spring seasonal?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The direction leans the folklore's way (positive mean, 63% hit rate, a borderline placebo *p* = 0.048) — but on 30 independent years nothing clears the desk bar. Suggestive, not certified. |

> **In one sentence:** a basket of the listed allergy-brand owners (Claritin/Bayer,
> Allegra/Sanofi, Zyrtec-Benadryl/Kenvue, Flonase/Haleon, plus private-label Perrigo)
> does beat the market by **+2%** through the March→May pollen window and does so 63% of
> years — exactly the sign the folklore predicts — but at *t* = 1.06 over 30 independent
> springs it is indistinguishable from zero, the one borderline placebo rides entirely on
> the basket's off-season underperformance, and the seasonal long/short is a mirage once
> you charge it costs and borrow.

## What we tested

Every spring, traders repeat a tidy demand-seasonality story: buy the allergy names
*before* pollen season, because tens of millions of US hay-fever sufferers restock
antihistamines and nasal sprays from March through May (a spike the OTC industry's own
category data — CHPA/IRI — shows every year). We steelman it on a **labelled, cited
pollen-season window** (last session of February → last session on/before May 31; AAFA
"Allergy Capitals" / AAAAI pollen calendars), build an equal-weight basket of the five
currently-listed brand owners (**BAYRY, SNY, PRGO, KVUE, HLN**, spin-off coverage named
honestly), and run a per-year event study of the basket's window return vs **SPY** (and
vs consumer-staples **XLP**) across **30 independent springs, 1997→2026** — with a
random-window placebo, a block-bootstrap CI, a calendar-month cross-check, a costed
long/short timer (short pays borrow), and a synthetic positive control. **As-of
2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the +2%-but-not-real headline, why "beats a random window" and "made money" are different questions, and the trade that fades to a mirage |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery, the bootstrap CI, the placebo-vs-zero teaching case, the staples/core-basket robustness, the month seasonality, the costed timer and the 20-seed synthetic null |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`pollen_season/`](pollen_season/). The pollen-season window is a hardcoded,
cited calendar rule (AAFA/AAAAI); BAYRY/SNY/PRGO/KVUE/HLN + SPY/XLP total-return closes
are fetched via yfinance. **Survivorship named:** the basket holds only currently-listed
brand owners, two are recent spin-offs (Kenvue 2023, Haleon 2022) that enter only
post-listing, cross-checked against a full-history 3-name core basket. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
