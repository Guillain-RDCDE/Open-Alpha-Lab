# Study 631 — Restatement Shock (Item 4.02) 💣

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does "do not rely on our financials" keep hurting for months? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | **Real on the bomb**: AR[0,+1] = **−2.32%** vs SPY (HAC *t* = **−2.27**, winsorized *t* = −4.55). **Not certified on the drift**: the raw 3-month bleed is a loud **−15.84%** (HAC *t* = −4.41) but collapses to **−4.36% at paired *t* = −0.85** against a same-stock **chronic-decay placebo** — the confessing stocks were melting at the same rate in a random pre-event window. **Deads-missing bias named** (65% of events don't map to a live ticker; bankruptcies drop out, so the tape understates the true damage). |
| **Tradability** — can you short the confession? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The modeled short survives costs+borrow on paper, but the surviving edge is **generic microcap decay, not the event** (event-specific part: *t* = −0.85) — harvested on unshortable names (median pre-event dollar volume **$0.18M/day**, sub-$1 scandal stocks, scarce locates, buy-in risk). |
| **Small-cap story?** — is the drift concentrated where arbitrage can't go? | ![Mixed](https://img.shields.io/badge/Small--cap_story%3F-Mixed-8b949e?style=flat-square) | Small half **−21.80%** (HAC *t* = −3.60) vs large half −9.85% (HAC *t* = −1.56) — the Shleifer-Vishny direction — but Welch *t* of the difference = **−1.52**, not separable; and the placebo shows the drift isn't event-specific anyway. |

> **In one sentence:** the Item 4.02 confession is an instant, certified ~2.3% bomb — but the
> famous "months of drift afterwards" is mostly **composition, not underreaction**: the kind of
> microcap that files a "do not rely" 8-K was already bleeding −13% per quarter vs SPY *before*
> confessing, and the event-specific increment (−4.4%, *t* = −0.85) certifies nothing.

## What we tested

We pulled **1,498 Item 4.02 "non-reliance" 8-Ks** from EDGAR full-text search (2004→2026,
quarter-stratified), mapped 520 to tickers (the unmapped 65% — including post-confession
bankruptcies — are the **named deads-missing bias**), and ran a market-adjusted event study on
the **359** usable events vs SPY: announcement window [0,+1], drift window [+2,+64] entered at
the close of day +1 (one documented lag). Inference is overlap-honest (calendar-month clusters
+ Newey-West), winsorized, and penny-floor-robust. The verdict-maker is an adversarial
**chronic-decay placebo** — the same stock, the same window length, one year earlier — which
absorbs the raw drift almost entirely. A three-world synthetic control (null / true
underreaction / melting-ice-cube confound) proves the machinery can tell drift from decay.
Tradability charges one-way costs × NAV plus borrow (shorts pay borrow). Sibling framing:
[229-beneish-m-score](../229-beneish-m-score/) *predicts* the manipulation; this study prices
the **confession** itself. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an Item 4.02 is, the two-day bomb, why the "months of bleeding" is mostly the kind of stock that confesses (not the confession), and why shorting it is a mirage — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | month-clustered HAC event-study inference, penny floors, horizon & era robustness, the chronic-decay paired placebo, the size split, borrow-aware shorting costs, and the three-world synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`restatement_shock/`](restatement_shock/). Events: EDGAR FTS `items` ∋ 4.02; benchmark SPY; entry close of day +1. Deads-missing bias named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
