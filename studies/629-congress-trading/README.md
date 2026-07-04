# Study 629 — Congress Trading 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do senators' disclosed purchases beat the market? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The replicable trade (buy at the first close **after disclosure**) earns a calendar-time alpha of **−0.0 to +3.7%/yr** with **Newey-West t ≤ 1.11** at 3/6/12-month holds — and a 20-seed random-dates placebo shows even that drift is the **stock list, not the timing** (real − placebo = **+0.4%/yr, +0.49 sd**). The Ziobrowski-looking pooled 12-mo number (+4.87%, naive *t* = 6.35) is an **overlap illusion**. On a **survivor-tilted** panel (~20% of events drop with dead tickers — a tailwind for the claim) the answer is still no. |
| **Tradability** — can you copy-trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The disclosure is a median **19 days stale**, the alpha is statistically zero and indistinguishable from random timing on the same tickers, and costs only subtract (net +3.3%/yr at 10 bps, *t* = 0.99). Copy-trading senators buys you an equal-weight stock basket, not an edge. |
| **"The famous names are the alpha"?** | ![Busted](https://img.shields.io/badge/Famous_names_beat%3F-Busted-8b949e?style=flat-square) | The meme subset (Perdue, Loeffler, Inhofe — the 2020 DOJ-probe names) shows a spectacular pooled 12-mo premium (**+7.6 pp, Welch t = 4.6**)… which is a **clustering artifact** of one senator's overlapping 2019–20 buys riding one rebound. Calendar-time: famous alpha **+0.09%/yr (t = 0.02)** vs boring **+4.66%/yr (t = 1.85)**. The zero lives with the famous names. |

> **In one sentence:** on the modern disclosed tape (2,776 Senate stock purchases, 2015–2021,
> entered at the only date a member of the public can trade — the disclosure), senators' picks
> show **no market-beating signal** (calendar-time NW *t* ≤ 1.11, and the drift that exists is
> fully explained by *which stocks* they hold, not *when* they disclose), the copy-trade is a
> **mirage** on 19-day-stale information, and the famous-names meme is **busted** — Ziobrowski's
> 1993–98 alpha does not survive into the disclosed era, exactly as Belmont et al. (2022) found.

## What we tested

We rebuild the congressional-trading claim on the **Senate Stock Watcher** PTR archive — every
per-day disclosure report (the filename carries the **disclosure date**, the only replicable
entry), 2,776 stock purchases by 26 senators, 2015-01 → 2021-03 (the scraper's hard stop; David
Perdue alone is 39% of events, so an ex-Perdue aggregate is reported). Entry at the first close
**strictly after** disclosure (one execution lag); 3/6/12-month buy-and-hold abnormal returns vs
**SPY** (both total-return) as color; the **verdict statistic** is a **calendar-time portfolio
Newey-West t** (Fama 1998 — the fix for overlapping windows). A 20-seed **random-dates placebo**
(same tickers, random timing) separates stock-list drift from disclosure-timing information;
party / size / famous splits use Welch t; costs charge 0/5/10 bps one-way over the hold.
**Survivorship named**: ~20% of events drop with tickers that no longer resolve — a bias *toward*
the claim, which still fails. A deterministic synthetic world with a planted post-disclosure
drift (null flat over 20 seeds, planted edge lights up at *t* = 12) proves the machinery. As-of
2026-07-03; fingerprint `884027dc84c9`.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a senator's disclosure actually is (a 19-day-old fax, not a signal), why the "+4.9% in 12 months" headline is double-counting one rally, and what copy-trading senators really buys you — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | BHAR pooled t vs calendar-time NW t under overlap, the 20-seed random-dates placebo (timing vs stock list), party/size/famous Welch splits, the famous-names clustering autopsy, costs, and the synthetic faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`congress_trading/`](congress_trading/). Sibling: [263-insider-buying](../263-insider-buying/) is **corporate insiders** (Form 4, own-firm trades); this is **politicians** (Senate PTRs, any stock). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
