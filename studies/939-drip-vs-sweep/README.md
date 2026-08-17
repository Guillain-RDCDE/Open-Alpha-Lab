# Study 939 — DRIP or Sweep 💧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | DRIP beats quarterly sweeping on all three funds and in both eras (SPY **+2.09**, VYM **+3.46**, SCHD **+3.78** bps/yr) — the right sign everywhere. But the HAC *t* is only **+1.09 / +1.18 / +1.41** (pooled **+1.15**, and *t* only rises to +1.6 at a 252-day HAC lag), and every bootstrap CI straddles zero. The yield ordering is **weaker than it looks**: on the matched 2011-2026 window the three collapse to **+3.01 / +3.73 / +3.78**, and *per unit of realised yield* the ordering **reverses** (1.70 / 1.21 / 1.18 bps per 1% of yield) — so the headline spread is mostly window, not yield. Only the *annual*-sweep variant clears \|*t*\| = 2 (SCHD +17.8 bps/yr, *t* = **+2.73**). Survivorship: three large surviving US ETFs, chosen because they are what people own. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The pooled edge is **+2.81 basis points a year** — **14 to 23 currency units a year on 10,000 invested**, less than one ETF ticket's half-spread. It moves by ±0.2 bp across a 0-10 bp cost sweep, while the *unobservable* pay-date lag moves it across **+0.7 to +7.0** bps/yr. The assumption is wider than the answer. |

> **In one sentence:** Reinvesting each distribution the day it lands does beat parking it in T-bills until quarter end — by **under three basis points a year**, a figure smaller than the pay-date lag we cannot observe — so switch DRIP on because it is free and requires no decisions, not because it earns you anything you could measure.

## What we tested

Two investors hold **the same shares of the same ETF** from day one and differ in one
respect only: **DRIP** buys more shares on the pay date; **SWEEP** parks the cash in
**BIL** and buys at the next quarter (or year) end. The distribution stream is
**reconstructed from the two price legs** — `D_t = P_{t−1}·(TR_t/TR_{t−1}) − P_t` — and
audited against the vendor's reported cash (77/77, 77/77, 59/59 events matched over the
raced window) and against the total-return index itself (a zero-lag DRIP reproduces it
to 0.04%). One execution lag, one-way costs × amount reinvested, excess-of-cash Sharpes
both sides, HAC *t*, 63-day block bootstrap, era and rate-regime cuts, and sweeps of
every labelled **ASSUMPTION** (pay lag 0-45 days, DRIP/sweep cost, quarterly vs annual).
SPY/VYM ∩ BIL 2007-05-30 → 2026-06-30, SCHD from 2011-10-20. **Dedup:** distinct from
**143-dividend-capture** (trading the ex-day drop), **516-dividend-month-premium** (a
cross-sectional stock-selection effect), **57-yield-trap** / **206-dividend-aristocrats**
(whether high-yield *stocks* win), **934-lump-sum-vs-dca** / **936-rebalance-bands**
(*new* money and whole-portfolio weights, not the fund's own payout stream),
**938-open-vs-close-execution** (which bar you fill on, not how long the cash waits),
and **916-withholding-drag-international** (how much income *reaches* you, not when you
put it back to work — though 939 reuses its two-leg cache convention).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the DRIP box is the most over-sold checkbox in retail investing, what three bps a year actually buys, and the one case where laziness costs real money |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the reconstruction and its audit, the terminal-wealth gap with HAC *t* and block-bootstrap CIs, the matched-window test that deflates the yield ordering, era / rate-regime cuts, the assumption sweeps, and the power calculation that settles the stamp |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`drip_sweep/`](drip_sweep/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
