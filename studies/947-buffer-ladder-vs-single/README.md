# Study 947 — The Buffer Ladder 🪜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does laddering add anything? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | **No gap clears \|*t*\| = 2 in either direction**: BUFR is **+1.33 pp/yr** over an equal-weight DIY basket of its own vintages (HAC *t* = **+1.18**, bootstrap CI [−0.81, +3.22]) and **−0.96 pp/yr** against a **beta-matched** DIY ladder (*t* = **−1.69**; that gap's bootstrap CI excludes zero on a 21-day block and **straddles zero on a 5- or 10-day block** — 9 of 15 (block, seed) settings, so the exclusion is the nuisance parameter talking and the HAC *t* is what we stamp on). The visible edge is **beta** — SPY-beta **0.579** vs the basket's **0.439** — not laddering. And the entry-point luck laddering exists to average away is worth a **2.4%** cut in one-year return variance (**4.2%** on daily returns — exactly what the equally-correlated-legs closed form predicts), because the four vintages are **0.889** correlated. Survivorship: the surviving flagships of a category that has closed products, and an **n-of-1** wrapper over 5.9 years with one down-year. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to bank either way. Risk-adjusted, the wrapper is **behind the basket it wraps** (excess Sharpe **+0.777** vs **+0.853**, and vs POCT alone at **+0.943**) with a **3 pp deeper** drawdown — including 2022, the one year a buffer was needed, where it lost **more than double** the DIY basket. Shorting it against a beta-matched DIY ladder targets −0.96 pp/yr across **2.35%** tracking error at *t* = −1.69: a coin-flip with a borrow bill. Both arms merely **tie** the dumb beta-matched SPY/BIL mix (+0.30 pp/yr, *t* = +0.31). |

> **In one sentence:** the laddered buffer wrapper did out-return every do-it-yourself alternative on this tape — and none of it was laddering: hold beta constant and it falls **behind** a home-made four-vintage basket by roughly its own extra fee, while the entry-point luck it is sold to average away turns out to be worth **2.4% of one-year return variance**, because four vintages of the same fund on the same index are **0.889 correlated** and averaging things that move together removes almost nothing.

## What we tested

Whether **BUFR** — one ticker holding the Innovator Power Buffer ladder, for a management fee **on top of** the underlying funds' expense ratios — earns that layer against what a private investor can build with four trades. Five arms, all **excess-of-cash** (minus BIL's total return) and all **total-return vs total-return**: the wrapper; each of the four quarterly vintages (**PJAN/PAPR/PJUL/POCT**) held alone; an equal-weight **DIY basket** of them; the **beta-matched DIY ladder** (that basket topped up with SPY to the wrapper's beta); and the beta-matched **SPY/BIL** mix. **One execution lag** — every estimated weight, the basket's rebalance target and every beta, is formed on data through the close of day *t* and applied at *t+1*; betas are expanding-window, never full-sample. HAC *t* on each gap, paired block-bootstrap CIs **plus a block-length sensitivity sweep on the only CI that excludes zero**, an era cut, a cost sweep, a rebalance-frequency sweep, a **declared-proxy** fee sweep (the quoted 0.79%/yr vintage ER and an **assumed** 0.20%/yr wrapper layer, swept 0.00→0.40), and a drawdown and calendar-year table. Synthetic panels certify the detector: it recovers a planted laddering premium at *t* = +4.8 and fires on 0/8 null seeds. BUFR∩vintages 2020-08-11 → 2026-06-30, as-of **2026-06-30**. **Dedup:** distinct from **624-buffer-etf-cost** (what **one** buffer fund costs vs the market — this study takes that as settled and asks whether *bundling* the vintages adds anything), **921-bill-ladder-vs-etf** (the same DIY-vs-wrapper frame on the cash shelf, where the margin is exactly the ER), **892-corporate-bond-ladder** (ladder vs ETF once **duration** is matched), and **934-lump-sum-vs-dca** / **937-tranched-rebalancing** (spreading purchases across dates, where the benefit vanishes on exposure-matching — here we measure the *mechanism*, the vintage correlation, not only the outcome).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a buffer vintage actually is, why four of them are nearly the same fund, the arithmetic of averaging correlated things, and the 2022 row that gives the whole game away |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-of-cash Sharpe race, HAC gap tests, paired block-bootstrap CIs, the beta decomposition, the dispersion measurement, era cut, cost / rebalance / declared-proxy fee sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`buffer_ladder/`](buffer_ladder/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
