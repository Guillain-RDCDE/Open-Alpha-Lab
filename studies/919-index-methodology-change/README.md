# Study 919 — Methodology Shock 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The pooled abnormal return is **−14.1 bps** on the announcement leg (placebo *p* = 0.80) and **+35.7 bps** on the effective leg (*p* = 0.54) — **opposite signs**, both sitting inside a randomisation null band of [−129, +124] bps. The smallest raw *p* across **nine** windows is 0.42 and every Bonferroni-adjusted *p* is 1.000; both bootstrap CIs straddle zero; the eras disagree in sign; dropping any one of the seven events swings the pooled CAR through zero. **Survivorship: none to name** — four continuously listed wrappers picked ex ante by the index each tracks; the selection risk lives in the hand-assembled event list, which the jackknife and placebo test interrogate — and one mis-transcribed announcement date, found and corrected in audit, moved the headline by 22 bps on its own. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The beta-hedged pair loses **−7.7 bps gross** and −29.3 bps net per event (*t* = −0.62, 43% win rate); **every** cell of the cost × borrow sweep is negative, including the zero-cost corner. The one version that looks profitable (+24 bps, naive 1×/1×) is unhedged market beta wearing a costume, and the deployed-capital overlay's +65 bps in nineteen years is five post-2007 trades of noise (Sharpe +0.19, HAC *t* = +0.37, against SPY's +0.54). |

> **In one sentence:** Index rule changes really do force every tracker to trade a large basket on a known date, but by the time that flow is diluted across the hundreds of names inside the wrapper you can actually buy, nothing survives that is big enough to pay for the spread — and the one window that looks significant (*t* = −3.43 before the announcement) is a seven-observation *t*-statistic that the randomisation test prices at a coin flip.

## What we tested

A **market-model event study** over a hardcoded calendar of **eight index rule changes** —
Nasdaq-100 special rebalances (2011, 2023), the S&P float-adjustment phases, the S&P
multiple-share-class ban and its reversal, Russell banding and the move to semi-annual
reconstitution. Each affected wrapper is raced against a sibling the change does not touch
(**QQQ/SPY**, **SPY/IWM**, **IWM/MDY**): fit `r_treated = a + b·r_control` on the 250
sessions ending 21 before the event, cumulate the abnormal return, pool. One execution lag
(news at the day-0 close, trade on at **+1**), 5 bps one-way × NAV on both legs, borrow on
the short and financing on the residual `1−beta` dollar exposure (both swept), daily
**total-return** closes, as-of 2026-06-30. The **event list is a PROXY** — transcribed by
hand from index-provider announcements, five of eight dates `approximate`, and one wrong
date caught in audit — so it is stress-tested with a 2,000-draw placebo, a nine-window
sweep with Bonferroni, and a drop-one jackknife.
**Dedup:** distinct from **320-russell-reconstitution**
(the recurring *annual* recon calendar, not the rulebook), **604-month-end-rebalancing-flows**
and **836-timing-luck** (portfolio rebalancing, not index construction), and from its lot
neighbours **913** (tracking difference) and **918** (creation halts), which race wrapper
*quality* rather than the index rule that governs it.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why forced index flow *should* be front-runnable, why it evaporates inside the wrapper, the *t*-statistic trap, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | both CAR legs, the placebo randomisation null, minimum detectable effect, the nine-window Bonferroni sweep, jackknife, era cut, beta-hedged costed pair with cost × borrow and financing sweeps, the deployed-capital race, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`methodology_shock/`](methodology_shock/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
