# Study 520 — External-Financing-Anomaly 💸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do big external-finance raisers underperform the retirers? *(survivor basket — dead raisers absent, an upper bound)* | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The sign is **reversed** here: long-retirers / short-raisers loses **−8.5%/yr** (one-sample *t* **−1.12**, HAC *t* −1.69), and a label-shuffle placebo beats the real sort **86%** of the time (p = 0.86). No signal on this tape. |
| **Tradability** — is there a spread to trade? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | yfinance exposes only ~5 fiscal years of cash-flow detail, so just **3 complete annual cross-sections** survive the lag + 12-month hold. Net of 10 bps/leg + 50 bps borrow the spread is **−9.2%/yr**. Nothing to trade. |
| **"Do big raisers underperform here?"** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | In a 2022–2024 mega-cap window the heaviest *raisers* (debt-funded AI-capex names) were the **winners** — the short leg out-earned the long leg in 2 of 3 years (2024 by +23pts). The engine *does* recover a planted penalty at *t* 7.6, so the flat result is the tape, not the sort. |

> **In one sentence:** the Bradshaw–Richardson–Sloan external-financing anomaly — big raisers of debt + equity should subsequently lag — does **not** replicate on a 45-name large-cap survivor basket: with only three complete annual cross-sections the long-retirers/short-raisers spread loses **−8.5%/yr** (*t* −1.12, placebo p 0.86), the sign reversed by a window where debt-funded AI-capex raisers were the winners, while our synthetic control recovers a planted penalty at *t* 7.6 — proving the machinery, not the verdict.

## What we tested

The **external-financing anomaly** (Bradshaw–Richardson–Sloan 2006): firms that raise a lot of
external finance — debt **and** equity together — subsequently underperform, part overinvestment,
part managers timing a dear stock. We compute their cash-flow-statement measure — **net external
financing** (equity issued − repurchased − dividends, plus debt issued − repaid) scaled by **average
total assets** — for a fixed large-cap survivor basket, sort it each year, and form a long-retirers /
short-raisers book with one execution lag (90-day reporting lag, enter one day after the public date,
hold 252 trading days), a one-sample *t*, a HAC *t*, a **label-shuffle placebo**, a tercile/quartile
robustness sweep, real costs + short borrow, and a deterministic **synthetic positive control** that
plants a penalty and proves the engine recovers it. *Distinct from
[64 Share-Shuffle](../64-share-shuffle/) (equity issuance only), [244 Asset-Growth](../244-asset-growth/)
(the uses of the cash, not the sources), and [153 Net-Operating-Assets](../153-net-operating-assets/) /
[231 Sloan-Accruals](../231-sloan-accruals/) (balance-sheet bloat / accruals).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "raising lots of money should be a red flag" is a real academic finding, and why it backfired on the AI-capex mega-caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the scaled-XFIN signal, the long-short with one-sample & HAC *t*, the label-shuffle placebo, costs + borrow, the cut-robustness sweep, and the synthetic positive control |

The fingerprinted real-data run (45 survivors, fundamentals fp `ca0833b2ddb8`, prices fp
`4106bae4e4d7`, 2022–2024) is in [docs/results.md](docs/results.md); the offline machinery proof
runs on the synthetic world in
[`external_financing_anomaly/data.py`](external_financing_anomaly/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine:
[`external_financing_anomaly/`](external_financing_anomaly/). Signal = the cash-flow statement's net
external financing scaled by average total assets. Basket is **survivors** — named on the Signal
axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
