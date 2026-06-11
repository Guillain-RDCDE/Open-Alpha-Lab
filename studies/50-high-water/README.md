# Study 50 — High-Water 🏔️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do stocks near their 52-week high out-earn? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No — they *underperformed*. The nearness long-short earns **−8.4%/yr (Sharpe −0.40, Lo t −2.2)** on large caps, 2000–2026: a significantly *negative* result, the opposite of the claim. |
| **Tradability** — is there a premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The gross hedge is negative, so monthly turnover only deepens the loss (net Sharpe −0.49 at 10 bp). |
| **"A distinct anomaly"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The 52-week-high hedge is **0.87 correlated** with the plain 12-month-momentum hedge — the same factor, relabelled with an anchoring story. |

> **In one sentence:** the 52-week-high effect is momentum wearing a behavioural hat — 0.87 correlated with ordinary momentum, *not* the distinct anomaly it's sold as — and on tradable large caps it didn't even pay, producing a significantly negative long-short over 2000–2026.

## What we tested

The **52-week-high effect** (George & Hwang 2004; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.153`): stocks trading near their 52-week high earn higher subsequent returns, via an anchoring mechanism the authors argue is *distinct* from momentum. We test both halves of the claim on **398 S&P 500 names with 20+ years of history**: rank by nearness (price ÷ trailing-12-month high), long the near and short the far, measure the hedge and its Lo t-stat — and crucially, **correlate it with a plain 12-month-momentum hedge** to see whether it's a separate factor at all. The offline control is a synthetic trending panel where nearness and momentum are both predictive and correlated (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy stocks near their highs" is just momentum, and why it lost on large caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the nearness hedge with its Lo t-stat, the 0.87 correlation to momentum, the decay, the cost sweep |

The fingerprinted real-data run (398 names, 2000–2026, fp `c768f59e31fe`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic trending panel in [high_water/data.py](high_water/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
