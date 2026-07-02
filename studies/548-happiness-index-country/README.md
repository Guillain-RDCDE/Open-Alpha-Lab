# Study 548 — Happiness-Index-Country 😊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do the stock markets of the 'happiest' countries (World Happiness Report) beat the gloomiest ones?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do happy nations' markets beat gloomy ones? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On the headline 2024-03 → 2026-06 window the spread is the **wrong sign**: the *gloomy* tercile earned **+87.0%** vs the *happy* tercile **+44.6%**, spread **−42.4%** (Welch *t* −1.65, placebo *p* 0.066). Spearman rho(happiness, return) = **−0.19** (*t* −0.90) — a nothing. The sign **flips across WHR editions** (positive in 2019-21, negative recently). No stable *t* ≥ 2. **Synthetic-only:** the investable cross-section is ~24 countries — a robust real-tape *t* ≥ 2 is unreachable, named on this axis. |
| **Tradability** — does the sort pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 24-name single-country-ETF sort, ~7 per bucket, sign-unstable across editions, wrong sign before costs (gross **−42.4%**, net **−43.8%** after 5 bps/leg + 60 bps borrow). Nothing to harvest. |
| **Spurious correlation?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Textbook: ~24 data points, a rank correlation of −0.19, a spread whose *direction* depends on which WHR edition you pick. Whatever macro factor led the window (EM/Asia here) drove the sort, not happiness. |

> **In one sentence:** sorting the world's investable equity markets by their World Happiness Report rank is a classic **spurious-correlation** demo — on the 2024-26 window the *gloomiest* markets (Korea +209%, Spain, South Africa) roughly *doubled* the happiest Nordics, the happiness-return rank correlation is −0.19 and insignificant, and the sign flips edition-to-edition, all on a cross-section too small (~24 countries) to ever earn a robust signal.

## What we tested

The alt-data folklore that **optimistic, well-governed societies have better stock markets**. We
take the **World Happiness Report 2024** country rank (1 = happiest), keep the *investable* set — 24
countries with a liquid single-country ETF — and join each rank to the ETF's forward **total
return**. Then the honest teardown: a rank-sorted **long-happy / short-gloomy** tercile spread with
a **Welch two-sample *t***, the **Spearman rank correlation** the folklore lives on, a country-level
OLS slope, a **label-shuffle placebo** null (the spurious-correlation antidote), costs + a modest
short borrow, a **four-edition robustness** sweep, and a deterministic, seed-robust **synthetic
positive control** that plants a happiness effect and proves the engine catches it (and stays flat
at the null). Because there is no free, clean, aligned happiness × tradable-index panel and the
investable cross-section is tiny (n ≈ 24), this study is **synthetic-first** and capped below REAL —
the data-availability / small-N limitation is named on the Signal axis. *Distinct from the
time-series mood studies ([257 AAII-Sentiment](../257-aaii-sentiment/),
[335 Buzz-Sentiment-ETF](../335-buzz-sentiment-etf/)): this is a **cross-country** sort on a
societal well-being index, not a sentiment timer.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the claim is, why "happy countries → better markets" is a spurious-correlation trap, and why the gloomy markets won |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the tercile sort with a Welch *t*, the Spearman rank correlation, the placebo null, the country-level slope, the four-edition sign-flip, costs + borrow, and the seed-robust synthetic positive control |

The illustrative real-data run (24 country ETFs, scored 2024-03, forward to 2026-06, panel fp
`90b4d9c25e41`) is in [docs/results.md](docs/results.md); the offline machinery proof runs on the
deterministic synthetic world in
[`happiness_index_country/data.py`](happiness_index_country/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`happiness_index_country/`](happiness_index_country/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
