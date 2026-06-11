# Study 48 — Groundhog 🔁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do stocks repeat their calendar-month performance? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — surprisingly. The long-short earns **+7.3%/yr, Sharpe 0.81 (Lo t = 4.1)**, wins 64% of months, and **doesn't decay** — *on a survivor panel, so the magnitude is inflated*; the **existence** is supported by the same-month-vs-other-month control (which shares the bias) and by Heston-Sadka's bias-free CRSP result. |
| **Tradability** — can you keep it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Thin. Counting turnover **one-way** (~3.2× NAV/mo), break-even is **≈19 bp** and net Sharpe **0.38 at 10 bp** (0.33–0.36 after 25–50 bp/yr short borrow on the ~80-name short book) — alive at large-cap costs, dead by ~20 bp, on a panel that flatters it. |
| **"Genuinely same-month seasonal"?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The decisive control: ranking on *other* months' history earns **−3.3%/yr**. The predictability is specific to the same calendar month — not generic momentum. |

> **In one sentence:** the strangest claim on the bench — that a stock repeats its calendar-month performance year after year — holds up as an *effect* (t 4.1, same-month-specific, undecayed) but is measured here on a survivor panel that inflates the magnitude; what's left after honest one-way costs (~19 bp break-even) and the short book is thin, so it's real-but-fragile, not a free lunch.

## What we tested

**Return seasonality** (Heston & Sadka 2008; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it as the *"12-Month Cycle"* at Sharpe `0.340`): a stock's average return in a given calendar month forecasts its return in that same month in future years. It sounds like astrology. We test it on **398 *current* S&P 500 names with 20+ years of history** — a survivor panel, double-conditioned on the future, so magnitudes are upper bounds (the local `fetch_panel` requires an explicit `allow_survivorship_bias=True`). Each month, rank stocks by their mean return in the *upcoming calendar month* over the trailing **up to** 20 years (≥5 same-month observations in the early years), long the top quintile and short the bottom, and measure the hedge. The decisive guard is a **control that ranks on *other* months' history** — it shares the survivorship bias, so if only the same month predicts, the *specificity* is bias-robust. We then charge turnover one-way (~3.2× NAV/mo) to see what survives. The offline control is a synthetic panel with a fixed per-(stock, calendar-month) bias (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "stocks have favourite months" sounds absurd and turns out to be real, and what the control proves |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the same-month vs other-month control, the Lo t-stat, the (absence of) decay, the cost sweep and break-even |

The fingerprinted real-data run (398 names, 2000–2026, fp `c768f59e31fe`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic panel in [groundhog/data.py](groundhog/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
