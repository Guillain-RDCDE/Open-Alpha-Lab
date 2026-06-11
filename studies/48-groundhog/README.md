# Study 48 — Groundhog 🔁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do stocks repeat their calendar-month performance? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — surprisingly. The long-short earns **+7.3%/yr, Sharpe 0.81 (Lo t = 4.1)**, wins 64% of months, and **doesn't decay** (Sharpe 0.81 in both halves of the sample). |
| **Tradability** — can you keep it? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | On paper yes (break-even **38 bp**, net Sharpe **0.59 at 10 bp**), but the book reshuffles *fully every month*, needs shorting ~80 names (borrow frictions), and rests on a survivorship-biased large-cap panel. |
| **"Genuinely same-month seasonal"?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The decisive control: ranking on *other* months' history earns **−3.3%/yr**. The predictability is specific to the same calendar month — not generic momentum. |

> **In one sentence:** the strangest claim on the bench — that a stock repeats its calendar-month performance year after year — actually holds up: it's significant (t 4.1), specific to the same month (a clean control fails), and undecayed; the open question isn't whether it's real but whether monthly turnover and the short book let you keep it.

## What we tested

**Return seasonality** (Heston & Sadka 2008; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it as the *"12-Month Cycle"* at Sharpe `0.340`): a stock's average return in a given calendar month forecasts its return in that same month in future years. It sounds like astrology. We test it on **398 S&P 500 names with 20+ years of history**: each month, rank stocks by their mean return in the *upcoming calendar month* over the trailing 20 years, long the top quintile and short the bottom, and measure the hedge. The decisive guard is a **control that ranks on *other* months' history** — if only the same month predicts, it's a true seasonal. We then charge turnover (the book reshuffles monthly) to see what survives. The offline control is a synthetic panel with a fixed per-(stock, calendar-month) bias (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "stocks have favourite months" sounds absurd and turns out to be real, and what the control proves |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the same-month vs other-month control, the Lo t-stat, the (absence of) decay, the cost sweep and break-even |

The fingerprinted real-data run (398 names, 2000–2026, fp `c768f59e31fe`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic panel in [groundhog/data.py](groundhog/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
