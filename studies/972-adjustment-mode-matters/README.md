# Study 972 — Adjusted or Not 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the adjustment convention change the numbers? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The gap between the two conventions is the reinvested dividend, and on this universe it runs from **0.69%/yr** (QQQ) to **6.31%/yr** (HYG) — which on HYG is **128% of its entire total return**. Volatility is almost untouched (largest difference 0.14%), so every risk-adjusted ratio moves with the numerator alone: the Sharpe gap reaches **0.57**. |
| **Tradability** — can it flip a conclusion, not just a decimal? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Yes. Ranking the same universe on price charts instead of total returns reorders **4%** of asset pairs in an average month and picks a different leader in **6%** of them. Run as a momentum sleeve — both arms scored on total returns, so only the *selection* differs — ranking on price gives +7.26%/yr against +6.41%/yr for ranking on total return (-0.85%/yr, Sharpe +0.51 vs +0.47), and it holds a portfolio yielding -0.16% less. The price-only signal is not neutral — it is a systematic bet against income. |

> **In one sentence:** A price chart is a total-return series with the dividends deleted, and on this universe that is up to **6.3% a year** — enough to reorder 4% of a monthly cross-sectional ranking, tilt a momentum sleeve away from income by 0.16%, and move a Sharpe ratio by 0.57.

## What we tested

A price chart and a total-return series are the same tape with one difference: the
second one reinvests the dividends. Which of the two a backtest reads is almost never a
deliberate decision — it is whatever the download defaulted to — and this study measures what
the default is worth. On eight tapes chosen to span the yield spectrum (**QQQ** at the bottom,
**HYG** and **VYM** at the top) we take both conventions from the same provider in the same
pass and measure three things: the return the price-only view deletes (which is exactly the
reinvested yield, and on some of these funds it is most of the total return); the
cross-sectional damage, counting how often a 12-1 momentum ranking of the universe puts the
assets in a different order; and the strategy damage, running the same momentum sleeve twice
— ranking on each panel but **scoring both on total returns**, so the comparison isolates the
*selection* effect from the income effect rather than conflating them.

The whole apparatus is validated on a synthetic panel where every asset earns the same total
return but pays a different yield: there, a price-only ranking must order the universe by yield
and nothing else, and it does.
**Dedup:** distinct from **971-tape-self-consistency** (whether the feed's adjustment is
*correct*, not which convention to use), **969-log-vs-simple-returns** (the return convention,
not the dividend convention), **143-dividend-capture** and **984-ex-day-drop-ratio** (trading
around the ex-date), **201-dividend-growth** / **206-dividend-aristocrats** (dividends as a
*signal*) and **939-drip-vs-sweep** (what a real holder does with the cash).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the same fund drawn two ways, the return the chart deletes, and the momentum sleeve that quietly stopped buying anything that pays income |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | implied-yield decomposition, Sharpe and drawdown under both conventions, Spearman and pair-flip counts on monthly rankings, a selection-isolating backtest with a parameter sweep, and the equal-total-return synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`adj_mode/`](adj_mode/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
