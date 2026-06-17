# Study 273 -- Lego-Returns

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do retired Lego sets beat the stock market?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Gross Lego CAGR **8.3%** trails the price-only S&P **8.8%**; mean annual excess return is **negative**, Newey-West HAC t = **-0.67** gross, **-1.54** net. Index is survivorship-/selection-biased upward. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net of a one-way ~12% resale fee (eBay/BrickLink) plus storage, insurance, single-item illiquidity, the Lego CAGR (~5.9%) trails a cheap, dividend-paying index fund. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Beta to the market is ~0.01 -- the *diversification* half of the claim is real; the *beats-stocks* half dies on dividends + fees + survivorship. |

> **In one sentence:** retired Lego is a fun, genuinely uncorrelated collectible, but tested against the honest benchmark (the price-only S&P, then net of the fees you pay to actually sell), it does not beat the stock market -- and the published returns are upper bounds.

## What we tested

The claim (Dobrynskaya & Kishilova 2022, "LEGO: The Toy of Smart Investors"): retired
Lego sets returned ~10%/yr on the secondary market, beating stocks with near-zero
correlation. We hardcode a curated annual Lego secondary-market price index (1987-2024,
base 100, in `data.py`), join it with S&P 500 calendar-year **price** returns from the
repo cache, and run the honest comparison: gross *and* net-of-resale-fee CAGR, a
Newey-West HAC t-stat on the mean annual excess return (the bar for a REAL signal), a
CAPM beta on the diversification claim, and a paired sign test. The synthetic positive
control confirms the machinery detects a Lego premium when one is planted; the real tape
shows a small, negative excess. Survivorship/selection bias is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the price-only benchmark, the resale-fee toll, the genuine low correlation, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Newey-West HAC t on excess return, CAPM beta, cost-sensitivity grid, paired sign test, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lego_returns/`](lego_returns/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
