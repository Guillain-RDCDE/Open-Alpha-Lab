# Study 121 — Magic-Formula

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Top-decile excess = **+1.2%/yr**, HAC *t* = **+0.44**; long-short inverts at −10.6%/yr (*t* = −1.43) — statistically indistinguishable from noise on this panel. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No real edge to harvest; the long-short sign is backwards on the S&P 500; survivorship bias inflates every raw number. |
| **Beats a random portfolio?** | ![No](https://img.shields.io/badge/No-8b949e?style=flat-square) | The top decile beats only **61%** of random same-size draws — barely above chance. |

> **In one sentence:** Greenblatt's quality+cheapness rank earns a statistically invisible +1.2%/yr above the equal-weight S&P 500 and inverts on the long-short — the formula's native edge lives in small-cap value, not in the blue-chip survivor panel tested here.

## What we tested

Joel Greenblatt's *The Little Book That Beats the Market* (2005) ranks every stock by two axes — **Return on Capital** (EBIT / Invested Capital, quality) and **Earnings Yield** (EBIT / Enterprise Value, cheapness) — sums the two ranks, and buys the top 20–30 for a year. We implemented it on real SEC EDGAR 10-K filings (~233 S&P 500 tickers, 2008–2024) using the desk's shared EDGAR concept caches. Invested capital = Assets − CurrentLiabilities − Cash; book EV = Equity + LongTermDebt − Cash. Fundamentals from fiscal year *y* predict returns in year *y+1* (conservative one-year reporting lag). The top decile is tested against the equal-weight market and against 500 random same-size portfolios. **Survivorship-biased panel** (current S&P 500 projected back) — all results are upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, the inversion on large caps, the random-portfolio comparison in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat, random-portfolio null distribution, synthetic positive control sweep, factor decomposition |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`magic_formula/`](magic_formula/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
