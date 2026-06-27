# Study 501 -- Idiosyncratic-Volatility

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do stocks with high *residual* (market-beta-removed) volatility really earn LOWER returns -- the famous IVOL puzzle?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The Ang-Hodrick-Xing-Zhang (2006) puzzle predicts a **positive** low-minus-high spread; on this survivor panel it is strongly **negative** (-21.5%/yr, HAC *t* = **-4.29**, placebo p = 0.000). The puzzle does **not** replicate -- it inverts. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The only "edge" is the reversed leg (long high-IVOL), a pure survivorship artifact you can only trade with hindsight. Sharpe **-1.16**, max DD **-94%**, net ~= gross. Not investable forward. |
| **Puzzle replicates on a survivor basket?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | High-IVOL stocks (NVDA/TSLA/AMD/META) **out-earned** low-IVOL by +21.5%/yr -- the textbook survivorship inversion. Quintile returns are *increasing* in IVOL, the mirror image of AHXZ. |

> **In one sentence:** the idiosyncratic-volatility puzzle inverts on a survivor large-cap basket -- high-IVOL names hugely out-earn low-IVOL (low-minus-high = -21.5%/yr, *t* = -4.29), a statistically overwhelming relation that points the wrong way for the published anomaly because conditioning on survival keeps exactly the high-IVOL lottery winners.

## What we tested

Ang, Hodrick, Xing & Zhang (2006): each month, sort stocks by **idiosyncratic volatility** --
the annualised std of the trailing-252-day CAPM *residual* (market beta removed, NOT raw total
vol, which is Study 330). Go long the lowest-IVOL quintile, short the highest, dollar-neutral,
entering one trading day after the signal (one documented execution lag). Charge 5 bps one-way
x turnover plus 50 bps/yr borrow on the short leg. Panel: 50 large-cap S&P 500 names, yfinance
daily prices 2013-2025 (142 monthly observations). RF = 3%/yr constant. The universe is
survivorship-biased -- we name it, and it is the whole story: it flips the puzzle's sign.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "idiosyncratic" vol means, why the puzzle is puzzling, the synthetic positive control, the real-tape inversion, the survivorship mechanism, honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling residual-vol construction, quintile table, window & quantile sweeps, equity curve and drawdown, placebo null, survivorship discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`idiosyncratic_volatility/`](idiosyncratic_volatility/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
