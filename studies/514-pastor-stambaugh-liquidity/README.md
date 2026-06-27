# Study 514 -- Pastor-Stambaugh Liquidity Risk

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do stocks whose returns swing harder with aggregate liquidity shocks really pay you more for the privilege?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | High liquidity-beta stocks out-earn low-beta stocks by **+4.28%/yr** in the predicted direction, but HAC *t* = **+1.466** (|*t*| < 2) and placebo *p* = **0.070**. Pastor-Stambaugh (2003) provide a strong prior on a broad CRSP universe; 155 months of large-cap survivors do not independently clear the bar. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Net **+3.52%/yr**, Sharpe **+0.298**, max drawdown **-19.8%**, net HAC *t* = **+1.207**. Turnover is low (gamma is a slow 36-month loading) so costs are mild -- but the gross signal already misses, and it is fragile to window/breadth choices. |
| **Survives the honest gauntlet?** (survivorship-biased) | ![Busted](https://img.shields.io/badge/Busted-c0392b?style=flat-square) | Universe = 45 of the most liquid stocks alive, projected backwards. The natural carriers of liquidity risk (illiquid, distressed, delisted names) are **absent** -- this is the most favourable possible test, and it still fails |*t*| >= 2 net of costs. |

> **In one sentence:** the Pastor-Stambaugh liquidity-risk premium shows up in the right direction (+4.3%/yr for high liquidity-beta) on a large-cap survivor panel, but at HAC *t* = +1.47 (placebo *p* = 0.07) it never clears the statistical bar -- Weak signal, Fragile tradability, and on the most liquidity-rich universe imaginable the honest gauntlet is Busted.

## What we tested

Pastor & Stambaugh (2003): build an **aggregate market-liquidity series** (sign-flipped,
z-scored cross-sectional average of monthly Amihud illiquidity, |return| / dollar-volume),
take its monthly first difference as the liquidity *shock*, and estimate each name's
**liquidity beta** gamma -- the loading on that shock in a trailing 36-month regression
controlling for the market. Each month, rank by trailing gamma (signal public at the prior
month-end close: **one execution lag**), go long the top-gamma tertile and short the
bottom-gamma tertile, equal-weight, dollar-neutral, hold one month. Panel: 45 large-cap
names, yfinance daily prices + volume 2010-2025 (155 monthly long-short observations).
Risk-free rate = 3%/yr constant. This is the liquidity *risk loading* -- deliberately
distinct from the Amihud illiquidity *level* of [Study 140](../140-amihud-illiquidity/).
Universe is survivorship-biased; we name it and treat results as upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the liquidity-risk mechanism in plain language, synthetic positive control, real-panel long-short, honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the aggregate-liquidity series, gamma cross-section, placebo null, costs, window/breadth robustness, equity curve, survivorship discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`pastor_stambaugh_liquidity/`](pastor_stambaugh_liquidity/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
