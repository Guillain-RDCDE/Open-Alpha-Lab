# Study 295 -- Stablecoin-Supply

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The tradable, **one-month-lagged** predictive slope (BTC return ~ standardised supply growth) is HAC *t* = **+1.59** across 99 months -- below the *t* >= 2 bar. The *contemporaneous* slope is larger (t = +2.15) but that is reflexive co-movement, not forecasting. What little lagged edge existed has **fully decayed**: t = +3.05 (2018-2020) -> +0.01 (2023-2026). Literature + on-chain support, synthetic positive control passes -- but no robust real lagged signal. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The supply-timed long/flat strategy beats buy-and-hold on paper (SR 0.85 vs 0.61) but the **excess over buy-and-hold is insignificant (HAC *t* = +0.72; bootstrap excess-Sharpe CI [-0.42, +0.90], 21% negative)** -- it is BTC beta-timing in a single secular bull market, not alpha. Costs are *not* the binding constraint (8.9% monthly switch rate); the absence of a tradable signal is. |
| **Reflexivity / data caveat** | ![Named](https://img.shields.io/badge/Reflexive--coincident-8b949e?style=flat-square) | Supply mints when money is already flooding in and prices are already rising; the contemporaneous-beats-lagged pattern is the textbook signature of a coincident, non-tradable series. Curated public-aggregate supply series; single-survivor BTC spot (price-only == total-return). |

> **In one sentence:** stablecoin supply growth *co-moves* with Bitcoin (contemporaneous t = +2.15) and a naive supply-timing rule looks great riding BTC's bull-market beta -- but the **tradable one-month-lagged signal clears no inference bar (t = +1.59), adds no significant excess over buy-and-hold (t = +0.72), and has decayed to zero since 2020**, making the "dry powder fuels the next leg" thesis a reflexive mirage.

## The claim

> *Does stablecoin supply growth fuel the next leg?*

## What we tested

We join a curated monthly **total stablecoin supply** series ($bn: USDT + USDC +
DAI + BUSD + the long tail, 2018-2026) with the BTC-USD monthly tape and build
two views of the "dry powder" thesis: (1) a **one-month-lagged timing rule** --
hold BTC next month when supply grew this month, else sit in cash -- pinned
against buy-and-hold BTC with HAC *t*-stats, a turnover/cost sweep, and a
bootstrap CI on the excess; and (2) a **predictive regression** of next-month
BTC return on standardised supply growth, with a contemporaneous (non-tradable)
twin that exposes reflexivity, plus a sub-period decay split. A deterministic
synthetic positive control confirms the engine recovers a planted dry-powder
premium and reads ~zero on the null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the dry-powder story in plain language, why timing rides BTC's beta, why "contemporaneous beats lagged" means *coincident not leading*, and how the edge vanished after 2020 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | lagged vs contemporaneous HAC regressions, bootstrap excess-Sharpe CI, sub-period decay, turnover/cost sweep, reflexivity caveat, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`stablecoin_supply/`](stablecoin_supply/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
