# Study 508 -- Momentum-Crashes

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Momentum prints a premium -- until it doesn't. Daniel-Moskowitz (2016): the factor carries
> rare, severe crashes that detonate in bear-market rebounds when the past-loser leg snaps back.
> Does the crash show up on a modern survivor basket, and does vol-scaling repair it?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- does the 12-1 WML spread pay? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The quintile winners-minus-losers earns **-0.34%/yr** gross, HAC *t* = **-0.072**, label-shuffle placebo *p* = **0.537** -- indistinguishable from a coin. **Survivorship is named on the signal axis**: the basket is names *still trading in 2026*, so the loser leg's natural shorts (firms that trended into delisting) are absent -- this flat number is already an *upper bound*, and the crash an *under*-statement. |
| **Tradability** -- is there anything to trade? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net **-1.43%/yr** after 10bps/leg + 50bps borrow, with a **-44.9%** drawdown, **-17.7%** worst month and **-0.63** skew. No positive expectancy, fat crash risk -- nothing to monetise. |
| **Does the crash exist, and does vol-scaling repair it?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Yes on both. Every crash month is the **short leg snapping back** (Oct-2020: losers **+25.8%** vs winners +8.1% -> WML -17.7%); momentum bleeds **-27.8%/yr** in bear-and-down months vs **+0.7%/yr** in calm ones. Vol-scaling halves the negative skew (**-0.63 -> -0.27**) and shaves 9pts off the drawdown (**-44.9% -> -35.6%**) -- the Daniel-Moskowitz repair, faithfully reproduced. |

> **In one sentence:** the canonical 12-1 momentum spread earns nothing on a 38-name large-cap
> survivor basket (HAC *t* ~ 0, placebo *p* > 0.5) -- but its **crashes are real and exactly as
> Daniel-Moskowitz describe**: loser-leg snap-backs in bear-market rebounds, a fat left tail
> (skew -0.63), and a -45% drawdown -- and **vol-scaling demonstrably repairs the tail** (skew
> halved, drawdown shaved) without ever conjuring a tradable edge.

## What we tested

Daniel & Moskowitz (2016), *Momentum Crashes*, built on the Jegadeesh-Titman (1993) 12-1 factor:
each month rank the basket by trailing 12-month return *skipping the most recent month*, go long
the top quintile / short the bottom, equal-weight, dollar-neutral, hold one month. We then
dissect the **crash anatomy** (worst months, deepest drawdown, leg attribution, skewness),
condition the WML on **Daniel-Moskowitz market regimes** (a trailing-12-month bear indicator and
a panic/rebound state), and test the **vol-scaling repair** (constant-volatility "dynamic
momentum"). The apparatus is proven on a deterministic synthetic panel with a *baked-in*
relative-strength drift **and** a planted loser snap-back (plus a no-momentum null). Real tape:
**38 large-cap survivor names** (yfinance daily adjusted-close, 2012--2025, 151 holding months),
SPY for the regimes. One execution lag (form on the month-end close, hold the next month -- no
same-bar fill, no look-ahead); costs of 10bps/leg/rebalance plus 50bps/yr short borrow; a
label-shuffle placebo null; survivorship named on the **signal** axis as an upper-bound caveat.
*Distinct from [507 Cross-Sectional-Momentum](../507-cross-sectional-momentum/) (the level test
on the same basket) and [237 Residual-Momentum](../237-residual-momentum/) (the crash-dodging
cousin) -- this study is about the **crash** and its **repair**.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why momentum crashes, the Oct-2020 loser snap-back in plain language, the synthetic control, the flat real premium, and the vol-scaling repair |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 12-1 WML build, HAC inference, label-shuffle placebo, the crash table and drawdown episode, the bear-lookback regime sweep, the vol-scaling Sharpe/skew/drawdown comparison, gross-vs-net |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run:
[docs/results.md](docs/results.md) (fingerprint `e76dff230f36`, as-of 2026-06-26).

---

*Engine: [`momentum_crashes/`](momentum_crashes/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
