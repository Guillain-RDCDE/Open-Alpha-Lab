# Study 509 -- Intermediate-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Does momentum really live in the *intermediate* past (t-12..t-7), not the recent past (t-6..t-1)?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- does either horizon carry a drift? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The intermediate long-short earns **-0.31%/yr** (HAC *t* = **-0.12**); the recent one **+0.25%/yr** (*t* = **+0.09**). Both sit inside their label-shuffle placebo distributions (p = **0.57** / **0.40**). Neither clears \|*t*\| >= 2 on this 46-name large-cap survivor basket. |
| **Tradability** -- does the spread pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Both sorts are flat gross and **negative net** of 5 bps/leg + 50 bps borrow: intermediate **-1.50%/yr**, recent **-0.94%/yr**, at ~29% monthly turnover. Nothing to monetise. |
| **"Does momentum live in the intermediate window?"** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The Novy-Marx (2012) prediction is that the intermediate sort beats the recent one. Here it does not -- the intermediate sort is, if anything, **marginally worse**. The horizon decomposition dissolves on a large-cap survivor universe. |

> **In one sentence:** Novy-Marx (2012) found the cross-sectional momentum premium lives in the
> intermediate window (twelve to seven months ago), not the recent one -- but on a 46-name
> large-cap survivor basket *neither* window carries a drift the tape can tell from zero
> (intermediate *t* = -0.12, recent *t* = +0.09, both placebo-confirmed), and the intermediate
> sort is marginally the worse of the two, so the echo is Busted here, the signal None and the
> trade a Mirage.

## What we tested

**Novy-Marx (2012), "Is momentum really momentum?"**: the momentum premium is driven by the
**intermediate** part of the formation window (t-12..t-7), not the **recent** part (here
t-6..t-1, where short-term reversal lurks). We build two monthly equal-weight dollar-neutral long-shorts
-- top tercile minus bottom tercile -- one ranked on the intermediate window and one on the
recent window, over a fixed 46-name large-cap survivor basket (yfinance daily adjusted-close,
2007-2025, 4780 days). Each carries **one execution lag** (enter one day after the signal),
turnover-based costs + short borrow, a **label-shuffle placebo**, a HAC *t*, and a synthetic
positive control that plants the premium *only* in the intermediate window to prove the engine
is faithful. The basket is **survivorship-biased** -- past losers that delisted (the natural
short leg) are absent, so results are upper bounds. *Distinct from [24 Stampede](../24-stampede/)
(plain 12-1 momentum) and [237 Residual-Momentum](../237-residual-momentum/) (residual-return
momentum): this study is specifically the **horizon decomposition**.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the intermediate-vs-recent idea in plain language, the synthetic control, the two real long-shorts side by side, honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the formation-window construction, HAC *t*, the label-shuffle placebo, net-of-cost spreads, equity curves, year-by-year, and the planted-premium control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run
(fp `e21fc3ab636a`): [docs/results.md](docs/results.md).

---

*Engine: [`intermediate_momentum/`](intermediate_momentum/). **Not investment advice** --
research & education. See [LICENSE](../../LICENSE).*
