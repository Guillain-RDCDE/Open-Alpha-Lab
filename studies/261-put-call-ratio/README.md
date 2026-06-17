# Study 261 -- Put-Call-Ratio

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Next-month ^GSPC return on the expanding put/call z-score has HAC *t* = **-0.16** (flat, wrong sign). Months *after* extreme fear return **+0.69%** vs **+0.84%** after calm months (Welch *t* = -0.18, p = 0.86) -- the *opposite* of the contrarian claim. A random in-market selection beats the fear-timed one **61.5%** of the time. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The extreme-fear timing rule earns **+2.2%/yr** vs buy-and-hold's **+9.6%/yr** with a *lower* Sharpe (0.20 vs 0.64), because it sits in cash ~74% of months and misses the bull run. Net of 5 bps switch costs the story is unchanged -- the edge simply does not exist. |
| **Sample / survivorship** | ![Named](https://img.shields.io/badge/Sample--limited-8b949e?style=flat-square) | ^GSPC is survivorship-clean, but the put/call history is short (~23yr, 276 months) and the "extreme" tail rests on a handful of clustered crisis spikes; price-only returns understate buy-and-hold's true dividend-inclusive edge. |

> **In one sentence:** the CBOE put/call ratio is a genuine fear gauge that spikes exactly where you'd expect (2008, 2018, 2020), but as a contrarian market-bottom timer it is worthless -- extreme-fear months are followed by *slightly worse* returns than calm months (Welch *t* = -0.18), the predictive regression is flat (*t* = -0.16), and the rule earns +2.2%/yr against the index's +9.6%/yr by hiding in cash three-quarters of the time.

## The claim

> *Does the CBOE put/call ratio time market bottoms?*

## What we tested

Read the hardcoded CBOE total put/call ratio at month-end; when it prints in the
top quintile of its expanding history (extreme fear), hold the S&P 500 for the
*next* month, otherwise hold cash (one full month of execution lag). We pin the
result against (a) always-invested buy-and-hold, (b) a continuous HAC predictive
regression of next-month return on the put/call z-score, (c) the conditional
next-month return after fear vs calm with a Welch test, (d) a random-timing
permutation null, and (e) a sub-period breakdown from 2006 to 2025. A
deterministic synthetic positive control (planted contrarian beta) confirms the
engine recovers a contrarian edge when one genuinely exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the fear-gauge story, where the ratio spikes, buying-the-fear vs owning the index, and the killer fact that extreme-fear months are not better than calm ones |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC predictive regression, gross/net timing returns, the Welch conditional test, a random-timing null, sub-period stability, and a synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`put_call_ratio/`](put_call_ratio/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
