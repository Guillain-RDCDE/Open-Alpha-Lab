# Study 293 -- MVRV-Ratio

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | MVRV stretch does not predict next-month BTC returns: HAC *t* = **-0.15**, R^2 ~ 0 (n=140); the slope dies further in a price-momentum horse race (*t* = -0.70). The only suggestive number -- the over-heated band's -6.7%/mo forward return -- rests on **n = 4** months (the 2017 and 2021 tops). No robust *t* >= 2 on the real tape. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The contrarian "cash when over-heated" rule is **97% buy-and-hold**; its +2.2%/yr edge over simply holding is insignificant (HAC *t* = +0.59) and flips sign with the band threshold (trails at 3.0, converges at 4.0). Any CAGR is just long exposure to a 1000x survivor. |
| **Single-survivor bias** | ![Named](https://img.shields.io/badge/Single--survivor-8b949e?style=flat-square) | BTC is the surviving moonshot; MVRV (market cap / realized cap) is mechanically derived from its own price path; the contrarian bands are fitted to ~four cycles. All results are conditioned on that survival. |

> **In one sentence:** the MVRV contrarian gauge has the right *story* -- high MVRV before tops, low MVRV before bottoms -- but on the real BTC tape it shows zero out-of-sample predictive content (HAC *t* = -0.15), its "over-heated sell" band fires only four times, and the timing rule it implies is buy-and-hold that occasionally blinks to cash, making it a textbook **None / Mirage**.

## The claim

> *Does on-chain MVRV time Bitcoin tops and bottoms?*

## What we tested

The Mahmudov-Puell (2018) recipe: read **MVRV** (market value / realized value)
as a contrarian valuation gauge -- sell when MVRV is over-heated (>= 3.5), buy
when under-valued (<= 1.0). We join a curated month-end MVRV series to the real
BTC-USD monthly close and ask three honest questions: (1) does this month's MVRV
*stretch* predict next month's return in a HAC regression; (2) does it add
anything beyond BTC's own price momentum (a horse race); (3) does a contrarian
"step to cash when over-heated" timing rule beat **buy-and-hold** net of 30 bps
one-way costs -- the only benchmark that matters for a single trending asset.
We also report the average next-month return in each MVRV band and a
deterministic synthetic positive control that confirms the engine recovers a
planted contrarian link when one exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the MVRV story in plain language, the two trending lines everyone points at, why "sell the top" rests on four months, and why the timing rule is really just holding BTC |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC predictive regression, price-momentum horse race, per-band forward returns, contrarian timing vs buy-and-hold, cost/lag honesty, single-survivor caveat, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mvrv_ratio/`](mvrv_ratio/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
