# Study 221 — Mayer-Multiple

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Long-only strategy HAC *t* = **+0.19** (gross); the 30-day forward-return band test shows a significant effect (*t* = −2.84) but **inverted** from the claim — cheap (MM < 1) underperforms neutral, not outperforms. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-only earns **+1.5%/yr** gross vs buy-and-hold **+34.0%/yr**; negative expected return net of 50 bps cost; strategy is flat 63% of the time, missing BTC's dominant trend. |
| **The claim: MM < 1 = cheap?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Mean-reversion intuition applied to a momentum asset. Decile analysis shows the highest-MM decile (1.70–3.93) produces +10.9% 30-day forward return; the lowest-MM decile (0.48–0.74) produces +0.5%. The "cheap" zone is a downtrend filter, not a valuation anchor. |

> **In one sentence:** the Mayer Multiple names Bitcoin's downtrend "cheap" and its uptrend "expensive" — applying static mean-reversion thinking to a momentum-driven asset; the long-only strategy earns +1.5%/yr gross (t = +0.19) while buy-and-hold earns +34.0%/yr, and the only statistically significant band effect runs in the *wrong* direction (t = −2.84, cheap underperforms neutral).

## What we tested

> Does the Mayer Multiple (price / 200-day MA) tell you when Bitcoin is cheap?

The Mayer Multiple (popularised by Trace Mayer in 2017–2018) is `price / 200-day simple moving average`. The canonical recipe: accumulate when MM < 1.0 (price below 200-day SMA — "cheap"), trim when MM > 2.4 (price more than 2.4× the 200-day SMA — "expensive"). We implement the long-only strategy (long when MM < 1, flat otherwise) and measure its forward-return distribution across all three bands. We further break MM into deciles to check whether any monotone relationship between MM and 30-day forward returns exists.

**Key finding:** The decile analysis shows the opposite of what the claim predicts. The lowest MM decile (most "cheap") produces the *lowest* 30-day forward returns; the highest MM decile (most "expensive") produces the *highest*. BTC is a momentum asset — price below its 200-day SMA signals a downtrend, not undervaluation. The band t-stat (cheap vs neutral) is −2.84: statistically significant, but pointing the wrong way. The long-only strategy t-stat is +0.19: indistinguishable from zero.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Mayer Multiple explained, why "cheap" is a downtrend not a bargain, cumulative return vs buy-and-hold, the decile reveal |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat dissection, band conditional return tables, forward-return decile chart, cost sweep, positive control (signal_strength sweep) |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mayer_multiple/`](mayer_multiple/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
