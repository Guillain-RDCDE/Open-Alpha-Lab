# Study 215 — Big-Mac-PPP

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Gross **-2.00%/yr**, t = **-1.40**; the pooled regression slope is **+0.03%** (wrong sign, R2 = 0.93%, t = +1.35). Big Mac over-valuation does not predict depreciation at the one-year horizon — PPP is a decades-long anchor, not an annual signal. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross is already negative; annual rebalance makes costs negligible, but there is no edge to preserve. The strategy fails because the signal is absent, not because costs are high. |
| **Does the Big Mac index predict FX reversion?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | At 1-year horizons: **No** — the slope is wrong-signed. At multi-decade horizons: loosely yes, consistent with PPP half-lives of 3-5 years (Rogoff 1996). The Economist's index is a clever long-run anchor, not a tradable tactical indicator. |

> **In one sentence:** the Economist Big Mac index earns -2.00%/yr gross on a long-undervalued / short-overvalued FX portfolio from 2000-2023 — statistically indistinguishable from zero, with a regression slope that is *wrong-signed* and nearly zero, confirming Rogoff's PPP puzzle: currencies revert to purchasing power parity over decades, not years.

## What we tested

The Big Mac hypothesis: each July, the Economist publishes the % over/under-valuation of ten major currencies vs the dollar (EUR, GBP, JPY, CAD, AUD, CHF, SEK, NOK, MXN, BRL) implied by local Big Mac prices. We go **long the 3 most under-valued** and **short the 3 most over-valued** currencies (equal weight, entering August to respect publication lag, holding one year). We test against a **random-ranking control** (i.i.d. random ranks, 20 seeds) and run a **pooled cross-sectional OLS regression** (mis-valuation at year T → FX return Aug T to Jul T+1). 197 currency-year observations, 2000-2023. No look-ahead: the July snapshot is published before the August entry. Annual rebalance minimises cost impact.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | plain-language explanation of PPP, why the Big Mac is an amusing but imprecise signal, Rogoff's puzzle, why decades ≠ years |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | pooled OLS regression with slope & t-stat, portfolio t-stats, sub-period decay, synthetic positive control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`big_mac_ppp/`](big_mac_ppp/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
