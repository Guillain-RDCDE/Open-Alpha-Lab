# Study 210 — Crypto-Trend

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Sharpe +0.902 vs BH +0.632; max DD −70.10% vs BH −83.40% (+13 pp). Timing vs random control: *t* = **+2.63** (above the inference bar). Timing vs BH alone: *t* = +0.32 (directional, not certified — 11-year tape is short). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | ~7 switches/yr; Sharpe above BH at 50 bps one-way (+0.853 vs +0.632). Key fragilities: short history (74 switches total), regime-dependence (whipsaws in choppy 2021-2022 top), behavioral challenge of holding cash during BTC bull runs. |
| **Drawdown shield?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Vol 67% → 51%; max DD −83% → −70%. Random-timing control cannot replicate (−83.28%): genuine timing skill, not exposure reduction. |

> **In one sentence:** the 200-day MA timing rule on Bitcoin is a genuine drawdown shield — it sidesteps most of BTC's brutal −80%+ bear markets through real timing skill (confirmed vs a random-timing control), at almost no transaction-cost penalty, but the 11-year history is too short to statistically certify the Sharpe improvement vs buy-and-hold, and the 2021-2022 choppy top shows the rule's whipsaw risk.

## What we tested

The folk recipe: hold Bitcoin when its daily close is above its 200-day simple moving average; move to cash (earning a 4%/yr flat proxy) when it falls below. Re-evaluated daily. No short selling. This is the Faber (2007) rule applied to crypto, tested on 4,290 days of BTC-USD (2014-09-17 to 2026-06-15). The key test: is the drawdown benefit from *timing* (picking the right days out) or merely from *reduced exposure* (being invested fewer days)? A random-timing control with the same 58.6% in-market fraction answers this: the random control has almost the same −83% drawdown as buy-and-hold, while the SMA rule achieves −70%. The timing t-stat vs the random control (+2.63) clears the inference bar; the t-stat vs buy-and-hold alone (+0.32) does not — the history is too short.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, the drawdown shield in plain language, the random-timing test, sub-era breakdown, cost insensitivity |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, the random-control test, sub-era Sharpe/DD, cost sweep, synthetic positive/negative controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`crypto_trend/`](crypto_trend/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
