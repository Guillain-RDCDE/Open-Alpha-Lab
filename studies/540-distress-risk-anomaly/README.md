# Study 540 — Distress-Risk-Anomaly 💀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the long-safe / short-distressed spread real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On the headline 2024-06 → 2026-06 window the spread is the **wrong sign**: distressed names earned **+82.7%** vs safe **+34.1%**, spread **−48.6%** (two-sample *t* **−2.64**, placebo *p* 0.033). The firm-level slope is **positive** (*t* +1.99). And the sign **flips across windows** — no stable *t* ≥ 2 for the puzzle. Survivorship-biased basket. |
| **Tradability** — does the spread pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 39-name survivor basket, annual-rebalance, with the distressed leg you'd short being exactly the expensive-to-borrow tail. The trade is the wrong sign before costs (gross **−48.6%**, net **−50.8%** after 5 bps/leg + 100 bps borrow) — there is nothing to harvest here. |
| **"Distress puzzle on the tape?"** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | The puzzle **does** appear (distressed underperform, firm slope-*t* **−2.02** / **−1.46**) in the 2021-23 and 2022-24 windows, but **inverts** in 2023-25 and 2024-26 (the AI/high-beta melt-up rewarded leverage and volatility). Present in 2 of 4 windows, gone in the other 2. |

> **In one sentence:** the famous Campbell-Hilscher-Szilagyi distress puzzle — distressed firms earning *anomalously low* returns — doesn't survive a 39-name blue-chip survivor basket: over 2024-26 it **inverts** (the most-distressed names roughly doubled the safe ones, *t* −2.64 the *wrong* way) and its sign flips across windows, because the survivor basket strips out the bankruptcies that drive the real effect and a high-beta melt-up rewarded exactly the levered, volatile names the puzzle says should lose.

## What we tested

The **distress puzzle** (Campbell, Hilscher & Szilagyi 2008; Dichev 1998): a failure-prediction
model flags the firms most likely to go bankrupt, and those firms go on to earn the *lowest*
returns — the inverse of a risk premium. We build a CHS-style **distress score** (high leverage −
profitability + high equity volatility, each z-scored across the basket), sort a fixed 39-name
large-cap survivor basket into terciles, and test whether the safe names beat the distressed ones:
a two-sample *t* on the safe-minus-distressed forward-return spread, a **label-shuffle placebo**
null, a firm-level cross-sectional regression (whose *sign* is the puzzle), costs + a punitive
short borrow, a **four-window robustness** sweep, and a deterministic, seed-robust synthetic
positive control that plants the puzzle and proves the engine catches it. *Distinct from the
distress **scores** — [123 Altman-Z](../123-altman-z/) and [230 Ohlson-O-score](../230-ohlson-o-score/)
replicate the static bankruptcy classifiers; this study is the **return anomaly and its sign**.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the distress puzzle is, why "distressed = lowest return" is so weird, and why on this basket the distressed names *won* |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the tercile sort with a two-sample *t*, the placebo null, the firm-level slope, the four-window sign-flip, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted real-data run (39 survivors, scored 2024-06, forward to 2026-06, panel fp
`0835f4217788`) is in [docs/results.md](docs/results.md); the offline machinery proof runs on the
deterministic synthetic world in [`distress_risk_anomaly/data.py`](distress_risk_anomaly/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`distress_risk_anomaly/`](distress_risk_anomaly/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
