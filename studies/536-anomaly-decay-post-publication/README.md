# Study 536 — Anomaly-Decay-Post-Publication 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a tradable edge here? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A methodology demo, not an anomaly hunt. The momentum decay is real, but the *point* is the post-publication leg **falls below t = 2**: 12-1 momentum drops from **+14.8%/yr (t = 3.14)** pre-1993 to **+3.7%/yr (t = 1.11)** after. No new signal is claimed, and the **survivor** basket inflates the pre-publication numbers it leans on. |
| **Tradability** — could you deploy any of it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of one-way costs × turnover (+ borrow), **every positive post-publication leg is sub-t = 2** (momentum +2.7%/yr at t = 0.82); the highest-turnover anomaly (1-month reversal, 0.67/mo) goes **negative**. The "edge" you would have wanted to trade is exactly the part that decayed away. |
| **Do anomalies decay after publication?** | ![Confirmed](https://img.shields.io/badge/Decay%3F-Confirmed-8b949e?style=flat-square) | Median post/pre ratio **0.31** across four classic anomalies; momentum **−75%** (label-shuffle placebo *p* = **0.039**), short-term reversal **−98%**. The McLean-Pontiff ~50% half-life shows up even on a tiny survivor basket — if anything stronger. |

> **In one sentence:** rebuild four textbook anomalies on a survivor basket and split each at its publication year, and you watch the McLean-Pontiff decay happen in front of you — 12-1 momentum more than halves (t = 3.14 → 1.11, placebo *p* = 0.039), short-term reversal all but vanishes (t = 1.93 → 0.07), the median post/pre ratio is 0.31, and net of costs not one post-publication leg clears t = 2 — so the lesson is defensive, not a trade.

## What we tested

We take four classic cross-sectional anomalies with real academic publication dates — **12-1 momentum** (Jegadeesh-Titman 1993), **low volatility** (Ang et al. 2006), **1-month reversal** (Jegadeesh 1990 / Lehmann 1990), and **3-year reversal** (De Bondt-Thaler 1985) — and rebuild each as a price-only, dollar-neutral **top-tercile-minus-bottom-tercile** monthly long-short on a fixed **40-name large-cap survivor basket** (entering the next month's return, one execution lag, no look-ahead). We then **split each anomaly's series at its publication year** and compare the mean monthly return before vs after, with a one-sample *t* on each leg, a 5,000-draw label-shuffle placebo on the pre/post boundary, and one-way costs × turnover plus short-leg borrow. A deterministic 20-seed-averaged synthetic control with a *planted* decay confirms the split engine recovers a known pre/post step and that a dead post leg cannot fake significance. **Survivorship** (the basket is names still trading in 2026, which inflates the pre-publication half) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "an anomaly decays after publication" means, why arbitrage erodes a public edge, and the momentum-halving picture in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | tercile long-shorts, the pre/post split with one-sample *t* per leg, the label-shuffle placebo, costs × turnover + borrow, the survivorship caveat, and the planted-decay synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`anomaly_decay/`](anomaly_decay/). Anomalies are price-only proxies (momentum, vol, reversal) split at their real publication years. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
