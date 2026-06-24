# Study 412 — Symmetrical Triangle 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout predict? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **291** confirmed breakouts on **21.5 years** of daily tape, the forward return's **excess over a matched random-day base rate never clears |*t*| = 0.5** at any horizon (1/5/20/60d; largest HAC *t* on the excess = **0.45**). Win-rate ≈ **50%**, permutation *p* ≥ **0.33**. A synthetic control with a *planted* continuation lights up (*t* = **5.79**), so this is a real null, not a blind harness. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no gross edge to charge costs against; the net column is negative wherever the signal is weak. Nothing survives costs because nothing survives **zero**. |
| **"Continuation figure"?** | ![Busted](https://img.shields.io/badge/Continuation%3F-Busted-8b949e?style=flat-square) | The one big number — up-breakouts **+3.9%** vs down-breakouts **−3.4%** at 60 days — is **market beta over a long window**, not the triangle: it collapses to excess *t* = **0.09** the moment you subtract a matched random-day return. |

> **In one sentence:** on the closest mechanical definition (swing-pivot trendlines + a real range contraction + a confirmed price-pierce breakout), a symmetrical-triangle breakout is a **50/50 coin flip** — its excess over a random day never clears |*t*| = 0.5, and the impressive-looking up/down asymmetry at 60 days is just the market's own drift, not the chart.

## What we tested

Technicians call the symmetrical triangle a **continuation** figure: price coils into converging lower-highs and higher-lows, then breaks out *in the trend's direction* and runs. Chart figures are partly subjective, so we test the **closest mechanical definition** and say so: an objective detector fits least-squares trendlines through `scipy.find_peaks` swing pivots, requires the lines to **converge** with roughly symmetric slopes **and** the range to genuinely **contract** toward an apex, then takes the **confirmed** breakout (the close piercing the projected trendline). We measure the signed forward 1/5/20/60-day return — entered **one day after** confirmation (no look-ahead) — against a **matched random-day base rate**, with a Newey-West HAC *t* on the *excess*, a 20,000-draw permutation placebo, costs, and an up-vs-down direction split. A deterministic synthetic control with a *planted* post-breakout continuation confirms the engine can detect an edge when one exists. The basket is **survivors** (named on the Signal axis).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a symmetrical triangle is, why "continuation" sounds compelling, how a confirmed breakout does no better than a random day, and why the scary 60-day asymmetry is just the market drifting — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical detector, forward returns vs a random-day base rate, HAC *t* on the excess + a permutation placebo, the beta-not-pattern direction split, costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`symmetrical_triangle/`](symmetrical_triangle/). Detector is the closest **mechanical** definition of a subjective figure — a hand-drawn triangle could differ. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
