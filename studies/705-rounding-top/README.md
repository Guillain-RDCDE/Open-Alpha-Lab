# Study 705 — Rounding Top 🌄

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the dome breakdown predict a decline? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The confirmed breakdown, shorted, earns **−1.61% at 20d / −4.39% at 60d** — but **shorting a random day** in the same stocks earns **−1.14% / −3.32%** (the equity drift working against every short, not the pattern). Welch *t* of breakdown-vs-base = **−0.96 / −1.38**, date-shuffle placebo **p = 0.83 / 0.89** (random shorts beat the pattern 83–89% of the time). |
| **Tradability** — is there excess to deploy? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No excess over a random short to harvest; 5 bps of cost plus a 30 bps/yr borrow rate just push an already-zero edge further negative. Nothing to size. |
| **A "distribution" signal?** | ![Not supported](https://img.shields.io/badge/Distribution_signal%3F-Not_supported-8b949e?style=flat-square) | Across base-window {60,90,120} and fit R² {0.45,0.55,0.70} the breakdown-vs-base *t* **never clears 2 in magnitude** (−1.03 to +0.75) — one config even points the *wrong* way for the bearish thesis. No regime-change footprint. |

> **In one sentence:** the rounding top is a beautiful shape our mechanical detector finds easily — and on 22.5 years of SPY + large-caps, shorting the confirmed breakdown does no better than shorting a random day in the same stocks (Welch *t* < 1.4, placebo *p* ≈ 0.8–0.9), so the "distribution → markdown" story is not supported — the exact mirror-image conclusion of Study 416's rounding bottom.

## What we tested

Chart figures are partly subjective, so we test the **closest mechanical definition** and say so: at every bar we fit a least-squares **parabola** to the trailing ~90-day window and require negative curvature with a good R² fit (a genuine dome), an **interior peak** (a roll-over in the middle, not a monotone climb), enough **height** above the rim, and a **confirmed breakdown** — the first close back below the left-rim support. We then **short at the next day's open** (one execution lag) and measure the forward 10 / 20 / 60-day return across SPY + 29 US large-caps (2004→2026, yfinance daily OHLC). The decisive test is **not** the breakdown return against zero (which only captures the equity-drift cost every short position pays) but against the **base rate** — shorting a random day in the same names — plus a **date-shuffle placebo** (random entry dates matched to the breakdown count). A deterministic synthetic control plants the *shape* with and without a real continuation, confirming the engine fires on the figure yet only flags an edge when a genuine decline exists (checked clean across 20 seeds). Survivorship (30 names still trading in 2026) is named on the Signal axis — here it works *against* the bearish claim, not for it, since the worst confirmed outcomes are structurally absent from a 2026-survivors panel. Short trades are charged one-way costs plus an annualized borrow rate.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a dome distribution is, why "short the breakdown" sounds smart, and why the loss is just the cost of fighting the market's own upward drift — in plain language, with a real detected dome and the breakdown-vs-random chart |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the parabola detector, forward 10/20/60-day SHORT returns, the *t*-vs-zero trap vs the Welch *t*-vs-base-rate, a 5,000-draw date-shuffle placebo, knob robustness, borrow/cost accounting, and a shape-vs-continuation synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`rounding_top/`](rounding_top/). Detector = least-squares parabola + interior peak + confirmed rim breakdown (one mechanical surrogate for a subjective figure) — the bearish mirror of [416-rounding-bottom](../416-rounding-bottom/). Basket is **survivors**, named on the Signal axis and working against, not for, the bearish claim. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
