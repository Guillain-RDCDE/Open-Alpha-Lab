# Study 416 — Rounding Bottom 🥣

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the saucer breakout predict anything? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The confirmed breakout earns **+1.51% at 20d / +4.07% at 60d** — but a **random day** in the same stocks earns **+1.14% / +3.31%**. Welch *t* of breakout-vs-base = **0.73 / 0.84**, date-shuffle placebo **p = 0.25 / 0.20**. The one-sample *t* vs zero (2.97 / 4.52) is just the equity drift. |
| **Tradability** — is there excess to deploy? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No edge over being long to harvest; 5 bps of cost nudges an already-zero excess negative. Nothing to size. |
| **An "accumulation" signal?** | ![Not supported](https://img.shields.io/badge/Accumulation_signal%3F-Not_supported-8b949e?style=flat-square) | Across base-window {60,90,120} and fit R² {0.45,0.55,0.70} the breakout-vs-base *t* **never clears 2** (−1.80 to +0.73). No regime-change footprint. |

> **In one sentence:** the rounding bottom is a beautiful shape that our mechanical detector finds easily — and that, on 22.5 years of SPY + large-caps, predicts *nothing* a random day wouldn't; the confirmed breakout's "+4% in 60 days" is the equity drift premium, indistinguishable from the base rate (Welch *t* < 1, placebo *p* ≈ 0.2), so the "smart-money accumulation" story is not supported.

## What we tested

Chart figures are partly subjective, so we test the **closest mechanical definition** and say so: at every bar we fit a least-squares **parabola** to the trailing ~90-day window and require positive curvature with a good R² fit (a genuine U), an **interior trough** (a dip in the middle, not a slide), enough **depth** below the rim, and a **confirmed breakout** — the first close back above the left-rim resistance. We then enter the **next day's open** (one execution lag) and measure the forward 10 / 20 / 60-day return across SPY + 29 US large-caps (2004→2026, yfinance daily OHLC). The decisive test is **not** the breakout return against zero (which only captures the drift every long position earns) but against the **base rate** — a random day in the same names — plus a **date-shuffle placebo** (random entry dates matched to the breakout count). A deterministic synthetic control plants the *shape* with and without continuation, confirming the engine fires on the figure yet only flags an edge when a real drift exists. Survivorship (30 names still trading in 2026) is named on the Signal axis — and only *flatters* a real effect, making the null conservative.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a saucer base is, why "buy the breakout" sounds smart, and why the gain is just the stock drifting up — in plain language, with a real detected saucer and the breakout-vs-random chart |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the parabola detector, forward 10/20/60-day returns, the *t*-vs-zero trap vs the Welch *t*-vs-base-rate, a 5,000-draw date-shuffle placebo, knob robustness, and a shape-vs-continuation synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`rounding_bottom/`](rounding_bottom/). Detector = least-squares parabola + interior trough + confirmed rim breakout (one mechanical surrogate for a subjective figure). Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
