# Study 410 — Cup & Handle ☕

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout predict a run? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **443** confirmed cup-with-handle breakouts (SPY + 29 large-caps, 21.4 yrs), the forward return *in excess of each name's own base rate* is noise: the only near-2 horizon (**10-day, one-sample t = 2.14**) is **busted by the same-tape placebo (p = 0.34)** — random dates on the same drifting tape beat it a third of the time — and 20–40-day excess goes flat then **negative**. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge to tax. The positive **raw** forward return (+0.5% to +1.5%) is just the market beta you were always paid for; costs are a footnote; the SPY-specific version rests on **8** events. Nothing to deploy. |
| **"Beats buy-and-hold"?** | ![Busted](https://img.shields.io/badge/Beats_buy--and--hold%3F-Busted-8b949e?style=flat-square) | "The stock goes up after the breakout" is **true and irrelevant** — it goes up by exactly its base rate. Subtract buy-and-hold and the edge vanishes (placebo p ≈ 0.34). The figure is a Rorschach test on an up-drifting tape, not a launch pad. |

> **In one sentence:** O'Neil's cup-with-handle breakout *does* tend to be followed by a rising price — but only by the same amount the stock would have risen on any random day, so once you subtract buy-and-hold the edge evaporates (the lone 10-day t = 2.14 is busted by a same-tape placebo, p = 0.34), and the SPY-only version rests on a mere 8 events.

## What we tested

Chart figures are partly subjective, so we wrote down the **closest mechanical definition** of the
cup-with-handle and tested *that* (saying so loudly): swing-pivot **cup** (12–50% deep, rims within
6%, U-shaped) + a shallow **handle** (≤15% pullback) + a **confirmed breakout** (first close above
the rim "pivot"). On daily auto-adjusted OHLC for **SPY + 29 US large-caps** (2005→2026), we enter
the next close after each breakout (one documented lag) and measure the forward 5/10/20/40-day return
as an **excess over each name's own buy-and-hold base rate** — the honest version of the hook
*"does it beat buy-and-hold?"* The Signal axis pairs a one-sample/HAC *t* with a **same-tape
random-date placebo** (the arbiter that exposes drifting-tape false positives); a deterministic
synthetic control with a *planted* post-breakout drift proves the detector + inference can bank a
real edge when one exists. The basket is **survivors** — named on the Signal axis (the bias tilts
the test *for* the figure, and it still fails).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a cup-with-handle is, why "it went up after the breakout" is a trick of the base rate, and why subtracting buy-and-hold kills the magic — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the objective detector, forward excess returns by horizon, the one-sample *t* vs the same-tape placebo, detector-strictness robustness, the SPY small-*n* mirage, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cup_and_handle/`](cup_and_handle/). Detector is one transparent mechanical definition of the figure — not the only one a human would draw. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
