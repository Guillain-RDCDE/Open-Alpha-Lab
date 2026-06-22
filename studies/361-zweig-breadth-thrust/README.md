# Study 361 — Zweig Breadth Thrust 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a thrust predict a bull move? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The folklore is partly real — 12-month win-rate after a thrust is **91%** — but the *excess* over the **81%** unconditional base rate is small, **fails t ≥ 2** (Welch *t* = **0.94**, placebo *p* = **0.21**), and **flips negative at 3 months**. A positive-but-insignificant point estimate, not an edge. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs are negligible, but the proxy fires **~22 times in 31 years** (the true NYSE version: ~once a decade). A signal you act on under once a year, whose excess is inside the noise, is **not a NAV-scale strategy**. |
| **"Never wrong"?** | ![Busted](https://img.shields.io/badge/Never_wrong%3F-Busted-8b949e?style=flat-square) | A **sample-size + base-rate illusion**: a dozen-ish events in a market that rises most years anyway *will* look "never wrong" by luck. **Study 167 (Hindenburg) inverted** — same rare-breadth mirage, opposite costume. |

> **In one sentence:** the Zweig Breadth Thrust's "never wrong" reputation is what a rare signal looks like in a market that rises most years anyway — on a transparent breadth proxy its 22 lifetime fires give a high *win-rate* but an excess return statistically indistinguishable from drawing 22 random dates (t = 0.94), so it is real-as-lore, weak-as-edge, and undeployable as a strategy.

## What we tested

True NYSE advance/decline breadth isn't on yfinance, so we **construct a transparent proxy**: from a fixed **40-name** large-cap basket we compute each day's **advance ratio** (advancers / (advancers + decliners)) — a narrower, noisier stand-in for Zweig's NYSE breadth, labelled a proxy throughout. A **Breadth Thrust** fires when the 10-day EMA of that ratio climbs from below **0.40** to above **0.615** within ~10 trading days (Zweig's exact rule). Over **31.5 years** (1995–2026, **7,918** days) the proxy fires **22** times; we measure forward 3/6/12-month SPY returns after each thrust vs the unconditional base rate, with a 1-day entry lag, a Welch *t*, and a 20,000-draw **placebo** null sized to the event count. A deterministic synthetic control with *injected* thrusts confirms the engine is faithful **and** that ~a dozen events can't reach significance unless the planted edge is implausibly large. (Same rare-breadth / sample-size mirage as [Study 167](../../167-hindenburg-omen/), inverted.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "never wrong" is really "the market mostly goes up," what a thrust is, and why a dozen events can't be a strategy — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | thrust detection on the breadth proxy, conditional vs unconditional forward returns, a Welch *t* + placebo randomization null, costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`zweig_breadth_thrust/`](zweig_breadth_thrust/). Breadth here is an explicit **proxy** (a 40-name basket), not true NYSE A/D. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
