# Study 448 — Point & Figure 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the count target get hit? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | The horizontal-count target is hit **56.5%** of the time (double-top buys) and **38.9%** (double-bottom sells) vs **40.8% / 26.9%** for a target placed the **same distance** away on random days — a **+15.6 pp / +12.0 pp** edge at one-sample *t* = **6.56 / 4.41**. The edge holds on **both** sides (so it's the count, not just drift), is robust to box size, and a synthetic null shows no manufactured edge. Breakouts genuinely continue. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Buy net **+3.27%**/signal (HAC *t* = 6.14) but sell net **−1.03%** (*t* = −2.02). Same chart edge, **opposite** P&L: the buy profit is the 26-year equity drift, the short leg bleeds. The full method nets ~zero before costs — beta in a P&F costume, not a tradable count edge. |
| **"Do the count targets get hit?"** | ![Confirmed](https://img.shields.io/badge/Do_targets_get_hit%3F-Confirmed-8b949e?style=flat-square) | The literal folklore claim is **true** — robustly, on both directions and every box size (1.5% / 2% / 3%). The chart's number is real even though no tradable edge rides on it. |

> **In one sentence:** Point & Figure's signature promise — the horizontal-count price target — is one of the rare chart claims that is *literally true*: the target gets hit far more than a same-distance random target (+15.6 pp, *t* = 6.6), on both the buy and the sell side, so it's a genuine continuation effect and not a drift artefact — but the **money** is one-directional (longs profit, shorts lose by the same amount), so as a tradable rule it's just the bull-market drift wearing a 100-year-old costume.

## What we tested

We encode the tightest **objective** Point & Figure a practitioner would accept — a fixed box size (2% of median price), the classic **3-box reversal**, the standard **double-top buy / double-bottom sell** signals, and the **horizontal-count** target (`T = breakout ± width × 3 × box`, stop = the breakout column's far edge) — and ask one falsifiable question: does the count target get hit before the stop **more often than a target placed the same distance away on random days** of the same instrument? The Signal axis tests the per-signal hit indicator against that same-distance baseline with a one-sample *t* and a 2,000-draw random-target placebo, on both the buy and sell sides separately (the symmetry test that distinguishes a real count from bull-market drift). Tradability charges one-way costs × turnover plus short-leg borrow on per-signal P&L. A deterministic synthetic control with a *planted* continuation confirms the engine is faithful and that a pure random walk cannot fake a hit-rate edge. Four daily tapes (SPY, ^DJI, AAPL, GLD, 2000–2026).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a P&F chart is, what the count target means, why the targets really do get hit, and why the money is just the rising market — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 3-box-reversal column engine, the horizontal-count target, hit-rate vs a same-distance baseline, the buy/sell P&L split, HAC *t* on per-signal returns, the placebo, and a planted-continuation synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`point_and_figure/`](point_and_figure/). Box = 2% of median price, 3-box reversal; targets graded HIT-before-STOP against a same-distance random-target baseline. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
