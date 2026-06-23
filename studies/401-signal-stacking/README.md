# Study 401 — Signal-Stacking 🧮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge in the composite? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A **methods demo, not a backtest.** The lesson (stacking lifts the information coefficient only like **√K**, and only for *real, decorrelated* signals) is proven on a controlled synthetic panel; the real SPY stack is an honest illustration whose equal-weight composite is **negative** (Sharpe **−0.26**, permutation *p* = **0.98**, beaten by buy-and-hold +0.62). That can never back a `REAL` stamp; **no real stacking edge is claimed.** |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | On pure noise the composite Sharpe sits in its own permutation null (*p* = **0.38**, IC lift **0.83** — no √K); redundant signals (corr 0.9) plateau at lift **1.05**; the only positive backtest is the **snooped** one, and its in-sample Sharpe evaporates out of sample (the SPY snooper even picks two pure decoys). |
| **A composite edge?** | ![Busted](https://img.shields.io/badge/A_composite_edge%3F-Busted-8b949e?style=flat-square) | The "empire of weak signals" is **√(effective breadth) in Greek letters**. It compounds only when signals are both *real* and *different* (positive control: lift **3.07 ≈ √10**, *p* = 0.000). Absent that — noise, redundancy, or a real tape with no decorrelated edge — it is luck plus in-sample selection. Same shape as [343–350](../343-data-mining-roulette/) and [399](../399-kalshi-efficiency/). |

> **In one sentence:** stacking K weak signals into a Sharpe-weighted composite z-score raises the information coefficient only like **√K**, and only when the signals carry genuine **decorrelated** edge — so on a pure-noise stack the "ζ-field" times nothing (permutation *p* = 0.38), redundant signals plateau (lift 1.05), the real SPY composite actually loses to buy-and-hold, and the gorgeous equity curve is in-sample selection that decays the moment it leaves the data it was mined from.

## What we tested

The viral pitch: take K signals "none significant alone, most hovering around 52%," normalise them into one composite z-score "weighted by historical Sharpe," and harvest "an edge retail has never seen." We build that exact machine and judge it with three honest arbiters the thread never applies: the **√K law** (composite information coefficient vs the average single signal — Grinold & Kahn's fundamental law of active management), a **permutation test** (is the composite better than its own signals reshuffled into noise?), and an **in/out-of-sample snoop split** (select signals + Sharpe-weights on the first half, pay on the second). Two knobs decide whether the magic can happen: `signal_ic` (0 = a pure-noise null, 0.05 = a real ~52% edge) and `signal_corr` (decorrelated → √K, redundant → a low ceiling). A deterministic synthetic null and positive control bracket the truth; the same 12-signal stack on a real SPY tape (cache-first, illustrative) shows what happens when there is no decorrelated edge to compound. (Same family as the research-method demos [343–350](../343-data-mining-roulette/) and [399](../399-kalshi-efficiency/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "fifty weak signals stack into magic" is half-true, what "decorrelated" secretly does, and why the beautiful curve is selection — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the composite z-score, the √K information-coefficient law, a permutation null on the composite Sharpe, the in/out-of-sample snoop tax, and the redundancy ceiling — with a synthetic null/positive control and the real SPY stack |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`signal_stacking/`](signal_stacking/). The real SPY tape is an explicit **illustration** of the machinery — a methods demo, not a tradable backtest. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
