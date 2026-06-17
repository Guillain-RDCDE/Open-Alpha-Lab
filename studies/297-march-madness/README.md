# Study 297 — March-Madness

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does March Madness distract the market lower?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | In-window mean is **+0.42 bps/day** — *above* the rest of the tape, not below (Welch t = **+0.09**); permutation p = **0.53**; volatility ratio **0.85** (calmer, not chaotic); the sub-period gap flips sign. The folklore predicts a drag; the data shows a non-event pointing the wrong way. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A lagged, cost-charged "hold cash during the bracket" strategy ties buy-and-hold (**+0.02 pp/yr**, HAC t = **−0.10**). No vehicle, no edge. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The office-pool productivity-loss headlines are real; the market impact is not. The marginal price-setter in S&P 500 futures is not filling out a bracket. |

> **In one sentence:** the NCAA tournament window earns a hair *more* than the rest of the year, with no extra volatility — the "distraction" never reaches anyone who sets prices.

## What we tested

We hardcode every NCAA Division I men's tournament window (first-round Thursday →
championship Monday) for the 64+ team era, 1985–2025 (2020 cancelled), in `data.py`,
label each ^GSPC trading day as in-window or outside, and compare:
mean daily **price** return (Newey-West HAC t), realised volatility (variance ratio
+ Levene), and a tradable **avoidance** strategy that holds cash during the window
with a one-trading-day execution lag and one-way costs on every switch. A
random-placement permutation null asks whether *these particular* ~5% of days are
special. The synthetic positive control confirms the machinery detects a planted
drag when one exists — the real tape has none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the in/out comparison, the "chaos?" check, plain-English verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t, Levene variance test, permutation null, lagged+cost avoidance, power, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`march_madness/`](march_madness/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
