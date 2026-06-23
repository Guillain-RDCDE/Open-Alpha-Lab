# Study 398 — Entropy-Efficiency 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does low entropy predict returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | SPY returns sit near the **white-noise ceiling** (permutation entropy mean **0.98**). The low-entropy regime's forward edge is **null at 1–5 days** and only **borderline at 21 days** (HAC *t* = **2.05**) — and that lone number **fails the clustering-aware bootstrap** (*p* = **0.30**), **vanishes under a second entropy measure** (sample-entropy *t* = **0.28**), and is flat across cutoffs. A non-robust point estimate, not an edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Long-in-low-entropy / flat earns **+1.43%/yr net at Sharpe 0.19** vs **+10.63%/yr at Sharpe 0.56** for buy-&-hold. Costs are trivial; the rule sits out **81%** of days and harvests drift at a **worse** Sharpe. Timing on the entropy clock turns 10.6%/yr into a 1.4%/yr stub. |
| **Predictable = profitable?** | ![Busted](https://img.shields.io/badge/Predictable_%3D_profitable%3F-Busted-8b949e?style=flat-square) | The pitch conflates *predictability of shape* with *expected return*. The synthetic control proves the gap: a far-more-predictable regime with **zero** planted edge pays **nothing** (*t* = −0.94). Low entropy buys forecastable *shape*, not return. |

> **In one sentence:** the "efficiency clock" idea — that a drop in the market's entropy opens a tradable window — fails on SPY because daily returns are *almost always* near maximum entropy, the bottom-entropy days carry no forward edge that survives a clustering-aware null or a change of estimator, and even the predictable-by-construction regime of our synthetic control pays nothing without a planted edge: predictability of shape is simply not expected return.

## What we tested

We build an **entropy clock** on SPY: a causal **rolling permutation entropy** of daily returns (Bandt-Pompe, 60-day window) measuring how *disordered* the recent ordinal pattern of returns is, with a slower **sample-entropy** (Richman-Moorman) cross-check. Days in the **bottom 20%** of the (expanding, no-look-ahead) entropy distribution are tagged the "low-entropy / predictable" regime. We then ask the only two questions that matter: do forward 1/5/21-day SPY returns differ between low- and high-entropy days by more than luck — judged by a **HAC (Newey-West) *t*** and a **stationary block-bootstrap** null that respects how low-entropy days *cluster* — and does any difference survive a 1-day execution lag and costs as a long/flat strategy. A deterministic synthetic control toggling a random regime against a *structured* (low-entropy) one with a **planted-edge knob** confirms the engine recovers a real edge and refuses to invent one when predictability comes without payoff. (Same regime question, different lens, as [Study 397](../397-hurst-regime/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "market entropy" even means, why the tape is almost always near-random, and why "more predictable" doesn't mean "more profitable" — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | permutation & sample entropy on returns, the low-vs-high regime split, a HAC *t* + block-bootstrap null sized to the clustering, a cost/Sharpe ledger, and a synthetic faithful-engine / predictability-without-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`entropy_efficiency/`](entropy_efficiency/). The entropy clock is computed **causally** (window ends in the past) and judged with a **clustering-aware** null. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
