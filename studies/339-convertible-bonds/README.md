# Study 339 — Convertible-Bonds 🪢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the payoff actually convex? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The quadratic convexity coefficient is **γ = −1.90** (HAC *t* = **−2.96**), bootstrap CI **[−3.48, −0.87]** *wholly below zero* — the payoff bends the **wrong way** (slightly concave), not the advertised smile. Holds in both halves (2009–15 & 2016–26). |
| **Tradability** — better than a plain blend? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A beta-matched **63% SPY / 37% AGG** blend *edged* CWB on excess Sharpe (**0.879** vs **0.817**) at a smaller drawdown (**−22%** vs −32%); the difference is a coin flip (*t* = 0.71). You pay CWB's ~**0.40%/yr** fee for a shape you can replicate for ~nothing. |
| **"Equity upside with bond downside"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Up-capture **0.66** ≈ down-capture **0.65** (asymmetry +0.02). In the March-2020 crash CWB fell **−13.1%** — *more* than SPY (−12.5%) and far more than the blend (−7.7%). No floor. |

> **In one sentence:** the convexity that *defines* a convertible — catch the rally, cushion the fall — simply does not show up in the most popular convertibles ETF (CWB); its fitted payoff bends slightly *down*, a beta-matched stock/bond blend matches or beats it for free, and the "bond floor" failed exactly when it was needed, so the wrapper is a linear stock/bond exposure with a fee.

## What we tested

The convertibles pitch, steelmanned: *"a convertible bond gives you **equity upside with bond downside** — your bond converts and rides the stock up, but a bond floor catches you on the way down, a **convex** one-way bet."* We take "convex" literally and measure it: a **Treynor–Mazuy-style quadratic regression** of CWB's daily total return on SPY *and the squared upside of SPY* (the convexity coefficient γ), with a Newey-West (HAC) *t* and a circular block-bootstrap CI on γ; then up-/down-**capture** and a race against a **beta-matched SPY/AGG blend** on **excess-of-cash** Sharpe (SHY proxy). The window is CWB's full history (**2009–2026**). The offline control is a synthetic tape that plants a genuinely convex payoff (positive γ) as the positive control and a linear blend (γ = 0) as the null — a machinery proof, never market evidence. **Distinct from [Study 97 (Balancing-Act)](../../97-balancing-act/)**, which tests *linear diversification*; here the object is *convexity*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "equity upside, bond downside" is a hockey-stick, and the chart showing CWB's curve bending the wrong way |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quadratic convexity regression with HAC *t* + bootstrap CI on γ, capture asymmetry, the beta-matched blend race, the credit-beta read |

The fingerprinted real-data run (CWB/SPY/AGG/SHY, as-of 2026-05-31, panel fp `0ab3e0356db0`) is in [docs/results.md](docs/results.md); reproduce via [examples/verify.py](examples/verify.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`convertible_bonds/`](convertible_bonds/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
