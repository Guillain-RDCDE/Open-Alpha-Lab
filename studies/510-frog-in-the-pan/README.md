# Study 510 -- Frog-In-The-Pan 🐸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Is momentum stronger when the past return arrived *gradually* (in many small steps) than when it arrived in a few jumps?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the FIP premium statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The LOW-ID (gradual) WML book earns **−0.37%/yr**, HAC *t* = **−0.09**; the FIP interaction (low − high ID) is **+4.55%/yr** but HAC *t* = **+1.01** (|t| < 2). A seed-robust label-shuffle placebo gives **p = 0.56** -- the real mean sits *below* its own null. Nothing clears the bar. |
| **Tradability** -- does the gradual book pay net? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Flat-to-negative gross (−0.37%/yr), and **−1.29%/yr net** of 5 bps/leg + 50 bps/yr borrow at 35%/mo turnover. There is nothing to monetise. |
| **"Does gradual beat jumpy?"** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | The *sign* is on the theory's side -- LOW-ID WML (−0.37%/yr) beats HIGH-ID WML (−4.92%/yr) by +4.55%/yr -- and the synthetic control proves the engine would catch a real gap. But the magnitude is statistically indistinguishable from zero on this tape. |

> **In one sentence:** the Frog-In-The-Pan idea -- the market under-reacts more to information that boils up slowly, so momentum lives in the *gradual* names -- shows the right *sign* on a 38-name large-cap survivor basket (gradual WML beats jumpy WML by +4.55%/yr) but clears nothing statistically (interaction *t* +1.01, gradual book *t* −0.09), is negative net of costs, and a seed-robust placebo puts the real mean below its own null: None signal, Mirage tradability, Mixed on the qualitative question.

## What we tested

Da, Gurun & Warachka (2014): proxy the *continuity* of past information with an
**information-discreteness** measure `ID = sign(PRET) · (%neg − %pos)` -- a stock that drifted up
on many small positive days has low (negative) ID; one that jumped on a few days amid many small
down days has high ID. Each month we double-sort the cross-section: split at median ID, then form
a 12-1 momentum winners-minus-losers (WML) book *inside* the LOW-ID (gradual) half and *inside*
the HIGH-ID (discrete) half, and measure their difference (the FIP interaction). One forward
execution lag (form on the *t* close, hold month *t+1*), a label-shuffle placebo with seed
robustness, costs + short borrow, and a deterministic synthetic positive control where the drift
is planted *smoothly* in the low-ID names and *lumpily* in the high-ID names. Basket: 38 large-cap
S&P 500 names, yfinance daily prices 2012-2025 (151 stamped WML months). Survivorship is named
**on the signal axis** -- a gradual slide into delisting is itself a low-ID loser, so the FIP
slice is the more inflated of the two. *Distinct from [507 Cross-Sectional-Momentum](../507-cross-sectional-momentum/)
(the unconditioned base WML this study refines) and [509 Intermediate-Momentum](../509-intermediate-momentum/)
(conditions on the timing of past returns, not their continuity).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the boiling-frog metaphor in plain language, what ID measures, synthetic positive control, the real low-ID vs high-ID race, honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the ID formula, the double-sort, low/high-ID WML with HAC *t*, the FIP interaction, the seed-robust label-shuffle placebo, costs + borrow, survivorship discussion |

The fingerprinted real-data run (38 names, 2012-2025, fp `e038fa42cc02`) is in
[docs/results.md](docs/results.md); the offline machinery proof runs on the synthetic world in
[`frog_in_the_pan/data.py`](frog_in_the_pan/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`frog_in_the_pan/`](frog_in_the_pan/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
