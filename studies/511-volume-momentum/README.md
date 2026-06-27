# Study 511 — Volume-Momentum 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does volume condition momentum? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The HIGH-volume winners-minus-losers book earns **+2.53%/yr** at HAC *t* = **0.44**; the LOW-volume book **+4.55%/yr** at *t* = **1.26** — the strongest slice is still under the bar. The Lee-Swaminathan interaction (**HIGH-vol minus LOW-vol WML**) is **−2.02%/yr at *t* = −0.37**, the *opposite sign* to the prediction. A seed-robust label-shuffle placebo gives **p ≈ 0.24**. **Survivorship is named on the signal axis**: the basket is names *still trading in 2026*, so the low-volume losers (the literature's strongest short) that faded into delisting are absent — these flat numbers are already an upper bound. |
| **Tradability** — does the spread pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net of 5 bps/leg × turnover + 50 bps/yr borrow, the HIGH-volume book is **+1.76%/yr at HAC *t* = 0.31** — indistinguishable from zero before *and* after costs, on a **6-name leg** of a 40-name survivor basket. There is no edge for costs to erode. |
| **"Momentum life cycle — high volume reverses faster"?** | ![Busted](https://img.shields.io/badge/Life_cycle%3F-Busted-8b949e?style=flat-square) | The double-sort ordering is **inverted** (LOW-volume WML beats HIGH-volume WML at every hold of 1/3/6/12 months), and the predicted faster high-volume reversal **never appears** — both books drift mildly *up* with the hold, neither reverses, none is significant. The synthetic control proves the engine *would* find the pattern if it were there. |

> **In one sentence:** Lee & Swaminathan's (2000) "price momentum and trading volume" — high-volume winners and low-volume losers should drive the drift, and high-volume names should reverse faster — shows up **with the wrong sign and no significance** on a 40-name large-cap survivor basket: the HIGH-volume WML (*t* 0.44) is *weaker* than the LOW-volume WML (*t* 1.26), the high-minus-low interaction is **negative** (−2.02%/yr, *t* −0.37), a seed-robust placebo puts *p* at ~0.24, the net spread is a coin flip, and the famous "momentum life cycle" never materialises — **None** signal, **Mirage** tradability, **Busted** life cycle.

## What we tested

We rebuild the **volume-momentum life cycle** (Lee & Swaminathan 2000) as a clean monthly
cross-sectional **double-sort** on a fixed **40-name large-cap survivor basket** (same family as
[507](../507-cross-sectional-momentum/) / [510](../510-frog-in-the-pan/)): per name we form the
**12-1 momentum** signal and a trailing **dollar-volume** (turnover) measure over the same window,
split the cross-section at its median turnover into a **HIGH-volume** and a **LOW-volume** half,
and build a winners-minus-losers (top-30% − bottom-30% by momentum) book inside each — dollar-
neutral, one forward execution lag, no look-ahead. The Signal axis tests each WML and the
**HIGH-minus-LOW interaction** against zero with a HAC *t* and a **seed-robust label-shuffle
placebo**; Tradability charges one-way costs × NAV × turnover + short borrow; the third axis traces
the **volume-conditioned reversal** (cumulative WML at 1/3/6/12-month holds, high vs low volume).
A deterministic synthetic control with a *planted, volume-tilted* momentum drift confirms the
engine is faithful and that zero edge cannot fake the ordering. **Survivorship** (names still
trading in 2026 — which inflates the low-volume loser leg the most) is named on the Signal axis.
*Distinct from [510 Frog-In-The-Pan](../510-frog-in-the-pan/) (continuity), [509 Intermediate-Momentum](../509-intermediate-momentum/) (timing) and [141 Turnover-Anomaly](../141-turnover-anomaly/) (turnover as a standalone signal).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "the busy stocks have the strongest momentum" means, why the obvious sort comes out backwards here, and why the trade is a coin flip — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the momentum × volume double-sort, HIGH- vs LOW-volume WML with HAC *t*, the signed Lee-Swaminathan interaction, a seed-robust label-shuffle placebo, costs × turnover, the volume-conditioned reversal term-structure, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run
(prices fp `99db07758961`, volume fp `ef22b095227e`): [docs/results.md](docs/results.md).

---

*Engine: [`volume_momentum/`](volume_momentum/). The conditioner is trailing mean daily dollar
volume (close × share volume) over the 12-1 formation window. Basket is **survivors** — named on
the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
