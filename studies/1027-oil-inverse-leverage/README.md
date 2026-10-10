# Study 1027 — Oil Is Equity in a Mirror 🪞

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does oil's volatility rise after rallies (the inverse leverage effect)? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Neither lens sees the mirror on 1986-2018 WTI spot. The GJR-GARCH asymmetry has the **equity** sign (inverse *t* = **−2.62**, EGARCH −4.40, robust SEs), and the model-free forward-volatility regression split by return sign reads *t* = **−1.28** (Newey-West, 21 days). The S&P 500 mirror reads −8.10. In the pre-registered eras, 1986-2007 is symmetric (*t* = −0.15) and 2008-2018 is equity-like (*t* = −4.81; difference z = +3.38): oil changed, but *away* from the mirror. A rolling three-year GJR clears +2 in only 9% of windows (around the 1990-91 Gulf War and 1996), none ending after 1997. |
| **Tradability** — does it change what a vol-target overlay is worth? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Capped by the tape: WTI here is a **spot** price, not an investable futures return. On that proxy a 21-day vol-target overlay (one lag, 10 bp one-way) cuts the excess Sharpe from 0.27 to 0.19 (gain −0.08, block-bootstrap CI [−0.21, +0.06]) and deepens the drawdown (−82% → −86%); on the S&P price index it gains +0.02 (CI [−0.17, +0.21]). The thermostat de-risks after falls on the S&P (reaction +0.31) and is nearly sign-blind on oil (+0.04) — it never de-risks after rallies. |
| **Is oil a mirror?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Oil is a fainter *copy* of equity, not its reflection — and only since 2008. |

> **In one sentence:** On 1986-2018 WTI spot oil is not equity in a mirror but a fainter copy — its volatility also rises more after falls (GJR t = -2.6), a tilt that appeared after 2008 (era z = +3.4) while the mirror never cleared the bar in any era; and a vol-target overlay does not pay on oil (net Sharpe gain -0.08) on a spot tape you cannot hold anyway.

## What we tested

In equities volatility rises when prices fall — the leverage effect of Black (1976) and Christie
(1982, *JFE*). The commodity steelman says oil is the mirror image: the shocks that matter are
supply disruptions that *spike* the price, so in oil it is the **rallies** that bring the storm,
an inverse asymmetry reported for gold by Baur (2012) and studied for energy futures by
Kristoufek (2014). If true, a vol-target overlay would de-risk oil after rallies, not after
crashes. We test it on the frozen daily **WTI spot** tape (1986-2018) against the daily **S&P 500
price index** (1999-2018), with monthly-average Brent as a cross-check: GJR-GARCH and EGARCH with
robust SEs, a model-free forward-volatility regression split by return sign (Newey-West), a
pre-registered 1986-2007 / 2008-2018 split plus a rolling GJR, skewness, and the overlay on both
assets (one lag, costs, excess-vs-excess). A synthetic GJR world whose asymmetry sign is set by
`signal_strength` proves the estimators recover sign and size.
**Dedup:** distinct from **993-leverage-effect-asymmetry** (the *equity* leverage effect, with
gold and Bitcoin as controls), **245-oil-equity-correlation** (whether oil *leads* stocks — a
return question), and **16-storm-shy** / **591-vol-managed-portfolio** (vol targeting on
equities); here the subject is the *sign* of oil's return-volatility asymmetry, its stability
across regimes, and what it does to the overlay.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the mirror story, what happens the month after a big up vs down day, the two eras, and the thermostat on oil vs stocks — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | GJR/EGARCH with robust SEs, the Newey-West sign-split regression, era z-test and rolling GJR, skewness with block-bootstrap CIs, the Brent cross-check, the overlay with cost sweep, and the synthetic calibration |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mirrorlev/`](mirrorlev/). Data: frozen tapes inside the `arch` package, SHA-256 pinned via `quantlab.bundled`. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
