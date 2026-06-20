# Study 327 — Disposition-Effect 🪤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Grinblatt-Han is a real, much-replicated CRSP anomaly — but on this small large-cap cross-section the overhang Q5−Q1 hedge runs **HAC *t* = −0.16**, CI straddling zero. Strong literature, *this* tape can't certify it. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross return ≈ **0%/yr**, the quintile fan is flat (~15%/yr in every bucket), and it only goes more negative as costs bite — and that's on a *survivorship-biased upper bound*. |
| **Just momentum in disguise?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Overhang is mechanically a recent-winner proxy; the momentum-orthogonalised residual is equally dead (*t* = −0.65). Whatever (absent) signal there is, it isn't separable from momentum. |

> **In one sentence:** "investors sell winners too early, so deep-in-the-money stocks under-react and outperform" is a genuine textbook anomaly on the broad market — but on a tradeable basket of liquid large caps the capital-gains-overhang factor is a flat line that can't clear the inference bar and is indistinguishable from momentum.

## What we tested

Grinblatt & Han (2005) made the disposition effect a cross-sectional asset-pricing claim: because
disposition-prone holders dump winners too soon, stocks sitting on a large **unrealised capital
gain** are held below fundamentals and subsequently outperform, while underwater names lag — and
they argue this *capital-gains overhang* even subsumes momentum. We take that literally: build the
turnover-weighted overhang ``g = (P − R)/P`` for a small, named, liquid large-cap cross-section,
sort into quintiles monthly, and run the Q5 − Q1 hedge with one execution lag, HAC *t*-stats, a
block-bootstrap CI, a cost sweep, and a momentum-orthogonalisation control. A deterministic
synthetic panel with a tunable overhang premium is the positive control (and the null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why selling winners too early *should* leave money on the table — and why the basket version goes nowhere |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile fan, HAC *t* + bootstrap CI, the momentum-orthogonalisation control, the cost sweep, the synthetic detector |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`disposition_effect/`](disposition_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
