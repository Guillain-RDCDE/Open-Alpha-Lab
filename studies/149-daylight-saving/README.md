# Study 149 — Daylight-Saving

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Post-DST Monday gap = **−19.89 bps**, Welch *t* = **−1.19** (placebo p = 0.084); directionally consistent with Kamstra et al. but below the |*t*| ≥ 2 bar on 93 events. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Two events per year at an insignificant gap: effectively zero tradable alpha. |
| **Contested?** | ![Fragile](https://img.shields.io/badge/Fragile--finding-8b949e?style=flat-square) | Pinegar (2002) showed the original KKL result disappears with minor changes; post-2000 out-of-sample gap is only −7.98 bps (*t* = −0.42). |

> **In one sentence:** the Monday after a US clock change is modestly negative on the S&P 500 — a directional hint matching the sleep-disruption story — but with only 93 events over 46 years the gap never clears the statistical bar, the out-of-sample decade is flat, and two trades per year buy you nothing.

## What we tested

Kamstra, Kramer & Levi (2000) famously argued that sleep disruption from the annual
Daylight Saving Time clock change causes investors to be more risk-averse on the
following Monday, depressing returns by an amount far larger than any other known
calendar anomaly.  We steelman the claim and test it rigorously: we compute the
correct US DST change dates from 1980 to 2026 (pre-2007 and post-2007 rules),
label the Monday after each change, and compare the **post-DST Monday return**
against **all other Mondays** on ^GSPC daily close-to-close returns.  A
random-Monday placebo (500 seeds) quantifies the false-positive risk of testing
on just ~93 events.  We also split by season (spring-forward vs fall-back) and
by era (pre- vs post-publication), and check for decay with Pinegar's (2002)
critique in mind.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the sleep story, the DST calendar, the honest Monday-vs-Monday comparison in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*, Welch test, placebo distribution, season split, era decay, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`daylight_saving/`](daylight_saving/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
