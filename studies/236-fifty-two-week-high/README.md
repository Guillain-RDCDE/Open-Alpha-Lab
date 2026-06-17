# Study 236 — Fifty-Two-Week-High

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Q5-Q1 spread = **−10.60 bps/week**, HAC *t* = **−0.85**; *negative* at every horizon from 5d to 65d. The 1-day horizon shows +10.54 bps but *t* = +1.57 (below the bar). The signal is indistinguishable from zero and runs in the wrong direction on this large-cap sample. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Top decile earns **+6.8%/yr** against a passive equal-weight basket at **+11.3%/yr** — a 4.5 pp drag before any transaction costs. Adding realistic costs makes it an outright loser. |
| **George-Hwang anomaly?** | ![Inverted on Large-Caps](https://img.shields.io/badge/George--Hwang-Inverted__on__Large--Caps-8b949e?style=flat-square) | The canonical 1963-2001 result does not survive on this 2013-2026 survivorship-biased mega-cap panel. Far-from-high names outperform near-high names — the opposite of the published finding. |

> **In one sentence:** buying stocks near their 52-week high as a momentum bet produces a 4.5 pp/yr drag versus the equal-weight basket on this large-cap sample — the famous George-Hwang anomaly is inverted here, likely because mega-cap survivors at their highs are mean-reverting, not underreacting.

## What we tested

Is nearness to the 52-week HIGH a better momentum signal than price itself?

George & Hwang (2004) argued yes: rank S&P 500 names cross-sectionally by proximity = close / 252d rolling high, form equal-weight quintile portfolios, and collect the Q5 (near-high) minus Q1 (far-from-high) spread. We replicate this protocol on 20 representative S&P 500 large-cap names, 2013-01-02 to 2026-06-16 (~13 years, 3,384 panel-days). **Survivorship-biased** — all names still trade in 2026. The original paper used a broader, unbiased CRSP universe (1963-2001) and 6-12 month hold periods; we test 1 to 65 trading days.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why the anchoring story feels right, the quintile chart that runs backwards, why the anomaly inverts on large-caps, the cost argument |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats per quintile, hold-period sweep (1d to 65d), long-only vs equal-weight comparison, synthetic positive control confirming the engine is a faithful momentum detector |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fifty_two_week_high/`](fifty_two_week_high/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
