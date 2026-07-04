# Study 597 — Guyton-Klinger Guardrails 🚧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do guardrails let you start at 5%+ where fixed 5% fails? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | On 1,470 monthly-start 30-year retirements (Shiller 1871–2023), Guyton-Klinger at a **5% start survives every cohort** where the fixed rule ruins **24.35%** (bootstrap CI on Δsuccess excludes 0); guardrails SAFEMAX **5.83%** vs fixed **3.68%**; lifetime real income beats the 4% rule by **+0.44 per $1** (HAC *t* = **+4.23**, CI [+0.13, +0.97]). Robust to 60/40 vs 65/35 and 0–25 bps costs. Named caveat: single-index US tape — history's best equity market flatters both rules. |
| **Tradability** — should a retiree deploy it as pitched? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Trivially implementable (two ETFs + one rule check a year, costs immaterial) — but what you get is **not a safe 5% paycheck**: 55% of cohorts spend a mean ~7 years **below the 4%-rule paycheck**, the worst floor is a **66% real cut** (income 0.0169 per $1), and **23%** of cohorts end with *less* lifetime income than the plain 4% rule. The advertised rate is regime-dependent by construction. |
| **"Is the 5%+ start a free lunch?"** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | Ruin risk is **converted into paycheck risk, not removed**. In 1929 and 1966 — the very regimes the pitch invokes — the "successful" guardrails plan paid **less lifetime income (0.988, 0.870)** than the fixed rules that *failed* (1.200, 1.029): it survives *by* cutting your income when you are poorest. The cut rule is the rescue; the raise rule is the income; the premium is deducted from the paycheck. |

> **In one sentence:** Guyton-Klinger guardrails really do what they claim — a 5–5.5% start survives all 152 years of US history because withdrawals that shrink with the portfolio can't easily deplete it (SAFEMAX 5.83% vs 3.68% fixed, lifetime income +0.44/$1 over the 4% rule at HAC *t* +4.23) — but the "safe 5%" is an insurance policy whose premium is your own paycheck: half of all cohorts spend years below the 4%-rule income, and in 1929/1966 the plan "succeeds" while paying less than the rules that officially failed.

## What we tested

We simulate 30-year retirements on Shiller nominal stock, 10-year-bond and CPI series (all outcomes deflated to real): the fixed Bengen rule vs the full Guyton-Klinger 2006 decision rules — inflation raises capped at 6% and **frozen** after a losing year, a **capital-preservation cut** (−10% when the current withdrawal rate breaches 1.2× the initial, >15 years remaining) and a **prosperity raise** (+10% below 0.8×) — at 4–6% initial rates, 60/40 annually rebalanced, 10 bps one-way costs, one clean execution lag. Inference: Newey-West HAC *t* on overlapping-cohort differences (bandwidth = full 360-month overlap) + circular block-bootstrap CIs (1,000 reps, 120-month blocks). A rule-decomposition shows the cut is the rescue and the raise is the income; a 20-world synthetic control reads exactly 0 where no fixed-rate ruin exists and lights up in proportion to planted ruin. Cousin of [173 — Four-Percent-Rule](../173-four-percent-rule/) (the FIXED rule, Real) and [596 — Bond Tent Glidepath](../596-bond-tent-glidepath/) (dynamic *allocation*, None); new here is the **dynamic withdrawal policy**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the guardrails are, why they can't run out of money, and what the "safe 5%" actually feels like in 1966 — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | cohort machinery, HAC/bootstrap inference, the rule decomposition, the income-price accounting, allocation/cost robustness, and the ruin-rescue synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`guyton_guardrails/`](guyton_guardrails/). Rules act on last year's completed inflation/return and the current portfolio value at the start-of-year withdrawal date (one clean lag); costs 10 bps one-way × traded value; nominal simulation, real (CPI-deflated) outcomes, labeled. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
