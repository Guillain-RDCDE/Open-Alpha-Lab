# Study 748 — CEO-Age-Effect 👴

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the CEO's *age* predict returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Long-young / short-old earns **+7.97%/yr** raw — but the Newey-West HAC *t* is only **+0.92** (4 lags, n = 101) and a label-shuffle placebo beats it **52%** of the time. Curated, growth-confounded 40-name tape — structurally can't certify `REAL`. |
| **Tradability** — does the spread pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Regress out the market and the CAPM **alpha is +2.98%/yr at *t* = 0.36** while the **beta is +0.35 (*t* = 2.7)** — the whole edge is growth beta. Net **+7.19%/yr** is a factor tilt you can rent from an ETF for less. |
| **Young-CEO edge?** | ![Misattributed](https://img.shields.io/badge/Misattributed-8b949e?style=flat-square) | Young basket Sharpe **0.90** < old **1.08** (more return, *much* more vol) — risk-adjusted the old CEOs win. The spread flips sign by regime (+27 → −34 → +17 %/yr) and its *t* wanders with the arbitrary age cutoff. It's sector/size/vintage, not the birthday. |

> **In one sentence:** the aggressive-young-CEO story has real papers behind it (Serfling 2014; Yim 2013) — young bosses do run wilder firms — but as a *trade* it's a mirage: a 40-name long-young/short-old book earns +8%/yr that is a coin flip (HAC *t* 0.92), evaporates to zero alpha once you subtract the market (alpha *t* 0.36, beta *t* 2.7), flips sign every regime, and leaves the young basket with a *worse* Sharpe than the old one — growth-tech beta wearing a birthday costume.

## What we tested

The claim, steelmanned with its academic backing: **young CEOs are aggressive, risk-taking,
growth-hungry** (Yim 2013, *"The Acquisitiveness of Youth"*) while **old CEOs play it safe**
(Serfling 2014, *"CEO age and the riskiness of corporate policies"*) — so a long-young /
short-old equity book should pay. We hardcode a **transparent, cited 40-name large-cap CEO table**
with each chief's **public birth year**, split them **young (< 55) vs old** at a fixed scoring date,
and build the equal-weight, dollar-neutral long/short on real yfinance **monthly total returns**. We
judge it with a **Newey-West HAC *t*** on the spread, a **CAPM alpha-vs-beta** decomposition (the
decisive control), a **label-shuffle placebo**, cutoff and regime robustness sweeps, costs + a short
borrow, and a seed-robust **synthetic positive control** that plants a real age premium to prove the
engine would catch one. The young bucket is a growth-tech / recent-IPO cohort — a confound named
loudly on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why young-CEO stocks are wilder but not better, how "young" is really "growth tech," and why the edge flips sign whenever growth falls out of favour — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC *t* on the long/short, the CAPM alpha-vs-beta control, the label-shuffle placebo, cutoff & regime sweeps, costs + borrow, and the seed-robust synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ceo_age_effect/`](ceo_age_effect/). The CEO→age table is a **hardcoded, cited** cross-section; the young bucket is a **growth-tech / recent-IPO** cohort (a confound) and not survivorship-free — both named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
