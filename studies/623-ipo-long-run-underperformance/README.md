# Study 623 — IPO Long-Run Underperformance 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do IPOs drift down for 3-5 years? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The literature record is real — across **45 published cohort years** (Ritter, 1980-2024, 9,253 IPOs) the market-adjusted 3-yr drift is **−14.2%/3yr at NW t = −2.26**, 71% of cohorts negative. But Ritter's **own style adjustment drops it to t = −1.52**, and on our live investable tape (**152 months** of the Renaissance IPO ETF) the vs-SPY alpha is **t = −0.99** and the style-matched alpha **≈ 0**. The literature says real; this tape can't certify it — **Weak** by the inference bar. (No survivorship on either track; the live window covers 2013+ only, named.) |
| **Tradability** — can you monetise the drift? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Short-IPO-ETF / long-SPY nets a statistically empty **+0.92%/yr** (HAC t = +0.15) at 5 bps + 1%/yr borrow, **goes negative at 2% borrow**, and sat through a **−57% drawdown** in the 2020 IPO melt-up. Avoidance ("don't buy IPOs early") costs nothing and reliably earns nothing vs a small-growth fund. |
| **Small-growth beta, not IPO-ness?** | ![Confirmed](https://img.shields.io/badge/Small--growth_beta%3F-Confirmed-8b949e?style=flat-square) | Ritter's style adjustment cuts the cohort drag **−14.2% → −8.5%** (t −2.26 → −1.52); live, the **−5.5%/yr** vs-SPY alpha collapses to **+0.03%/yr vs IWO** and **−0.56%/yr (t = −0.12)** two-factor, with a **0.999** loading on the small-growth spread. Brav & Gompers were right: the drag is the *style*, the IPO wrapper is mostly incidental. |

> **In one sentence:** Ritter's famous 3-5 year IPO drift is real in the published 1980-2024
> cohort record (−14.2%/3yr market-adjusted, NW t = −2.26) but evaporates the moment you
> style-match it — his own tables (t = −1.52) and our 152-month live test of the Renaissance
> IPO ETF (alpha ≈ 0 vs small-growth, t = −0.12) agree the bagholder's true sin is buying
> expensive small-growth beta, and there is nothing deployable in shorting it — **Weak, and a
> tradability Mirage**.

## What we tested

Two tracks. **Track A** re-runs Ritter's published cohort table (Table 19, updated Feb 2026;
hardcoded, source PDF cached) through our stats: one mean 3-yr buy-and-hold abnormal return
per cohort year, Newey-West *t* across 45 overlapping cohorts (lag 3), raw / market-adjusted /
style-adjusted. **Track B** is the live judge: monthly total returns of the Renaissance IPO
ETF (a rules-based basket of recent IPOs held ~2-3 years — Ritter's aftermarket window,
investable) vs SPY, IWM and IWO, Nov 2013 → Jun 2026, with HAC-t spreads and CAPM/two-factor
alphas (lag 6). Tradability charges the short leg borrow (1-2%/yr) plus one-way costs × NAV on
the monthly re-hedge; the rule is calendar-known, so the effective lag is zero by construction
(documented). A 20-seed synthetic control proves the HAC-alpha machinery is unbiased (null
rejects at the nominal 5%) and quantifies its power. Distinct from its siblings:
[`219-ipo-pop`](../219-ipo-pop/) is the **day-1** pop and [`265-ipo-volume`](../265-ipo-volume/)
is the **timing signal** — this is the **long-run drift**, measured from the first close, never
the offer price. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the IPO pop isn't yours, what actually happens in years 1-3 after listing, the 45-year cohort record, the live ETF experiment, and why "avoid shiny new listings" is really "avoid expensive small-growth" — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | NW-t cohort inference on overlapping 3-yr windows, HAC spread/alpha tests vs three benchmarks, the style-absorption two-factor regression, borrow-and-cost sweeps on the short, and a seed-averaged power analysis of the detector |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ipo_long_run_underperformance/`](ipo_long_run_underperformance/). Track A is
published data re-presented (literature story, never the REAL stamp); Track B is the
investable tape and the judge. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
