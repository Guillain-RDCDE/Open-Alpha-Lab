# Study 528 — Labor-Hiring-Rate

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> *Do the firms that hire fastest go on to underperform the disciplined ones?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Long-low-hire / short-high-hire hedge **−4.68%/yr**, HAC *t* = **−1.29** (wrong sign, \|t\| < 2); placebo p = **0.81** (the real result sits in the *left* tail of the shuffle null); low-hiring beats only **48%** of random draws. The fast-hiring tercile *out*-earned the disciplined one (+22.2% vs +17.5%/yr). Literature supports the effect on a broad universe, not on survivor large-caps. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No positive gross edge to begin with; after 10 bps × turnover and a 50 bps short borrow the hedge is **−5.38%/yr net**. Shorting the high-hiring leg means shorting the AI/hyperscale winners. Nothing to trade. |
| **Predicted sign (fast hiring → low returns)?** | ![Busted](https://img.shields.io/badge/Sign-Busted-8b949e?style=flat-square) | The sign *reverses* on the survivor large-cap panel — fast hirers earned **+22.2%/yr** vs **+17.5%/yr** for disciplined hirers — the same fate as the capex investment channel in [Study 523](../523-investment-to-assets/). |

> **In one sentence:** Belo-Lin-Bazdresch (2014) found that fast-hiring firms underperform on the broad US cross-section, but on a survivorship-biased large-cap basket the hiring anomaly is not merely absent — it *inverts* (hedge −4.68%/yr, *t* = −1.29; high hirers beat low hirers), because the aggressive hirers that survived are the AI, hyperscale and managed-care growth winners, not over-expanding failures.

## What we tested

Belo, Lin & Bazdresch (2014) model labor as a quasi-fixed factor: hiring carries adjustment
costs (search, training, severance), so the **hiring rate** behaves like an investment rate
and inherits the investment anomaly's prediction — fast hirers earn low future returns. This
is the **labor-input channel** of the broad investment anomaly, the employment cousin of the
capex channel in [Study 523](../523-investment-to-assets/) and the total-asset-growth channel
in [Study 244](../244-asset-growth/).

We compute the hiring rate **HN = (N_t − N_{t-1}) / (0.5·(N_t + N_{t-1}))** from a curated
panel of 10-K cover-page full-time-employee counts, sort a fixed 28-name large-cap survivor
basket into terciles each fiscal year, lag the signal by a conservative report lag (12-month
forward return beginning 4 months after fiscal year-end, entered one trading day later —
exactly one execution lag, no look-ahead), and test whether the low-hiring tercile beats the
high-hiring tercile. A label-shuffle placebo, a random-portfolio null, explicit costs + short
borrow, and a seed-robust synthetic positive control round out the inference.

**Data limitation (named honestly):** employee counts are *not* machine-readable. Firms
report headcount as 10-K cover-page narrative text, so SEC EDGAR companyfacts/companyconcept/
frames carry the numeric `dei:EntityNumberOfEmployees` tag for only ~5–11 filers market-wide,
and yfinance exposes only a single current snapshot. The hiring rate is therefore computed
from a **curated, hardcoded panel** of 10-K cover-page headcounts (FY2013–FY2024, public
reported facts), anchored to the live yfinance `fullTimeEmployees` snapshot — capping the
sample at 11 stampable sort years.

The basket is **survivorship-biased**: it covers only firms still large-cap in 2026.
Critically, the high-hiring group on a survivor basket skews toward *successful* expanders —
AI/semiconductors (NVDA), hyperscalers (AMZN, GOOGL, MSFT), warehouse retail (COST) and
managed care (UNH) — the opposite of the over-expanding failures the anomaly was built around.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the hiring-rate recipe in plain English, why the survivor basket flips the sign, year-by-year results |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | tercile monotonicity (inverted), HAC t-stats, label-shuffle placebo, random-portfolio null, costs, seed-robust synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`labor_hiring_rate/`](labor_hiring_rate/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
