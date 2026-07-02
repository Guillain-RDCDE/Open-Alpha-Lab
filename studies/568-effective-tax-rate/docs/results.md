# Results — Study 568 (Effective-Tax-Rate): the tax-rate return anomaly on a survivor basket

*Generated from [`effective_tax_rate/`](../effective_tax_rate/) over this study's cached tape:
income-tax expense and pretax income from **EDGAR** annual 10-K facts (data.sec.gov) for a fixed
**40-name large-cap survivor basket** (673 firm-years, fundamentals fingerprint `ca80cb42a4ce`),
and daily adjusted closes from **yfinance** (2008-01-02 → 2026-07-01, 40 names). The ETR panel
(fiscal years 2007-2025, fingerprint `74b7313496ca`) is aligned to next-full-calendar-year returns
(fingerprint `91df5a37e544`); the **17 sortable years are fiscal 2008 → 2024** (forward returns
2009 → 2025; the current partial year is dropped). As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE`

The accounting claim: a firm's **effective tax rate** (ETR = income-tax expense / pretax income)
predicts returns. Two competing stories give it *opposite* signs — the *quality / tax-avoidance
premium* (low-ETR firms are efficient cash machines the market underprices → higher returns) and
the *red-flag / risk* story (a suspiciously low ETR is a fragile loophole that reverses → lower
returns). We build a **low-minus-high ETR** quintile sort (long the lowest-ETR fifth, short the
highest) on the real tape and let it decide.

**It decides nothing — the sort produces a clean null.** Over the 17 sortable years the
long-low/short-high-ETR hedge earned **−3.3%/yr** (HAC Newey-West *t* = **−0.91**, Sharpe −0.22,
47% hit rate). That is the *wrong* sign for the quality story (low-ETR firms very slightly
*under*-earned) but statistically indistinguishable from zero, and the **label-shuffle placebo
*p* = 0.422** confirms it: the observed hedge sits squarely in the middle of the null, not its
tail. The cross-sectional rank **information coefficient is −0.02 (*t* −0.46)** — no monotone
ETR→return relation. The **change-in-ETR** signal is equally dead (hedge −2.3%/yr, *t* −0.77,
IC ≈ 0.00). And the sign is unstable across sub-samples (+3.0% in 2009-13, −2.5% in 2014-18,
−4.2% in 2019-24), every window's |*t*| below 1.1. So `NONE` on the signal axis (no robust
*t* ≥ 2 on the real tape, wrong-signed, sign-unstable, placebo-dead) and `MIRAGE` on tradability
(a survivor basket, annual rebalance, the short leg being high-ETR names with nothing to harvest —
net −3.9%/yr).

## Data stamp

- **Fundamentals**: income-tax expense (`IncomeTaxExpenseBenefit`) and pretax income
  (`IncomeLossFromContinuingOperationsBeforeIncomeTaxes…`) from EDGAR annual 10-K FY facts,
  40 names, 673 firm-years, fingerprint `ca80cb42a4ce`
- **Prices**: 40 large-cap survivors, daily adjusted close, 2008-01-02 → 2026-07-01
- **ETR panel** (fiscal 2007-2025): fingerprint `74b7313496ca`; **forward returns** (year y → y+1):
  fingerprint `91df5a37e544`; 17 sortable years (fiscal 2008 → 2024)

## The ETR sort — the low-ETR premium does NOT show up

| Quintile (≈7 names) | What it holds | Mean next-year return |
|---|---|---|
| **Q1 (lowest ETR)** — the "long" leg | PFE, IBM, GE, BAC, VZ, AMGN… (mean ETR ~6-17%) | the long leg |
| **Q5 (highest ETR)** — the "short" leg | CVX, COP, DE, COST, CAT, XOM… (mean ETR ~28-33%) | the short leg |
| **Hedge (Q1 − Q5), 17 yrs** | long low-ETR, short high-ETR | **mean −3.3%/yr, HAC *t* −0.91** |

The quality story predicts a *positive* hedge (low-ETR firms out-earn). The tape delivers a small
*negative* mean that a permutation test cannot separate from zero. Note *which* names populate the
legs: the low-ETR bucket is dominated by banks, telecom and old-line tech/pharma carrying big loss
shields and deferred items; the high-ETR bucket is energy and heavy industrials that pay close to
statutory. The sort is really picking up sector, not a durable quality signal — and it pays nothing.

## Placebo and information coefficient

| | value |
|---|---|
| Hedge mean (Q1 − Q5) | **−3.3%/yr** |
| Hedge HAC *t* | **−0.91** |
| Label-shuffle placebo *p* (1000 perms) | **0.422** |
| Rank-IC(ETR, fwd return), 17 yrs | **−0.023** (*t* **−0.46**) |
| Change-in-ETR hedge / IC | **−2.3%/yr** (*t* −0.77) / IC **−0.003** |

The placebo *p* = 0.42 is the headline: a genuine cross-sectional edge sits in the tail of the
shuffled null; this one sits in the middle. There is no signal in either the level or the change.

## Robustness — the sign is not even stable, and never significant

| Sub-sample | Hedge mean/yr | HAC *t* | Reads as |
|---|---|---|---|
| 2009-2013 | **+3.0%** | +0.62 | weak low-ETR premium |
| 2014-2018 | **−2.5%** | −1.06 | inverted |
| 2019-2024 | **−4.2%** | −0.87 | inverted |
| 2009-2024 (full) | **−1.4%** | −0.46 | nothing |

No window clears |*t*| = 1.1, and the sign flips between the early and later sub-samples. A signal
that is small, insignificant *and* sign-unstable is not a signal — `NONE`.

## Costs

| | value |
|---|---|
| Gross hedge (Q1 − Q5) | **−3.3%/yr** |
| Net (10 bps/leg one-way × 64% turnover + 50 bps/yr borrow) | **−3.9%/yr** (*t* −1.08) |

Costs are academic here: the trade does not pay before frictions, and it pays less after.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `premium` (return per z-ETR) | Mean hedge-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **−0.14** | flat — no false signal |
| −0.03 (low-ETR out-earns) | +6.29 | premium visible |
| −0.05 | **+10.58** | strong |
| −0.08 | +17.02 | very strong |
| +0.05 (high-ETR out-earns) | −10.81 | inverted premium caught |

At the null the hedge-*t* is ≈ 0; planting a genuine low-ETR premium (`premium < 0`) drives the
Q1−Q5 hedge strongly *positive*, and a high-ETR premium drives it negative. The detector works and
is not sign-confused — so the real-tape null is a statement about **this survivor basket over
2008-24**, not a broken engine. (Control only; never cited for the real-tape stamp.)

## Why the anomaly doesn't certify here

1. **Survivorship, again.** The basket is names *still trading in 2026*. Any red-flag effect lives
   in firms whose aggressive tax positions *reversed and blew up* (restatements, delistings) — the
   exact tail a survivor basket deletes. This biases the real tape *against* the risk story and
   toward a weak-to-null reading; named on the SIGNAL axis. A survivor-only tape can never earn
   `REAL`.
2. **ETR is mostly a sector proxy on 40 blue chips.** With only ~36 valid names a year, the low-ETR
   fifth is banks/telecom/pharma and the high-ETR fifth is energy/industrials — the sort measures
   *industry* more than *tax quality*, and industry-neutralising would strip most of what little is
   there.
3. **Point-in-time, not restated, and coarse.** EDGAR FY facts are the last-filed annual value, and
   ETR is a noisy one-year ratio (one-off items, deferred-tax swings, negative pretax years excluded).
   A real replication needs industry-adjusted, multi-year-smoothed ETRs on a survivorship-free
   Compustat panel.

## The honest takeaway

The effective-tax-rate anomaly is a live debate in the literature (quality vs risk, sign
contested) — but on a 40-name large-cap survivor basket over 2008-24 it simply **does not appear**:
the low-minus-high-ETR hedge is a small, *wrong-signed*, insignificant −3.3%/yr (HAC *t* −0.91),
the placebo *p* = 0.42, the rank-IC is ≈ 0, the change-in-ETR is equally dead, and the sign flips
across windows. `NONE` × `MIRAGE`. The synthetic control confirms the engine would light up on a
real low-ETR premium of either sign — so this is the tape talking, not the code. The spec's prior
lean was *Weak*; the honest run is a notch below that — a clean **null**.
