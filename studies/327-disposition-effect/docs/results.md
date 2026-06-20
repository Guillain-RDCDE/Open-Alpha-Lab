# Results — Study 327 (Disposition-Effect) on a real cross-section

*The Grinblatt & Han (2005) **capital-gains-overhang** factor — long the deep-in-the-money
quintile (high unrealised gain), short the underwater quintile — rebalanced monthly on a
small, named, liquid large-cap cross-section (a Dow-30-style list). Built from daily
adjusted close + volume via `yfinance`; the overhang ``g = (P − R)/P`` uses a turnover-weighted
1260-day reference price ``R`` (the Grinblatt-Han cost-basis proxy). Signal at month-end M →
return of month M+1 (one calendar-known lag, applied once). As-of **2026-06-19**; match the
fingerprint to confirm you hold the same tape.*

> **Survivorship — named on the Signal axis.** The universe is a hand-picked list of *current*
> large caps projected backwards (one name, `WBA`, dropped as delisted during the fetch). The
> opt-in guard (`allow_survivorship_bias=True`) was used; every magnitude below is therefore an
> **upper bound / illustrative**, never a certified premium.

## Data stamp

| Field | Value |
|---|---|
| Universe | 29 surviving large caps (Dow-30-style; `WBA` delisted, dropped) |
| Window | 2010-01-31 → 2026-05-31 (197 monthly rebalances) |
| Median names per cross-section | 28 |
| Overhang fingerprint | `4fe74d39c1cc` |
| Forward-return fingerprint | `6525f33dc337` |

## The headline — overhang Q5 − Q1 hedge (equal-weight, monthly, gross)

| Quintile (by overhang) | annualised mean |
|---|--:|
| Q1 (underwater) | +16.3% |
| Q2 | +15.2% |
| Q3 | +14.9% |
| Q4 | +15.2% |
| Q5 (deep in the money) | +15.6% |
| **Hedge (Q5 − Q1)** | **−0.7%** |
| Equal-weight market | +15.5% |

| Stat | Hedge (Q5 − Q1) |
|---|--:|
| Annualised return | **−0.7%** |
| Annualised vol | ~16% |
| Sharpe | −0.04 |
| **HAC *t* (monthly)** | **−0.16** |
| Hit-rate | 48% |
| Max drawdown | −62% |
| Block-bootstrap 95% CI (monthly mean) | **[−0.74%, +0.65%]** — straddles zero |

- The quintile fan is **flat**: every overhang bucket returns ~15%/yr, indistinguishable from
  just owning the equal-weight basket. There is no monotone gradient from underwater to
  in-the-money — the central Grinblatt-Han prediction.
- The hedge HAC *t* is **−0.16**, and the block-bootstrap CI straddles zero. On this tape the
  disposition / overhang premium is **statistically absent** — it cannot clear the |t| ≥ 2
  inference bar, so the Signal axis reads **WEAK** (the literature is strong; *this* tape can't
  certify it).

## Is it just momentum? — the orthogonalised residual

Overhang is mechanically correlated with 12-1 momentum (a deep-in-the-money stock is, almost by
definition, a recent winner). Cross-sectionally regressing overhang on momentum each month and
re-sorting on the residual asks whether "disposition" carries any information beyond momentum:

| Sort | HAC *t* | annualised |
|---|--:|--:|
| Raw overhang hedge | −0.16 | −0.7% |
| Momentum-orthogonalised residual hedge | −0.65 | −2.3% |
| Overhang IC (Spearman, monthly) | −0.66 | mean IC −0.012 |

Removing momentum does not *rescue* the overhang signal — there is nothing underneath to
rescue. On this cross-section the factor is a non-event with or without the momentum control.

## Could you trade it? — cost sweep (one-way bps × NAV, 4× one-way notional/month)

| round-trip cost | net annualised | HAC *t* |
|---|--:|--:|
| 0 bps (gross) | −0.7% | −0.16 |
| 5 bps | −3.1% | −0.74 |
| 10 bps | −5.5% | −1.31 |
| 20 bps | −10.3% | −2.47 |

A long-short quintile book turning over ~4× one-way notional a month bleeds steadily as costs
rise; starting from a gross edge of essentially zero, there is no break-even to speak of.

## Synthetic positive control — the engine is a faithful detector

The offline deterministic generator confirms the harness *can* bank a planted disposition
premium (and finds nothing when none is planted), so the null real-tape result is a statement
about the tape, not a broken pipeline:

| Planted overhang premium | hedge HAC *t* | hedge Sharpe |
|---|--:|--:|
| 0.000 (null) | ~0.9 | ~0.2 |
| 0.004 (default control) | ~7 | ~1.4 |

In the momentum-only synthetic world (overhang premium 0, momentum priced), the *raw* overhang
hedge fires spuriously (it inherits momentum) and the momentum-orthogonalised residual collapses
— exactly the confound the real-tape control rules out.

## Verdict

- **Signal — WEAK.** Grinblatt & Han (2005) document a genuine, much-replicated overhang
  premium on the broad CRSP universe; on this small large-cap cross-section the hedge HAC
  *t* = −0.16 with a CI straddling zero. Strong literature, but *this tape cannot certify it* —
  WEAK by the desk's inference bar, not REAL.
- **Tradability — MIRAGE.** Gross return ≈ 0; survivorship-biased toward an *upper* bound and
  still flat; negative and increasingly significant once costs are charged. Nothing to harvest.
- **Just momentum in disguise? — CONFIRMED (as a confound).** Overhang is mechanically a
  momentum proxy; the orthogonalised residual is equally dead, so whatever (absent) signal there
  is is not separable from momentum on this tape.
