# Results — Study 564 (Short-Report-Event)

*Generated from [`short_report_event/`](../short_report_event/) over this study's cached yfinance
tape. The folklore that a stock **craters and stays down** after an activist short-seller
publishes a hit piece, tested on a **transparent, hardcoded basket of famous short-report
campaigns** (Hindenburg, Muddy Waters, Citron, and adjacent research firms). For each report we
pull the target's and SPY's daily adjusted closes (yfinance, no key), measure the **report-day
crater** (day-0 return — the allegation being priced in) and the **post-report excess drift** vs
SPY if you **short one day after** the report and hold 1 / 3 / 6 months. As-of **2026-06-30**;
price fingerprint `946a53954ab8`.*

## The verdict, earned — Signal `WEAK` · Tradability `FRAGILE` · "Free short?" `BUSTED`

When an activist short-seller publishes a hit piece, does the stock actually fall and stay down?
**Directionally, yes — but the mean short is destroyed by fat-tailed short squeezes.** The
post-report excess drift is **negative at every horizon** (the short works on average), the
**median** target underperforms SPY by **−42.7%** over six months, and the **hit-rate is 64–77%**
— the short is right far more often than a coin flip. But the mean-based Welch *t* **never clears
the |t| ≥ 2 bar** (best is 3-month **t = −1.55**), because a handful of names *squeeze* violently
(Carvana ran **+318%** excess after being left for dead), dragging the mean back toward zero. So
`WEAK` on the signal axis (right sign, high hit-rate, but no robust *t* ≥ 2 on the mean), `FRAGILE`
on tradability (a short with unbounded left-tail loss, a punitive borrow, on a rare event), and
`BUSTED` on the "free short" myth (the memorable take-downs are exactly what survivorship looks
like — and the delisted winners you'd want are *missing* from the free feed).

## Data stamp

| Field | Value |
|---|---|
| Real tape | yfinance daily adjusted closes — activist short-report targets + SPY (public, no key) |
| Event set | **hardcoded** table of 32 well-known short-report campaigns, 2015–2023 reports |
| Priced | **22 of 32** targets (10 delisted/acquired names dropped from the free feed — see caveat) |
| Window | 2010-01-04 → 2026-06-26 (**4,145** trading days; ~16.5 years of price tape) |
| Crater | day-0 return of the target (first trading day on/after the report) |
| Drift | target − SPY excess return; **1-day** short-entry lag; hold 1 / 3 / 6 months |
| Costs | one 10-bps round-trip + an **800 bps/yr short borrow** (punitive, pro-rated) |
| Price fingerprint | `946a53954ab8` |

> **Selection + survivorship caveat (named on the Signal axis).** A hardcoded basket of *famous*
> short reports is **selected on outcome**: campaigns enter folklore precisely because they were
> prominent, and many because the short *worked* (Nikola, Lordstown, Luckin). That bias runs **in
> favour** of the claim — so this is the *easiest* possible test. And worse: **10 of 32 targets
> (NKLA, RIDE, TWTR, MULN, XL, EBIX, SESN, CVAC, FUV, WKHS-era names) delisted or were acquired**
> and drop from the free yfinance feed — and those are disproportionately the *biggest short
> wins* (a fraud that goes to zero de-lists). So the surviving 22-name tape is biased **against**
> the short even as the roster is biased **toward** it. Both effects are named here; the verdict
> does not hinge on the exact roster.

## The report-day crater — real news, but you can't short it

The day-0 return is the market re-pricing the allegation. You cannot capture it unless you were
already short before the report dropped (you weren't), so it is **not a strategy** — it is the
efficient-market re-pricing. We report it to show the news is real:

| | n | mean | median | negative-share | one-sample *t* |
|---|--:|--:|--:|--:|--:|
| report-day move (day 0) | 22 | **−1.97%** | −0.27% | 55% | **−0.75** |

A ~2% average drop, negative more often than not — but even *this* uncapturable day-0 move is only
**t = −0.75** on the surviving basket (the biggest craters de-listed and are gone; several names
had already been falling before the report). The news matters; on the survivors it is small and
noisy.

## The signal — post-report excess drift vs SPY (the only tradable leg)

Short at the close **one day after** the report (no look-ahead), hold H months, measure the target's
return **in excess of SPY** over the same window. A *negative* excess is the short working. The
placebo *p* is the share of 20,000 random-date baskets — same horizon, **same targets** — whose mean
excess is at least as **negative** as the report basket (so it nulls out the targets' own drift,
isolating the *report timing*).

| horizon | n | mean excess | median excess | hit-rate (short works) | Welch *t* | placebo *p* |
|---|--:|--:|--:|--:|--:|--:|
| 1-month | 22 | **−1.63%** | −9.13% | 77% | **−0.22** | 0.359 |
| 3-month | 22 | **−11.55%** | −18.52% | 64% | **−1.55** | 0.068 |
| 6-month | 22 | **−14.58%** | −42.68% | 73% | **−0.81** | 0.095 |

Read it carefully. The drift is **negative at every horizon** — the folklore points the right way,
and the **median** target is *crushed* (−42.7% under SPY at six months). The **hit-rate is 64–77%**:
the short is right far more often than a coin flip. But the **mean** *t* **never clears −2** (best is
3-month **t = −1.55**), because the distribution is violently **right-skewed** — a few names squeeze
so hard they wipe out the median. The gap between the median (−42.7%) and the mean (−14.6%) at six
months *is the story*: activist shorting has a great hit-rate and an ugly left tail (for the short,
the *right* tail of the target's return).

## Why the mean fails — the short squeeze

| top squeezer (6m excess vs SPY) | excess |
|---|--:|
| **CVNA** (Carvana) | **+318.2%** |
| PDD (Pinduoduo) | +50.5% |
| ADT | +36.1% |
| IQ (iQIYI) | +22.4% |

Carvana was left for dead in December 2022 and then rallied roughly ten-fold — a single +318%
excess that, on a 22-name book, swamps the eighteen names where the short *worked*. This is the
defining feature of activist shorting: **bounded gains (the stock can only fall to zero) against
unbounded losses (a squeeze can 10x you)**. Drop the three biggest squeezers and the 6-month *t*
snaps to **−4.81** — but removing your worst three trades *after the fact* is exactly the
cherry-picking the placebo and the honest verdict refuse to do.

## Short book — net of costs and a punitive borrow

The short book's P&L is **−excess** (you profit when the target underperforms SPY), charged one
10-bps round-trip plus an **800 bps/yr borrow** (freshly-crashed allegation-hit names are the
hardest, most expensive borrow):

| horizon | n | gross short P&L | net (10 bps + 800 bps/yr borrow) |
|---|--:|--:|--:|
| 1-month | 22 | +1.63% | **+0.87%** |
| 3-month | 22 | +11.55% | **+9.45%** |
| 6-month | 22 | +14.58% | **+10.48%** |

The *mean* short book is **positive net of a heavy borrow** — but that mean carries the CVNA-style
left-tail risk above, so a positive average is not a bankable edge. The borrow (a footnote here in
aggregate) is exactly the friction that makes a name-by-name short punishing precisely when the
squeeze is running against you.

## Robustness — flat in the entry lag

| entry lag (6m) | n | mean excess | Welch *t* | placebo *p* |
|---|--:|--:|--:|--:|
| lag = 0 (same-day close) | 22 | −10.61% | −0.51 | 0.164 |
| **lag = 1 (headline)** | 22 | **−14.58%** | **−0.81** | 0.095 |
| lag = 2 | 22 | −14.45% | −0.75 | 0.097 |
| lag = 5 | 22 | −14.19% | −0.69 | 0.101 |

Shifting the short entry from same-day to a week later barely moves the mean or the *t* (it never
approaches −2), so the result is not an execution artefact — the drift is a genuine, slow,
right-skewed underperformance that a one-bar entry change does not create or destroy.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

On a deterministic set of **30 synthetic short-report events** (each target a high-beta GBM path
sharing a market factor, a report-day crater on day 0, and a *known* post-report excess drift). The
house rule averages the Welch *t* over **25 seeds** so no single lucky RNG seed can manufacture (or
hide) significance:

| planted 6-month drift | mean Welch *t* (25 seeds) | reads as |
|---|--:|---|
| **0.00** (no edge) | **+0.70** | flat — no false signal |
| −0.10 | −1.01 | short edge emerging |
| −0.20 | **−2.91** | clears the bar |
| −0.35 | **−6.10** | unmistakable |

At the null the mean *t* is ≈ 0; planting a genuine negative drift (the short edge) drives *t*
negative and past −2 as it grows. The detector works — so the real-tape result is a statement about
**this survivor basket on these events**, not a broken engine. *(A single seed-564 draw at the null
happened to print t = +2.1 on 30 high-vol names — which is exactly why the house rule averages ≥ 20
seeds; the seed-robust mean is +0.70. Control only; never cited for the real-tape stamp.)*

## The honest takeaway

Activist short reports **do** knock the stock down — the median target underperforms SPY by −42.7%
over six months and the short is right 64–77% of the time. But this is **not a bankable edge**:
(1) the **mean** is destroyed by fat-tailed short squeezes (one Carvana +318% swamps eighteen
winners), so no horizon clears a robust *t* ≥ 2 (best −1.55); (2) the payoff is **asymmetric** —
bounded gains, unbounded losses — which no position-sizing fixes; (3) the basket is **selected on
fame** (bias *for* the claim) yet the biggest wins **delisted** and are missing (bias *against*),
so the surviving tape both flatters and starves the test; and (4) the borrow on a crashed name is
punitive exactly when you most need to hold. `WEAK` × `FRAGILE`, with the "free short" myth
`BUSTED`. The synthetic control confirms the engine would catch a real, stable short edge — so this
is the tape talking: a high hit-rate wrapped around a left tail that eats the mean.
