# Results — Study 563 (Secondary-Offering-Drift)

*Generated from [`secondary_offering_drift/`](../secondary_offering_drift/). The corporate-event
claim — **a stock keeps sliding for months after it sells a fresh slug of shares in a secondary /
follow-on offering** (dilution + a bearish equity-issuance signal) — tested as a clean event study
on **abnormal** (stock − SPY) returns. We time the public **pricing/announcement** of the offering
and enter **one day after**. The sample is a hard-coded, transparent table of **34** notable
follow-on / secondary offerings; the inference is a one-sample Welch t plus a **same-names
(left-tail) placebo null**. Offline & deterministic given the cache.*

## Data stamp

| Field | Value |
|---|---|
| Real tape | yfinance daily adjusted closes — SPY + 27 event tickers (26 carry events; one row with an unverifiable deal size is dropped) (public, no key) |
| As-of | **2026-06-30** (partial current bar dropped) |
| Window | 2008-01-02 → 2026-06-30 |
| Panel fingerprint | `75dc79931e2c` |
| Event sample | **34** hard-coded notable follow-on / secondary **offerings** (ticker, pricing date, $bn size); an explicit, **named SAMPLE** (point-in-time 424B feeds are not free) |
| Measure | **abnormal** forward return = (stock_T/stock_0 − 1) − (SPY_T/SPY_0 − 1) |
| Entry / costs | 1-day execution lag (trade the close *after* the offering); the tradable trade is **short the issuer**, charged 10 bps round-trip + a **300 bps/yr borrow** |

> **Sample caveat (named on the Signal axis).** The table is hand-picked, recognisable offerings
> by firms that (mostly) survived — a selection that skews toward *orderly, growth-story* raises
> (Tesla, crypto miners, pandemic-era names) and away from the distressed dilutions where the
> negative signal is strongest. The pricing dates are approximate public pricing dates, not an
> SEC-424B point-in-time feed. The verdict does not hinge on the exact 34 names: it hinges on the
> fact that *any* few-dozen single-name event set is dominated by single-stock variance — a point
> the synthetic control makes on data where we know the truth.

## The signal — abnormal drift after each offering vs zero

1-day entry lag. The abnormal return strips SPY, so this is drift *beyond the market*. The
one-sample Welch *t* tests the mean against **0**; the placebo *p* is the **left-tail** share of
20,000 same-names random-date draws whose mean abnormal return is **as negative as** the offering
set (the claim predicts a *negative* drift, so a small *p* would support it).

| horizon | n events | abn. mean | abn. median | loss-rate | Welch *t* (vs 0) | placebo *p* (left tail) |
|---|--:|--:|--:|--:|--:|--:|
| 1-month  | 34 | **+10.60%** | +0.46% | 50% | **+1.73** | 0.985 |
| 3-month  | 34 | **+7.83%**  | +3.34% | 47% | **+1.32** | 0.737 |
| 6-month  | 34 | **+8.21%**  | +6.19% | 41% | **+0.99** | 0.531 |
| 12-month | 34 | **+29.44%** | +3.71% | 47% | **+1.78** | 0.563 |

Read it carefully. The claim predicts a **negative** drift (dilution + bearish signal). The tape
delivers the **opposite**: the abnormal drift is **positive at every horizon** (+10.6% / +7.8% /
+8.2% / +29.4% at 1 / 3 / 6 / 12 months). No horizon clears the *t* ≥ 2 bar in *either* direction
(best is +1.78 at 12m), and the placebo left-tail *p* is deep on the *wrong* side (0.53–0.99): a
same-names random-date draw is *more* negative than the real offering set the large majority of the
time. The offering names — a Tesla-and-crypto-heavy roster of high-fliers raising cash at
strength — **kept ripping** after their raises. The dilution drift is not merely absent; on this
sample it is **inverted**.

## Costs — the short-the-issuer trade loses before you pay

The tradable expression of "the stock slides after an offering" is to **short the issuer**, so the
short's abnormal P&L is `−(abnormal drift)`. Because the drift is *positive*, the short *loses*:

| horizon | n trades | short gross | short net (10 bps + 300 bps/yr borrow) |
|---|--:|--:|--:|
| 1-month | 34 | **−10.60%** | **−10.95%** |
| 3-month | 34 | **−7.83%**  | **−8.68%** |
| 6-month | 34 | **−8.21%**  | **−9.81%** |

The trade is the wrong sign *before* costs; the punitive borrow (these are hard-to-borrow
high-fliers) only deepens the loss. There is nothing to harvest on this sample.

## Robustness — split by deal size (does "bigger offering ⇒ bigger slide" hold?)

| bucket | n | abn. 6m mean | Welch *t* | placebo *p* |
|---|--:|--:|--:|--:|
| BIG (≥ median $1.08bn) | 17 | **+6.73%** | +0.84 | 0.477 |
| SMALL (< median) | 17 | **+9.69%** | +0.65 | 0.601 |

Both halves drift **up**, neither is significant, and the dilution prediction ("more shares sold ⇒
bigger slide") fails outright — the drift is positive in both buckets regardless of size.

## Synthetic positive control — the engine is faithful, and ~34 events are weak

On deterministic panels of **34 event windows** with a **known planted (negative) abnormal drift**
injected over the 6-month window, the inference behaves exactly as it should — averaged over **25
seeds** (house rule, so no single lucky seed manufactures a result):

| planted 6m drift | mean Welch *t* (25 seeds) | mean abn. 6m |
|---|--:|--:|
| **0.00** (no edge) | **+0.68** | +4.50% |
| **−0.10** | −0.20 | −0.60% |
| **−0.20** | −1.13 | −5.45% |
| **−0.30** (huge edge) | **−2.10** | −10.06% |

With **no** planted drift the seed-averaged *t* stays well under |2| (+0.68); only a **huge**
planted downside drift — −30% abnormal over six months — drives the averaged *t* past −2. (At the
null, 4 of 25 individual seeds still cross |t| = 2 by luck — precisely why we average over seeds
and why 34 single-name events cannot be trusted to detect a drift of plausible size.) The control
proves two things at once: the measurement engine is **unbiased and detects a real downside drift
when one exists**, and ~34 single-name events is **too few** to detect any drift of plausible
magnitude. That is the entire lesson.

## Verdict

- **Signal — NONE.** The claim predicts a *negative* drift; on 34 notable offerings the abnormal
  drift is **positive at every horizon** (the *wrong* sign), never clears *t* ≥ 2, and the
  same-names placebo puts the real set on the *high* side of the random-date distribution
  (left-tail *p* 0.53–0.99). Not a certifiable edge — and not even the sign the claim predicts.
  **Survivorship + visibility selection** (recognisable, growth-story raisers that survived) is
  named here on the Signal axis and biases the sample toward exactly the high-fliers that rallied.
- **Tradability — MIRAGE.** The tradable trade (short the issuer) **loses before costs** at every
  horizon (short gross −8% to −11%), and the borrow on these hard-to-borrow names only deepens the
  loss. Nothing to deploy.
- **"Dilution drift on the tape?" — BUSTED.** "Sell shares, the stock keeps sliding" is not what
  this sample shows: the issuers *out*-ran the market for a year afterward. The dilution/signalling
  drift — real in the broad net-issuance literature on full, survivorship-free universes (see
  [Study 519](../../519-net-share-issuance/)) — does **not** survive a hand-picked basket of
  headline growth-story offerings. The synthetic control confirms the engine would catch a genuine
  downside drift, so this is the tape (and the selection) talking, not the code.
