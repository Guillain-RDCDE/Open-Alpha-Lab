# Results — Study 546 (Nobel-Announcement-Drift): sector CARs after the Nobel science prizes

*Generated from [`nobel_announcement_drift/`](../nobel_announcement_drift/) over this study's
cached yfinance tape: daily adjusted close for four sector ETFs (**XLV**, **IBB** → Medicine;
**XLK**, **SMH** → Physics/Chemistry) plus **SPY** (the market-model benchmark), 2000-01-03 →
2025-06-27 (fingerprint `ea4f6fc0a551`). Events: the **72** public announcements of the three
Nobel **science** prizes (Physiology/Medicine, Physics, Chemistry), **2001 → 2024** (events
fingerprint `d6630448011d`). Post-event **cumulative abnormal returns** (market-model residuals
vs SPY, trailing 120-day rolling beta) over the H=10-session window (t0, t0+H], t0 = first
session on/after the announcement. CAR table fingerprint `e7c04bc8ac6d`. As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE`

The folklore: when the Nobel Prizes are read out each October, the *thematically related* sector
should catch a news-attention bid and **drift** upward — pharma/biotech (XLV, IBB) after
**Medicine**, tech/semis (XLK, SMH) after **Physics** & **Chemistry**. We map each science prize
to its "supposed" sectors, measure the post-announcement CAR net of the market, and test the mean
against a random-October placebo.

The tape says **no drift — if anything the wrong way.** Across all 144 mapped (prize, sector)
events the mean 10-day CAR is **−0.33%** with a one-sample *t* of **−1.36** (not significant, and
the *wrong sign* for the folklore). The random-October **placebo *p* = 0.984**: the observed mean
CAR is *worse* than 98.4% of random-October draws (random October weeks actually drifted **+0.14%**
on average). So there is no post-Nobel bid — Nobel weeks, if anything, slightly *underperform* a
random October week. The one nominally significant cut is **Medicine → XLV/IBB at *t* −2.33** —
the *opposite* sign to the claim (pharma drifted *down*, not up, in the 10 days after the Medicine
prize). `NONE` on the signal axis (no positive drift; the only significant number is wrong-signed
and era-unstable), `MIRAGE` on tradability (the long-the-sector trade loses **−0.33%** gross,
**−0.37%** net, before you even count the noise).

## Data stamp

- **Prices**: XLV, IBB, XLK, SMH + SPY, daily adjusted close, 2000-01-03 → 2025-06-27,
  fingerprint `ea4f6fc0a551`
- **Events**: 72 Nobel science-prize announcements (3 prizes × 24 years, 2001-2024),
  fingerprint `d6630448011d`
- **CAR table** (H=10, mapped prize→sector, market-model AR vs SPY): 144 rows,
  fingerprint `e7c04bc8ac6d`

## The headline event study — no post-Nobel drift

| Cut | n | Mean 10-day CAR | one-sample *t* |
|---|---|---|---|
| **All mapped events** | 144 | **−0.33%** | **−1.36** |
| Medicine → {XLV, IBB} | 48 | −0.94% | −2.33 |
| Physics → {XLK, SMH} | 48 | +0.08% | +0.20 |
| Chemistry → {XLK, SMH} | 48 | −0.14% | −0.29 |

The folklore predicts a *positive* CAR at *t* ≥ 2. The full sample is slightly negative and
insignificant; the only cut that clears |*t*| = 2 is Medicine, and it does so with the **wrong
sign** — pharma/biotech drifted *down* after the Medicine prize.

## Per-sector detail

| Sector (prize) | n | Mean 10-day CAR | *t* |
|---|---|---|---|
| XLV (Medicine) | 24 | −0.67% | −2.50 |
| IBB (Medicine) | 24 | −1.22% | −1.58 |
| XLK (Physics/Chem) | 48 | +0.27% | +1.10 |
| SMH (Physics/Chem) | 48 | −0.33% | −0.60 |

No sector shows a positive, significant post-Nobel drift. The health names actually *fell*.

## The random-October placebo — Nobel weeks are worse than random

| | value |
|---|---|
| Observed mean CAR | **−0.33%** |
| Placebo mean CAR (random October days, 2000 draws) | **+0.14%** |
| Placebo *p* (share of draws ≥ observed) | **0.984** |

A random week in the same October beats the actual Nobel-announcement week 98.4% of the time.
There is no announcement-attention bid; the "drift" folklore is not on the tape.

## Robustness — the sign is unstable and never positive-significant

| Horizon H | Mean CAR | *t* |
|---|---|---|
| 3 | −0.01% | −0.07 |
| 5 | −0.24% | −1.42 |
| 10 (headline) | −0.33% | −1.36 |
| 21 | +0.13% | +0.32 |

| Variant | Mean CAR (H=10) | *t* |
|---|---|---|
| Entry at t0 (headline) | −0.33% | −1.36 |
| Entry at t0+1 (one bar later) | −0.31% | −1.26 |
| Pre-2013 events | +0.30% | +0.74 |
| Post-2013 events | **−0.97%** | **−3.60** |

At no horizon is the drift positive-significant. The entry-lag variant changes nothing. The
pre/post-2013 split *flips sign* (weakly positive early, significantly **negative** late) — a
signal whose sign depends on the era is not a signal.

## Costs

| | value |
|---|---|
| Gross mean CAR (H=10) | **−0.33%** |
| Net (2 bps/side round-trip, long-only ETF, no borrow) | **−0.37%** (net *t* −1.53) |

The trade is negative before costs; frictions only deepen the loss. Nothing to harvest — `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted post-event `drift` (per day) | Mean CAR-*t* (25 seeds) | |
|---|---|---|
| 0.0000 (null) | **+0.19** | flat — no false signal |
| 0.0005 | +0.89 | drift emerging |
| 0.0010 | +1.60 | drift visible |
| 0.0015 | **+2.30** | clears the bar |
| 0.0020 | +3.01 | strong |

At the null the CAR-*t* is ≈ 0; planting a genuine post-event drift drives it positive and past +2
as it grows. The detector works — so the flat/negative real-tape result is the tape talking, not a
broken engine. (Control only; never cited for the real-tape stamp.)

## Why the drift doesn't certify here

1. **The thematic map is a stretch.** A Nobel Medicine prize honours decades-old basic research,
   not a tradable catalyst for a broad health-care ETF; a Physics prize on, say, black-hole imaging
   moves no semiconductor earnings. The "attention bid" has no cash-flow channel, so there is no
   reason for a persistent CAR — and there isn't one.
2. **Tiny event count, huge October noise.** 24 announcements per prize over a quarter-century, in
   a month that carries its own well-known volatility (earnings season, the "October effect"). The
   placebo shows a random October week is a *better* bet than the Nobel week.
3. **Sign instability.** The one significant cut (Medicine, negative) and the post-2013 split are
   *negative*; the early era is weakly positive. Nothing survives across horizons, entry lags and
   eras.

## The honest takeaway

The Nobel-announcement-drift folklore does not appear on the tape. The average post-announcement
sector CAR is slightly **negative** (−0.33%, *t* −1.36), *worse* than a random October week
(placebo *p* 0.984), with the only significant cut (Medicine → health, *t* −2.33) pointing the
**wrong way** and the sign flipping across eras. `NONE` × `MIRAGE`. The synthetic control confirms
the engine would catch a real drift — so this is the absence of an effect, not a blind spot.
