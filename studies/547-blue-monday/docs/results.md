# Results — Study 547 (Blue-Monday): does the 'most depressing day' sag the market?

*Generated from [`blue_monday/`](../blue_monday/) over this study's cached yfinance tape: daily
adjusted-close (total-return) **SPY**, 8,409 sessions **1993-01-29 → 2026-06-26**, fingerprint
`d37d9f8a7153`. 'Blue Monday' is the **third Monday of January** (Cliff Arnall, 2005). As-of
**2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Is Blue Monday tradable at all?" `BUSTED`

'Blue Monday' — the third Monday of January, sold as the most depressing day of the year — is a
pseudo-scientific PR label. If seasonal mood really moved markets (the SAD thesis, Study 150) its
returns should **sag** and realised **volatility** should **spike** versus an ordinary Monday. Two
findings kill the test:

1. **Blue Monday is almost always a market holiday.** Since 1986 the US observes **MLK Day** on the
   third Monday of January — the *same day* — and equity markets have closed for it since 1998. On
   the 1993-2026 SPY tape, **30 of 34** Blue Mondays fall on non-trading days. Only **4** Blue
   Mondays (1994-1997, before markets closed for MLK) are actually tradable, giving a literal-day
   difference of **+7.2 bps** (*t* +0.41) on a sample far too thin to mean anything. **You cannot
   trade a day the market is shut.**

2. **The tradable proxy — the next open session (the Tuesday after Blue Monday) — hints at a dip
   but fails every bar.** Across all 33 Blue-adjacent days the mean return is **−22.0 bps** versus
   **+4.8 bps** on all other days, a difference of **−26.8 bps** — the *right direction* for a mood
   dip, but the Welch *t* is only **−1.24** (below |t| ≥ 2), the random-day placebo *p* = **0.184**,
   the realised-vol difference is a flat **+1.1 vol points** (no spike), and the sign **flips in
   every sub-period** (+27, −101, +20, −59 bps). No stable, significant effect.

So `NONE` on the signal axis (no |t| ≥ 2, insignificant placebo, sign-unstable, no vol spike),
`MIRAGE` on tradability (the literal day is untradable — market closed — and the proxy is a
sign-flipping coin toss), and `BUSTED` on the myth: Blue Monday is not a market event, it is a
holiday.

## Data stamp

- **Prices**: SPY daily adjusted close (total-return), 8,409 sessions, 1993-01-29 → 2026-06-26,
  fingerprint `d37d9f8a7153`
- **Blue-Monday calendar**: third Monday of January per year; 30/34 collide with MLK Day
  (non-trading), 4 are tradable (1994, 1995, 1996, 1997)

## The literal Blue-Monday test — untestable (market closed 30 of 34 years)

| | value |
|---|---|
| Blue Mondays in span | 34 |
| Blue Mondays that were **trading days** | **4** (1994-1997) |
| Blue Mondays on MLK holiday (market closed) | **30** |
| Blue-Monday mean (n=4) | **+12.9 bps** |
| Other-Monday mean (n=1,576) | **+5.7 bps** |
| Difference (Blue − other) | **+7.2 bps** (Welch *t* **+0.41**) |

Four observations is no sample. The claim's own headline day is one the market does not trade.

## The tradable proxy — the day after Blue Monday (usually Tuesday)

Because Blue Monday is nearly always shut, the market's first chance to price the 'most depressing
day' is the next open session. That restores a full 33-observation sample.

| | value |
|---|---|
| Blue-adjacent days (n) | **33** |
| Blue-adjacent mean return | **−22.0 bps** |
| Other-day mean return (n=8,375) | **+4.8 bps** |
| Difference (Blue-adjacent − other) | **−26.8 bps** |
| Welch two-sample *t* | **−1.24** (bar is |t| ≥ 2) |
| Random-day placebo *p* | **0.184** |
| Blue-adjacent realised vol (ann.) | **19.7%** |
| Other-day realised vol (ann.) | **18.6%** |
| Vol difference | **+1.1 pts** (no meaningful spike) |

The direction is *consistent* with a mood dip — but it is one standard error shy of nothing, the
placebo says a random calendar day matches it 18% of the time, and the vol 'spike' is a rounding
error.

## Robustness — the sign is not stable

| Sub-period (Blue-adjacent day) | Difference vs other days | Welch *t* | Reads as |
|---|---|---|---|
| 1993-2001 | **+26.9 bps** | +1.05 | *up* (anti-dip) |
| 2001-2009 | **−101.3 bps** | −1.51 | dip |
| 2009-2018 | **+20.0 bps** | +1.24 | *up* (anti-dip) |
| 2018-2026 | **−58.8 bps** | −1.50 | dip |

The Blue-adjacent day is *up* in two sub-periods and *down* in two — a sign that alternates with
the calendar is not a signal. The full-sample −26.8 bps is the average of noise, not a stable mood
effect.

## Tradability — you can't trade a closed market

| Strategy | CAGR | Sharpe | Switches |
|---|---|---|---|
| Buy-and-hold SPY | **10.76%** | **0.643** | 0 |
| Sit out the (4 tradable) literal Blue Mondays | **10.74%** | **0.642** | 9 |

The literal timer is indistinguishable from buy-and-hold — there are only four days to act on, and
even those show no edge. Shorting the sign-flipping proxy day (net of 1 bp/switch + 100 bps/yr
borrow) is a coin toss with a fee. `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

On a deterministic synthetic tape where Blue Monday *is* a trading day, we plant a mood dip
(`blue_dip` subtracted from every third-January-Monday return) and average the Welch *t* over 25
seeds:

| Planted `blue_dip` | Mean Welch *t* (25 seeds) | |
|---|---|---|
| 0.000 (null) | **−0.32** | flat — no false signal (→ −0.01 over 100 seeds) |
| 0.001 | −0.96 | dip emerging |
| 0.002 | −1.60 | dip visible |
| 0.003 | −2.25 | clears the bar |
| 0.004 | **−2.89** | unmistakable |

At the null the *t* sits at ≈ 0 (−0.01 averaged over 100 seeds); planting a genuine dip drives *t*
negative and past −2 as it grows. The detector works — so the real-tape non-result is a statement
about the *tape*, not a broken engine. (Control only; never cited for the real-tape stamp.)

## Why the myth doesn't survive

1. **'Blue Monday' is a PR invention, not a finding.** Cliff Arnall's 2005 'equation' for the most
   depressing day was a Sky Travel marketing stunt with no scientific basis — psychologists have
   repeatedly debunked it. There is no reason to expect a market echo.
2. **It is a market holiday.** The one testable prediction runs into MLK Day: the market is *closed*
   on the third Monday of January in 30 of the last 34 years. The claim self-destructs on the
   calendar.
3. **Even the proxy fails the bar.** The next-open-session dip (−26.8 bps, *t* −1.24, placebo *p*
   0.18) is directionally suggestive but statistically empty and sign-unstable across sub-periods —
   the same fate as the SAD-effect it is a cousin of (Study 150).

## The honest takeaway

Blue Monday is a *marketing* holiday that happens to be a *market* holiday. The literal day is
untradable (market shut 30 of 34 years, n=4 otherwise); the tradable next-day proxy leans the way
the mood story predicts (−26.8 bps) but never clears |t| ≥ 2, has an insignificant placebo, no vol
spike, and flips sign in every sub-period. `NONE` × `MIRAGE`, myth `BUSTED`. The synthetic control
confirms the engine would catch a real dip — so this is the tape (and the calendar) talking, not
the code.
