# References & literature map -- Study 224 (Monday Effect)

## The claim under test

The "Monday Effect" (a.k.a. the "day-of-week effect"): the strong, sold-at-full-strength
version is that **Monday close-to-close returns are systematically negative** -- the market
drifts down over the weekend and continues falling into Monday's close -- so a simple
calendar rule ("avoid Monday" or "buy only on the best day") **beats buy-and-hold**.

- Popular framing: Investopedia, *"Monday Effect"*:
  <https://www.investopedia.com/terms/m/mondayeffect.asp>
- The "Monday Effect" / "Weekend Effect" are staples of introductory finance folklore and
  frequently cited in market-seasonality discussions.

## Why the steelman is almost coherent

- **The Monday Effect is one of the best-documented seasonal anomalies in the early
  academic literature.** Kenneth R. French, *Stock Returns and the Weekend Effect*,
  Journal of Financial Economics 8 (1980), pp. 55-69, found significantly **negative average
  Monday returns** on the S&P composite over 1953-1977 using close-to-close returns. This
  is the canonical citation; the data were clean and the effect was large.
- Michael R. Gibbons and Patrick Hess, *Day of the Week Effects and Asset Returns*,
  Journal of Business 54 (1981), pp. 579-596, confirmed the pattern across the S&P 500 and
  individual stocks.
- A plausible micro-structure story exists: settlement timing, clustering of bad corporate
  news after Friday's close, and dealer inventory effects could push weekend-spanning
  close-to-close returns negative.

## Why it is likely to fail *as stated* ("beats buy-and-hold today")

- **The effect famously decayed -- and in many samples reversed -- after publication.**
  Abraham and Ikenberry, *The Individual Investor and the Weekend Effect*,
  Journal of Financial and Quantitative Analysis 29 (1994); Mehdian and Perry,
  *The Reversal of the Monday Effect: New Evidence from US Equity Markets*,
  Journal of Business Finance and Accounting 28 (2001), pp. 1043-1065; Olson, Mossman
  and Chou (2015) all document the weakening or disappearance of the Monday Effect in
  modern U.S. data, consistent with arbitrage after publication.
- **The SPY tape (1993-present) is a strictly post-publication sample.** French found the
  effect on 1953-1977 data and published in 1980. Testing on 1993-2026 asks whether the
  effect survived decades of awareness and professional arbitrage -- the answer is no.
- **Calendar-of-the-week rules are a selection minefield.** Five weekdays, five tests; the
  "worst" weekday is the minimum of five noisy sample means and can clear a naive threshold
  by construction. Without a snooping correction, a negative Thursday that looks like a
  Monday on bad-luck data is indistinguishable.
- **Even a real per-day tilt rarely survives as a tradable rule.** A "buy Monday only"
  rule sits in cash 81% of the time and forfeits the equity premium on the other days --
  the lost beta dwarfs any weekday tilt, before a single basis point of cost.

## Relationship to Study 90 (Weekend Effect)

Study 90 tested the *overnight* (close-to-open, Friday-close to Monday-open) return to
isolate the *weekend gap* specifically. This study (224) uses *close-to-close* returns --
the full intra-day move is included -- which is how French (1980) measured the effect.
Despite the different return definition, both studies reach the same verdict on the SPY
tape: **no negative Monday**.

## Method lineage

- **Newey-West HAC standard errors** on the mean of an autocorrelated return series and
  for the difference of means across weekday groups and sub-periods: Newey and West (1987),
  *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix*, Econometrica 55, pp. 703-708.
- **Test of the *difference* across a pre-registered split** (pre-2000 vs post-2000) rather
  than two separately-reported sub-period means -- the desk rule that a "decayed since..."
  claim must carry a test of the change, on a justified, not snooped, split.

## Data sources used

- **SPY**, daily, **total-return adjusted** (dividends folded in) via `quantlab.data`
  (Yahoo Finance), cached to parquet under `_cache/`. Cash is assumed to earn **0%** --
  a stated, conservative choice. The SPY tape starts **1993-01-29** (ETF inception), so
  this is entirely a post-publication sample of French's 1980 effect.

## Related desk studies

- [Study 90 -- Weekend](../../90-weekend/) -- the overnight (close-to-open) version of the
  same effect; same verdict, different return definition.
- [Study 89 -- Turn-of-the-Month](../../89-turn-of-the-month/) -- another calendar anomaly
  from the same era.
