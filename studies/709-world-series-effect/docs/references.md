# References & literature map — Study 709 (World-Series-Effect)

## The claim under test

- **The folklore.** Every October, financial media half-jokingly revives a family of
  "sports omens" for the stock market. The best-known is the **Super Bowl Indicator**
  (Krueger & Kennedy, 1990, *An Analysis of the Super Bowl Stock Market Predictor*, *The
  Journal of Finance* 45(2), 691–697): an NFC/original-NFL win supposedly means a bull
  year, an AFC/AFL win a bear year. The **World Series version** — sometimes floated as
  "does the AL or NL pennant predict the market" — is its baseball cousin, and the
  brief's **city-mythology variant** ("a New York team wins, Wall Street's hometown club
  is hot, stocks follow") is the same species of story with the mascot swapped for a
  skyline. Neither the league nor the city variant has ever had a published economic
  mechanism proposed for it — unlike, say, this desk's FOMC vol-crush study (637), whose
  claim comes with a one-sentence causal story (event-premium expiry). That absence is
  itself diagnostic.
- **Why calendar omens keep reappearing.** Krueger & Kennedy's own paper is instructive:
  their headline 28-for-28 (at the time) Super Bowl streak was a coincidence of the NFC's
  on-field dominance coinciding with a secular bull market, not a causal channel — and it
  broke down repeatedly once tested against the correct baseline and extended
  out-of-sample. The methodological lesson (test against the *unconditional* up-rate, not
  a 50% coin; watch for a streak that is really an era effect) is exactly what this study
  applies to baseball.

## What we measure, and the honesty rails

- **Two variants, one lag convention.** The primary signal (NL win → bullish) mirrors the
  NFC mnemonic; the secondary signal (New York franchise win → bullish) is the brief's
  named "champion-city" alternative. Both use the identical, single execution
  convention: enter at the World Series season's December 31 close, hold through the
  following calendar year — the champion is public information weeks before that date,
  so there is zero look-ahead.
- **The correct baseline is the unconditional up-rate, not 50%.** The S&P has closed up
  in roughly three years out of four over this sample; any "predict bullish" signal
  inherits that base rate for free. The binomial test therefore uses **p₀ = the sample's
  own unconditional up-rate** (73.0%) as the null, following the same correction Study
  158 (Super-Bowl) applies to the NFC/AFC indicator — the single most common error in
  popular write-ups of these omens.
- **Permutation, not just a parametric t.** With only 35-74 events split into two groups,
  a 20,000-draw permutation test on the two-sided mean contrast sidesteps any normality
  assumption and is reported alongside the Welch *t*.
- **A separate "myth-check" against a flat coin.** Because the folklore itself is
  usually stated informally ("does it call the market right?"), we also report a looser,
  two-sided test against p=0.5 — purely as the grey third axis, never used to certify the
  Signal stamp.
- **No survivorship.** ^GSPC is a price index; every played World Series 1950-2025 is
  included, with 1994 (players' strike, no Series) named and dropped rather than silently
  imputed, and the still-open 2025 champion's unscoreable "next year" excluded rather than
  force-filled.

## Data sources

- **^GSPC daily close** — yfinance (no key), cached under `_cache/` (`wse_gspc.csv`),
  1949-11-01 → 2026-06-30, resampled to December-to-December calendar-year returns
  (price-only — the S&P 500 index carries no dividends; this is a price return, not a
  total return, and is labeled as such throughout).
- **World Series champions, 1950 → 2025**, hardcoded in
  [`data.py`](../world_series_effect/data.py). Source: MLB official postseason history /
  Baseball-Reference (https://www.baseball-reference.com/postseason/world-series.shtml),
  cross-checked against Wikipedia's "List of World Series champions". League and host-city
  assignments follow each franchise's status in the season it won (e.g. Brooklyn Dodgers
  1955, Milwaukee Braves 1957, Anaheim Angels 2002, Florida Marlins 1997/2003).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [158-super-bowl](../158-super-bowl/) — the **original** sports-omen debunk this study
  is modeled on: NFC/AFC win vs S&P return, same base-rate correction, same "None /
  Mirage / Busted" verdict shape. This study runs the identical playbook on a different
  sport and adds the city-mythology variant; the conclusion rhymes on purpose — that's the
  point of testing the whole indicator *family*, not a coincidence in the write-up.
- [235-world-cup-effect](../235-world-cup-effect/) — a different mechanism entirely: it
  tests whether the S&P **drifts during the ~5-week World Cup window itself** (a
  confounded macro-crisis story, not a winner effect) — no "who won" signal at all. This
  study, by contrast, tests a **post-event winner/hometown label predicting the following
  calendar year** — the Super Bowl Indicator's structure, not the World Cup study's.
- [234-olympic-year](../234-olympic-year/) — the same test design (hardcoded event-year
  table vs ^GSPC annual returns, HAC/permutation inference, an event-year timing
  strategy) applied to a *symmetric* calendar marker (every Olympic year) rather than a
  binary winner-vs-loser split. This study borrows its annual-return machinery almost
  directly.
- [708-eurovision-effect](../708-eurovision-effect/) — the same omen family transplanted
  to a non-sporting annual contest (the Eurovision Song Contest), testing whether the
  host country's or winner's market reacts. A sibling debunk, different underlying event.

None of the siblings test the **World Series champion's league or hometown** — this
study's own, previously-untested corner of the sports-omen family.
