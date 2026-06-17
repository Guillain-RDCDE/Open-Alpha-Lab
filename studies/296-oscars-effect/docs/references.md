# References & literature map — Study 296 (Oscars-Effect)

## The claim under test

There is no canonical academic paper claiming "the S&P 500 reacts to the Academy
Awards" — which is itself the point. The Oscars-Effect is *folklore by analogy*: the
ceremony is one of the most-watched live broadcasts in the United States, the winning
films and studios enjoy a measurable post-Oscar box-office and streaming bump, and so
the popular intuition ("a giant cultural event must move the market") gets transplanted
onto the broad index. We test the cleanest tradeable version: an event study of ^GSPC
on the first session after the ceremony.

## Where there *is* a real effect — and where there isn't

- **The box-office / "Oscar bump" is real, and local.** Nelson, Donihue, Waldman &
  Wheaton (2001), "What's an Oscar Worth?" (*Economic Inquiry* 39(1), 1–16) document a
  substantial revenue uplift for nominated and winning films. This is a *firm-level,
  product-level* effect on a single film's theatrical run — not an index-level signal.
- **Single-studio share-price studies are mixed and tiny.** Event studies on the listed
  parents of winning studios (Disney, Sony, Lionsgate, Warner) occasionally find small,
  short-lived abnormal returns, but the winning film is usually a rounding error in a
  diversified media conglomerate's cash flows, and the result is known to be priced long
  before the telecast (bookmaker odds and guild awards are highly predictive). The market
  has nothing new to learn on Oscar night.
- **The broad index has no mechanism.** Best Picture does not move aggregate corporate
  earnings, rates, or risk premia. Any "Oscars-Effect" on ^GSPC would be a calendar
  coincidence, in the same family as the Super Bowl Indicator.

## Why a broad-index reaction is implausible on priors

- **Information is pre-released.** By ceremony night, precursor awards (Golden Globes,
  SAG, PGA, DGA, BAFTA) and betting markets have already collapsed the uncertainty about
  the major winners. Markets price information when it arrives; here it arrives weeks
  earlier and in pieces.
- **Tiny n.** Only 31 ceremonies fall inside a clean ^GSPC daily window (1995–2025).
  With ~110 bps daily S&P volatility, the minimum detectable one-day mean at |t| = 2 is
  roughly 40 bps — larger than any plausible Oscar effect.
- **Multiple windows / specs.** Day-0, day-after, CAR over several windows, by genre —
  each is a separate test; searching across them inflates false-discovery risk. We fix a
  single primary spec (event day 0) in advance and treat the rest as illustration.

## Method lineage

- **Event study.** The standard MacKinlay (1997), "Event Studies in Economics and
  Finance" (*Journal of Economic Literature* 35(1), 13–39) framework: define event day 0,
  compute abnormal returns vs a benchmark expected return (here a constant full-tape mean,
  the simplest market model), aggregate into average abnormal returns (AAR) and cumulative
  abnormal returns (CAR).
- **HAC / Newey-West.** `_hac_tstat` applies a Bartlett-kernel long-run-variance
  correction so the event-day t-stat is robust to any mild serial dependence; with one
  observation per ceremony and n = 31 the correction is small but honest.
- **Permutation / placebo control.** Relabel 5,000 random sets of 31 sessions as
  "post-Oscar" and ask how often a random set's mean swing matches the real one — a
  distribution-free check that the observed mean is not a tail event.
- **Costs & lag.** One execution lag (day 0 = first full session after the broadcast);
  one-way costs charged on entry and exit of the tradable rule; the tape is price-only
  ^GSPC (no dividends), labelled as such.

## Data sources

- **^GSPC daily.** Yahoo Finance auto-adjusted close, close-to-close simple returns,
  staged at `_cache/oscars_gspc_daily.parquet` (the loader also reads the repo-level
  `_cache/chinese_zodiac_GSPC_daily.parquet`). Price-only — no dividend reinvestment,
  consistent with a one-day event window where dividends are negligible.
- **Oscar-ceremony table.** Hardcoded in `data.py`. Sources: the Academy of Motion
  Picture Arts and Sciences (oscars.org) and Wikipedia "List of Academy Awards
  ceremonies" for ceremony dates and Best Picture winners.

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the canonical "a cultural event
  predicts the market" folklore — same base-rate/tiny-n teardown, different sport.
- **[Study 164 — Mercury-Retrograde](../../164-mercury-retrograde/)**: the same daily
  ^GSPC event/regime machinery applied to financial astrology.
- **[Study 165 — Chinese-Zodiac](../../165-chinese-zodiac/)**: calendar folklore on the
  same daily tape; another null dressed as an omen.
