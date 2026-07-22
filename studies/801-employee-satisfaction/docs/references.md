# References & literature map — Study 801 (Employee Satisfaction)

## The claim under test

- **The source paper.** Alex **Edmans (2011)**, *"Does the stock market fully value
  intangibles? Employee satisfaction and equity prices,"* Journal of Financial Economics
  101(3): 621-640. A value-weighted portfolio of the **"100 Best Companies to Work For in
  America"** (the Fortune / Great Place to Work annual list, published since 1998) earned a
  **four-factor (Carhart) alpha of ~3.5%/yr, 1984-2009** — Edmans reads this as the market
  *under*-pricing employee satisfaction, an intangible that shows up in future earnings
  surprises. Follow-up: Edmans, Li & Zhang (2014, cross-country) find the premium concentrated
  where labour markets are flexible, and Edmans himself has noted the US alpha **shrank** in the
  years after the finding was published — the classic post-publication decay we test for here.
- **The mechanism, steelmanned.** Satisfied employees → lower turnover, higher productivity and
  better service → earnings that beat expectations *later*; if the market treats "soft" HR
  intangibles as noise, a list of the best employers is a cheap proxy for mispriced future
  cash-flows. This is a genuine, thoughtfully-argued efficient-markets anomaly, not folklore.
- **The open question we test.** On a *modern* tape (2016-2026), with an honest **survivorship**
  accounting, does a basket of perennial list members still earn **risk-adjusted** alpha — or is
  the outperformance just market beta and factor tilt on a hand-picked set of winners?

## What we measure, and the honesty rails

- **Market-model (CAPM) alpha**, not four-factor. We regress the equal-weight basket's monthly
  total return on SPY and read the intercept, with a **Newey-West (1987)** 3-lag HAC *t* as the
  decisive statistic and an OLS *t* cross-check. We are explicit that controlling only for the
  market makes this an **upper bound** on any true satisfaction alpha: a tech/quality-tilted
  survivor basket also loads on SMB/HML/UMD/quality that a 4-factor model would strip out (and
  Edmans' own alpha is a 4-factor number). Building a free, point-in-time 4-factor attribution
  offline is out of scope; naming the gap is the honest move.
- **Survivorship, named on the Signal axis.** The basket is hand-picked from *today's* known,
  still-listed winners — a survivor selection that biases the raw race toward the basket. We
  size it directly with a **random survivor-basket placebo**: 5,000 random equal-weight baskets
  drawn from a `CONTROL` pool of large-cap survivors *not* chosen for workplace prestige. If a
  random survivor basket earns alpha just as easily, the effect is survivorship, not
  satisfaction. Delisted / privatised perennials (Whole Foods, Ultimate Software, Nordstrom) are
  excluded and listed in `data.py` `DELISTED`.
- **Persistence over snooping.** The first/second-half split is a pre-declared, non-snooped cut
  (halve the sample); the win-rate carries a **Wilson (1927)** interval; the synthetic control
  checks the null over **20 seeds** so no single stream decides the false-positive rate.
- **One execution lag, costs one-way × NAV, long-only.** Weights are set at the month-end close
  and earn the next month (calendar rebalance, zero look-ahead); the sample ends at a **complete**
  month (2026-06-30); every leg is total-return.

## Data sources

- **Daily total-return (auto-adjusted) closes** — yfinance (no key), cached under `_cache/` as
  one parquet per ticker, resampled to month-end total returns, 2016-01 → 2026-06.
- **The basket and control lists** are hand-coded and cited in
  [`data.py`](../employee_satisfaction/data.py) (each name carries its perennial-membership
  reason). Fortune publishes the "100 Best Companies to Work For" annually with Great Place to
  Work; membership here is curated from that public list and its sibling "World's Best
  Workplaces."
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [392-glassdoor-sentiment](../../392-glassdoor-sentiment/) — the **crowd-rating** construct
  (employee reviews on Glassdoor as a live sentiment signal). Different data (anonymous
  crowd-sourced star ratings, not a curated editorial list) and a different question (does the
  *rating* predict returns?). This study tests the **specific Fortune "Best Companies to Work
  For" list** Edmans used, as a static prestige basket.
- [526-intangible-value](../../526-intangible-value/) — intangibles broadly (R&D / brand /
  organisational capital as a value adjustment). Employee satisfaction is *one* intangible; this
  study isolates the "best employer" list rather than a balance-sheet-wide intangibles measure.
- [751-fortune-500-inclusion](../../751-fortune-500-inclusion/) — a Fortune ranking too, but a
  **size** list (revenue-ranked 500) tested as an **event study** (does joining/leaving move the
  stock?). This is the **best-employer** list tested as a **held basket** (does owning it earn
  alpha?) — a different Fortune list, a different construct, a different research design.

None of the siblings test the specific claim here — **owning the Fortune "100 Best Companies to
Work For" earns risk-adjusted alpha** — which is this study's own axis.
