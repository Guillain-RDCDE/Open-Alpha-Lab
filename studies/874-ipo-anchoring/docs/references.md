# References & literature map — Study 874 (IPO-Price Anchoring)

## The claim under test

- **Anchoring & adjustment.** Amos **Tversky & Daniel Kahneman (1974)**, *"Judgment under
  Uncertainty: Heuristics and Biases"* (Science 185): people estimate from an initial
  reference value and adjust insufficiently. The IPO **offer price** — the single round number
  printed across the prospectus and every first-day headline — is a natural candidate anchor for
  a newly public stock that has no trading history to price against.
- **Anchoring on salient prices in markets.** **George, T. & Hwang, C.-Y. (2004)**, *"The
  52-Week High and Momentum Investing"* (Journal of Finance 59): a salient reference price (the
  52-week high) predicts returns — direct evidence that traders anchor on round, memorable
  price levels. **Baker, M., Pan, X. & Wurgler, J. (2012)**, *"The Effect of Reference Point
  Prices on Mergers and Acquisitions"* (Journal of Financial Economics 106): peak/anchor prices
  shape offer prices and deal outcomes. This study asks the IPO analogue: is the **offer price**
  such an anchor for the aftermarket?
- **Loss aversion / the disposition effect.** **Shefrin, H. & Statman, M. (1985)** and
  **Odean, T. (1998, JF)**: investors hold losers and sell winners relative to a purchase-price
  reference. For an IPO, the offer price is the cohort's collective cost basis; crossing *below*
  it puts "everyone who bought the deal" under water — the folklore that a **below-offer** stock
  carries a persistent drag until it reclaims the line. That is Test 2 here.

## What we measure, and the honesty rails

- **The anchor.** For each of 44 curated recent US listings, the **offer price** (traditional
  underwritten IPO, from the 424B4) or the exchange **reference price** (direct listing — a
  softer anchor, tested separately). The behavioural distance is `gap = log(price / offer)`.
- **Market-adjusted forward returns.** Every name-month's forward return is the name's
  next-month total return **minus SPY's**. This is deliberate: it nets out the generic "IPOs
  fell in 2022" tape so the test isolates whatever is specific to the *offer-price anchor*, not
  IPO-ness in general (that is study 623's question, not this one).
- **Two honest tests.** (1) A monthly **Fama-MacBeth** cross-sectional slope of forward
  abnormal return on the gap (anchoring ⇒ negative). (2) A **below-offer − above-offer** basket
  spread (drag ⇒ negative). Both first collapse each calendar month to a single number, *then*
  run a time-series HAC *t* over months — because these names are one heavily time-correlated
  cohort, and treating 2,698 name-months as independent would badly overstate significance.
- **Newey-West (HAC, Bartlett, 6-lag)** *t* on both monthly series; a **1,000-permutation
  placebo** breaking the gap → forward-return pairing within each month; a **two-era** cut; a
  **20-seed synthetic positive control** that plants (and recovers) the anchoring pull.
- **Low N, stated up front.** ~45 names dominated by the 2020-21 cohort ⇒ low power ⇒ the honest
  prior is None. Curation/selection bias is named on the **Signal** axis. One documented
  execution lag (gap at close of `t` → hold `t+1`), no partial months (as-of 2026-06-30).

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent (HAC)
  covariance; the *t* used on both monthly series.
- **Fama, E. & MacBeth, J. (1973)** — the month-by-month cross-sectional-slope estimator whose
  time-series average and *t* form Test 1.
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **Curated anchor table** (`ipo_anchor/data.py`, `IPOS`): 44 well-known recent US listings —
  ticker, public **offer / reference price**, **first-trade date** — from SEC EDGAR 424B4
  prospectuses and listing-exchange reference-price notices, widely reported in the financial
  press on each listing date. Encoded directly as public dated facts, per desk convention.
- **yfinance daily adjusted closes** (`auto_adjust=True`, total-return) for those tickers plus
  `SPY`, 2014-01 → 2026-06-30, cached under `_cache/ipo_anchor_prices.csv`.
- All headline numbers are pinned in [`results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [219-ipo-pop](../../219-ipo-pop/) — the **first-day pop** (offer → first close), a
  one-day underpricing phenomenon. This study ignores day one and tests the **aftermarket's**
  relation to the offer *anchor* over months.
- [265-ipo-volume](../../265-ipo-volume/) — the **IPO-volume / issuance** market-timing signal
  (hot-issue windows predicting aggregate returns), a macro calendar signal, not a per-name
  anchor.
- [623-ipo-long-run-underperformance](../../623-ipo-long-run-underperformance/) — whether IPOs
  as a class **underperform** over 3-5 years (Ritter). We explicitly **market-adjust that away**
  and ask the *different* question of whether the **offer price** acts as a behavioural anchor
  (gap-predicts-drift; below-offer drag) on top of the cohort's overall path.
- [783-ipo-deal-of-year](../../783-ipo-deal-of-year/) — the "deal of the year" / headline-IPO
  folklore, a single-name attention story, not a systematic offer-price-anchor cross-section.

None of the siblings tests whether the **distance of the current price from the IPO offer
price** predicts drift, nor whether **trading below the offer** is a persistent drag — this
study's own axis.
