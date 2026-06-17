# References & literature map — Study 250 (Reverse-Split)

## The claim under test

The popular narrative is that reverse splits are the "kiss of death" — a desperate
corporate action performed by companies teetering on the edge of delisting, which
dooms the stock to further decline.  We test whether this claim survives statistical
scrutiny: is post-reverse-split underperformance statistically real, and is it
separable from the underlying distress?

## Foundational event studies on reverse splits

- **Desai & Jain (1997).** *Long-Run Common Stock Returns following Stock Splits and
  Reverse Splits.* Journal of Business 70(3), 409–433.  The most-cited academic
  treatment: reverse splits are associated with significant long-run *underperformance*
  (−10% to −30% over 1–3 years post-effective-date).  Importantly, they use the
  announcement date on a 1970s–1990s universe — which we cannot replicate with
  yfinance.  Their result is consistent with ours but uses a different window and
  a much larger sample (n ≈ 650 events from CRSP).

- **Han (1995).** *Insider Ownership and Firm Value: Evidence from Real Estate
  Investment Trusts.* Also studied reverse splits briefly; corroborates Desai & Jain.

- **Peterson & Peterson (1992).** *A Further Understanding of Stock Distributions:
  The Case of Reverse Stock Splits.* Journal of Financial Research 15(2), 189–205.
  Found negative abnormal returns in the months following reverse splits, consistent
  with signal of distress rather than a clean corporate action effect.

## Why reverse splits correlate with distress (and why the confound matters)

- **Listing standards.** NYSE and NASDAQ minimum bid-price requirements ($1 for NASDAQ)
  force companies to perform reverse splits or face delisting.  Companies in this
  situation are almost by definition in financial or operational distress.  The
  negative post-RS returns are therefore partly — or entirely — the continuation of
  the distress trend, not a causal effect of the RS.

- **Signalling model.** Ritter (1991) and others established that corporate actions
  signal management's private information.  A reverse split signals that the company
  *cannot* grow its way out of a low price — a negative signal, unlike a forward split
  (which signals confidence).  But separating the signal from the pre-existing trend
  requires controlling for distress (leverage, Z-score, earnings).

- **Altman Z-score.** Altman (1968), *Financial Ratios, Discriminant Analysis and the
  Prediction of Corporate Bankruptcy* (Journal of Finance) — the standard tool for
  measuring financial distress.  A proper reverse-split study would compare RS names
  vs matched-distress non-RS names using Z-score or similar.  This study does not
  have access to that data.

- **Survival bias in reverse.** Unlike forward-split studies (which over-sample past
  winners), reverse-split studies face *negative* survivorship: the worst outcomes
  (complete bankruptcy, delisting) are excluded from samples limited to listed
  companies.  Our sample is biased toward names that survived — almost certainly
  understating the typical post-RS loss.

## The "big company" exceptions

- **Citigroup (C, 2011).** 1-for-10 reverse split to clean up the post-TARP balance
  sheet.  The bank had already stabilised; the RS was an administrative tidying, not
  distress signalling.  C significantly outperformed after the event.

- **AIG (2009).** 1-for-20 RS after the government bailout — again, the distress was
  already in the past; the RS marked a recovery phase.

- **GE (2021).** 1-for-8 as part of a large strategic restructuring.  Not a distress
  signal.

These three cases (out of 17) have positive post-RS returns and illustrate that the
narrative "reverse split = kiss of death" fails for large companies undergoing
deliberate restructuring rather than forced by listing threats.

## Out-of-sample evidence

- **Kim, Klein & Rosenfeld (2008).** *Return Predictability Following Reverse Stock
  Splits.* Financial Management 37(3), 571–588.  Find continuing underperformance
  post-RS but note the confound: RS is proxying for prior poor performance and
  ongoing distress, and adjusting for distress reduces the apparent RS effect.

- **Byun & Rozeff (2003).** *Long-Run Performance after Stock Splits: 1927 to 1996.*
  Journal of Finance 58(3), 1063–1086.  While focused on forward splits, discusses
  methodological issues relevant to event studies in corporate actions.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica).
  Implemented in [`strategy._hac_tstat`](../reverse_split/strategy.py).

- **Event-study inference.** Fama, Fisher, Jensen & Roll (1969), *The Adjustment of
  Stock Prices to New Information* (International Economic Review) — the foundational
  event-study framework.

- **Barber & Lyon (1997).** *Detecting Long-Run Abnormal Stock Returns: The Empirical
  Power and Specification of Test Statistics.* Journal of Finance 52(3), 581–618.
  Discusses bias in long-run event studies, including the importance of matched controls
  and the distortion introduced by skewness and cross-sectional correlation.  Directly
  relevant to our 17-event sample limitation.

## Related desk studies

- **[Study 142 — Split-Drift](../../142-split-drift/)**: the mirror image — post-forward-
  split drift (Ikenberry et al. 1996).  Also None/Mirage on effective dates.  Reading
  both together illustrates how splits (bullish signal) and reverse splits (distress
  signal) are treated asymmetrically by the market.

- **[Study 229 — Beneish M-Score](../../229-beneish-m-score/)** and
  **[Study 230 — Ohlson O-Score](../../230-ohlson-o-score/)**: other distress-predicting
  signals with the same confound problem — are they detecting "distress" or something
  additional?

- **[Study 231 — Sloan Accruals](../../231-sloan-accruals/)**: quality-based short signal
  with similar separability challenges between the factor and the underlying fundamental.

## Data sources

- **Yahoo! Finance daily closes** (via `yfinance`) — split-adjusted closes for the 17
  reverse-split tickers, starting 2008-01-01.  Effective dates sourced from public
  corporate announcements and SEC filings.  Only tickers still accessible via yfinance
  are included (excludes fully delisted names).
