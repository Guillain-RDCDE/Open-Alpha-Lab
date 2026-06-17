# References & literature map — Study 274 (Fine-Wine)

## The claim under test

> *Fine wine is a low-correlation alternative asset: add a sleeve to an equity
> portfolio and you raise risk-adjusted returns because wine "marches to its own
> drum."* — the standard pitch from wine-investment merchants and the Liv-ex
> marketing literature.

The question this study answers: **does the Liv-ex 100 diversify, or does it
just lag the S&P 500 while *looking* uncorrelated because its prices are
smoothed?**

## The wine-as-an-asset literature

- **Masset, P. & Henderson, C. (2010).** "Wine as an Alternative Asset Class."
  *Journal of Wine Economics*, 5(1), 87–118. The canonical academic treatment.
  Finds wine has had attractive returns and *low measured* correlation with
  equities, but flags illiquidity, heterogeneity and the difficulty of
  constructing an investable index. The "low correlation" is the headline that
  this study stress-tests.

- **Dimson, E., Rousseau, P. L. & Spaenjers, C. (2015).** "The Price of Wine."
  *Journal of Financial Economics*, 118(2), 431–449. A 1900–2012 history of
  investment-grade Bordeaux. Real financial returns ~4.1%/yr after storage and
  insurance — *below* equities — with substantial idiosyncratic risk. Directly
  supports the "wine lags" finding here.

- **Masset, P. & Weisskopf, J.-P. (2010/2018).** Studies of fine-wine returns
  through crises: wine held up in some downturns (a diversification point) but
  is highly exposed to the same global-wealth shocks as risk assets, and its
  apparent crisis-resilience partly reflects *stale* mid-prices.

- **Sanning, L. W., Shaffer, S. & Sharratt, J. M. (2008).** "Bordeaux Wine as a
  Financial Investment." *Journal of Wine Economics*, 3(1), 51–71. Documents a
  small CAPM/Fama-French exposure for wine — i.e., a nonzero, low equity beta,
  consistent with the de-smoothed +0.28 found here.

## Why the *measured* correlation is too low — the smoothing problem

- **Geltner, D. (1991).** "Smoothing in Appraisal-Based Returns." *Journal of
  Real Estate Finance and Economics*, 4(3), 327–345. **(1993).** "Estimating
  Market Values from Appraised Values without Assuming an Efficient Market."
  *Journal of Real Estate Research*, 8(3), 325–345. The foundational work on
  appraisal/mark-to-model smoothing: reported returns on illiquid assets follow
  an AR(1), which *mechanically* depresses measured volatility and correlation.
  The first-order unsmoothing `r_true_t = (r_obs_t − ρ·r_obs_{t−1})/(1−ρ)` is
  the standard correction we apply to the Liv-ex 100.

- **Okunev, J. & White, R. (2003).** Unsmoothing of hedge-fund and alternative
  returns — the same AR(1) correction generalised. Liv-ex mid-prices are a
  textbook case: each component is the average of the live bid and offer, never
  a transaction print, so the index is smoothed by construction.

## Why this is a small-sample / tradability question, not just a stats question

- **Storage, insurance, provenance and spread.** Round-trip costs on physical
  fine wine via merchants or auction are an order of magnitude larger than for
  listed equities (~10–15% spread plus ~1%/yr carry). The Liv-ex 100 is a
  *mid-price* benchmark and is **not itself investable** — funds that tried to
  track it (e.g., several London wine funds in the 2010s) largely closed.

- **Survivorship & basket curation.** The Liv-ex 100 is rebalanced to the most
  actively traded wines; defunct or illiquid wines drop out. Like any curated,
  rebalanced index, this flatters measured returns and understates risk — named
  on the Signal axis.

- **Currency.** The Liv-ex 100 is quoted in GBP; a USD-based investor inherits
  GBP/USD swings (visible in the 2016 Brexit-driven jump). The diversification
  measured here mixes a wine factor with an FX factor.

## Method lineage

- **Geltner first-order unsmoothing** — `strategy.desmooth`, with ρ estimated as
  the first-order autocorrelation of reported annual returns.
- **Newey-West HAC t-stat** — `strategy.hac_tstat`, applied to the per-year
  blend-minus-equity excess return (the quantity whose positive mean *is* a
  diversification benefit), robust to the residual autocorrelation that smoothing
  leaves behind.
- **Two-asset mean-variance frontier** — Sharpe across the long-only wine/equity
  weight grid, gross/net and raw/de-smoothed.

## Data sources

- **Liv-ex 100 Fine Wine Index.** London International Vintners Exchange,
  https://www.liv-ex.com/ — monthly index, base 100 = July 2001. Year-end levels
  hardcoded in `data.py`. Mid-price construction (bid/offer average).
- **S&P 500 (^GSPC).** Repo-level `_cache/^GSPC_split_only.parquet` (daily
  OHLCV); December/December closes give calendar-year price returns. Price-only,
  no dividends — labelled as such throughout.

## Related desk studies

- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: the
  "uncorrelated sleeves raise Sharpe" claim, tested honestly on real assets.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: 60/40 diversification
  — the same mean-variance machinery on liquid assets.
- **[Study 209 — Eth-Btc-Ratio](../../209-eth-btc-ratio/)** and the gold/cross-
  asset studies: other "alternative" diversifiers stress-tested for real,
  net-of-cost benefit.
