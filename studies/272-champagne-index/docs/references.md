# References & literature map — Study 272 (Champagne-Index)

## The claim under test

The **Champagne Indicator** is market folklore, not an academic finding: when
champagne shipments boom, collective euphoria has peaked and equities are near a
top; when the corks stop popping, gloom has bottomed and a rally is near. It is a
*contrarian sentiment* signal — buy when no one is celebrating, sell when everyone
is. We test the tradable, look-ahead-free version: does this year's worldwide
champagne-shipment growth predict *next* year's S&P 500 return (negatively)?

## Why this belongs to the "exuberance marks the top" family

The champagne story is one of a cluster of luxury / sentiment "indicators" that
share the same behavioural premise and the same small-sample fragility:

- **The Hemline Index** (Hansen, attributed to George Taylor, 1920s): skirts rise
  with markets. The canonical luxury-as-sentiment folklore.
- **The Skyscraper Curse** (Lawrence, 1999, "The Skyscraper Index"): record-breaking
  tall buildings are completed near business-cycle peaks. Same euphoria-marks-the-top
  logic; see [Study 160 — Skyscraper-Curse](../../160-skyscraper-curse/).
- **The Magazine-Cover Indicator** (Arnold, Earl & North, 2007, *Financial Analysts
  Journal*, "Are Cover Stories Effective Contrarian Indicators?"): extreme cover
  stories tend to coincide with the end of a trend.
- **The Lipstick Index** (Leonard Lauder, ~2001): small-luxury spending *rises* in
  downturns — the inverse-sentiment cousin.

All four are vivid, low-frequency, and powered by a handful of dramatic episodes —
exactly the structure that produces seductive but statistically thin results.

## Sentiment and the equity premium — the serious literature

- **Baker, M. & Wurgler, J. (2006).** "Investor Sentiment and the Cross-Section of
  Stock Returns." *Journal of Finance*, 61(4), 1645–1680. The rigorous version of
  the champagne idea: a composite sentiment index predicts lower forward returns,
  especially for hard-to-value stocks. Champagne shipments are a (much noisier)
  single-proxy stand-in for this.
- **Shiller, R. J. (2000).** *Irrational Exuberance*. Princeton University Press.
  The behavioural backbone: euphoria and overpricing precede mean-reversion.
- **Tetlock, P. C. (2007).** "Giving Content to Investor Sentiment: The Role of
  Media in the Stock Market." *Journal of Finance*, 62(3), 1139–1168. Media-tone
  sentiment as a contrarian short-horizon predictor.

## Why a vivid pattern can still be a mirage

- **Data snooping.** Sullivan, Timmermann & White (1999), "Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap," *Journal of Finance*, 54(5),
  1647–1691: searching many cute indicators inflates apparent significance; the
  honest benchmark accounts for the universe of stories implicitly tested.
- **The t ≥ 3 hurdle.** Harvey, Liu & Zhu (2016), "… and the Cross-Section of
  Expected Returns," *Review of Financial Studies*, 29(1), 5–68: with hundreds of
  candidate predictors, the appropriate t-stat hurdle is ~3, not 2. A folklore
  indicator at |t| = 1.3 is nowhere close.
- **Tiny-n power.** With ~25 annual observations and ~17% equity vol, the minimum
  detectable correlation at 80% power is ≈ 0.5; the observed −0.33 is below it.

## Method lineage

- **Predictive OLS.** Regress next-year return on standardized champagne growth;
  the folklore predicts a negative slope.
- **Newey-West (HAC) t-stat.** `_nw_long_run_var` in `strategy.py` implements the
  Bartlett-kernel long-run variance of the regression score, with the standard
  rule-of-thumb lag `floor(4·(n/100)^(2/9))`. Annual macro predictors are mildly
  autocorrelated; the HAC correction strips spurious precision out of the slope SE.
- **Permutation test.** Shuffle champagne-growth labels 10,000 times; the p-value
  is the fraction of shuffles with a slope (or tercile spread) at least as extreme.
- **Leave-one-out.** Re-estimate the correlation dropping each year to check that
  the tilt is not a single-episode artifact.
- **Power calculation.** Fisher-z minimum detectable correlation at 80% power.

## Data sources

- **Worldwide champagne shipments.** Comité Interprofessionnel du Vin de Champagne
  (CIVC / "Comité Champagne") annual shipment statistics, reported each January for
  the prior year, supplemented by trade-press reporting (The Drinks Business,
  Reuters). Hardcoded as rounded millions-of-bottles in `data.py`. The *shape* —
  the 2007 pre-GFC peak, the 2009 crash, the 2020 COVID low, the 2021 record
  rebound — is what the indicator trades on.
- **S&P 500.** ^GSPC daily **price** index, staged at
  `_cache/^GSPC_split_only.parquet`. We compute December/December calendar-year
  price returns (dividends excluded — total return understated).

## Related desk studies

- **[Study 160 — Skyscraper-Curse](../../160-skyscraper-curse/)**: tall buildings
  as a top signal — the same euphoria-marks-the-peak structure.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the folklore-teardown
  template this study mirrors (hardcoded event table + real market returns).
- **[Study 174 — Magazine-Cover](../../174-magazine-cover/)**: cover-story
  contrarian indicator — sentiment-as-signal in the same family.
