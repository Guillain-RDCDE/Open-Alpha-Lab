# References & literature map — Study 332 (Downside-Beta)

## The claim under test

- **Ang, Chen & Xing (2006), *Downside Risk* (Review of Financial Studies, 19(4),
  1191–1239).** The canonical statement: cross-sectionally, stocks with higher *downside
  beta* — beta estimated on days when the market return is below its mean — earn higher
  average returns, and this premium is *not* subsumed by ordinary (symmetric) market
  beta, size, book-to-market, momentum or liquidity. Their headline: the realised return
  spread between the highest and lowest downside-beta quintiles is about **6% per annum**.
  The economic story is loss aversion / disappointment aversion: investors especially
  dislike assets that fall *with* the market in bad times, and demand to be paid for
  holding them. This is a testable cross-sectional pricing claim, which we run directly
  and — crucially — control for plain beta.

- **The control that decides it — relative downside beta.** Ang-Chen-Xing's own key
  variable is the *relative* downside beta, β⁻ − β: the part of downside sensitivity not
  already in ordinary beta. If a raw β⁻ sort pays only because high-β⁻ stocks are also
  high-β stocks, the "downside" story adds nothing. We sort on β⁻, on β, and on β⁻ − β
  side by side; the third is the honest test of the *incremental* downside premium.

## The downside-risk family

- **Bawa & Lindenberg (1977), *Capital Market Equilibrium in a Mean-Lower Partial Moment
  Framework* (Journal of Financial Economics).** The original lower-partial-moment CAPM —
  the theoretical root of downside beta, where systematic risk is measured only on the
  downside of a target return.
- **Roy (1952), *Safety First and the Holding of Assets* (Econometrica).** Safety-first
  portfolio choice — the intellectual ancestor of measuring risk as downside, not
  variance.
- **Hogan & Warren (1974), *Toward the Development of an Equilibrium Capital-Market Model
  Based on Semivariance*.** Semivariance pricing — the variance analogue of downside beta.
- **Estrada (2002), *Systematic Risk in Emerging Markets: The D-CAPM*.** Downside-beta
  CAPM applied cross-country; popularised "D-CAPM."

## The confound — is it just beta, idiosyncratic vol, or coskewness?

- **Harvey & Siddique (2000), *Conditional Skewness in Asset Pricing Tests* (Journal of
  Finance).** Coskewness (assets that crash with the market) is priced — a close cousin
  of downside beta, and a candidate for what β⁻ actually proxies.
- **Ang, Hodrick, Xing & Zhang (2006), *The Cross-Section of Volatility and Expected
  Returns* (Journal of Finance).** Idiosyncratic-vol effect — a confound that loads on the
  same high-beta names; the famous "low-vol anomaly" runs in the *opposite* direction.
- **Frazzini & Pedersen (2014), *Betting Against Beta* (Journal of Financial Economics).**
  The leverage-constraints story under which *low*-beta is the one that pays — directly at
  odds with a positive downside-beta premium, and tested on this desk (Study 238).
- **Bali, Cakici & Whitelaw (2011), *Maxing Out* (Journal of Financial Economics).** The
  MAX effect — lottery demand — another tilt that contaminates beta-sorted portfolios.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../downside_beta/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*
  (JASA); the overlapping-block CI respects the autocorrelation that i.i.d. resampling
  destroys — [`strategy.block_bootstrap_ci`](../downside_beta/strategy.py).
- **Cross-sectional quintile sort with a one-month execution lag** mirrors the desk's
  factor-zoo studies (long-term-reversal, one-month-reversal); the random-partition
  control is the same concentration null used throughout.

## Data sources used here

- **Yahoo! Finance daily adjusted closes** (via `yfinance`) for a large-cap S&P 500
  basket, 2005–2026, fetched through the opt-in survivorship guard
  `quantlab.universe.sp500_symbols(allow_survivorship_bias=True)`. The market proxy is the
  equal-weight mean of the same panel, so the cross-section and benchmark share one
  adjustment convention. All headline numbers carry an as-of date and a content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and the
  test-suite run on the deterministic [`data.synthetic_panel`](../downside_beta/data.py),
  never the network.

## Survivorship caveat (named on the Signal axis)

The real universe is the *current* membership projected backwards: every name survived to
today. A downside-risk premium estimated on survivors is an upper bound — the firms that
took the worst down-market hits and never recovered are simply missing from the panel, so
the realised high-β⁻ leg looks healthier than it was live. The caveat travels with the
stamp.

## Related desk studies

- **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta/)**: the Frazzini-
  Pedersen low-beta story — the *mirror image* of this claim (low-beta pays vs high-
  downside-beta pays). Same beta machinery, opposite tilt.
- **[Study 43 — Free-Lunch](../../43-free-lunch/)**: betting against beta as the "low-risk
  free lunch" — the leverage-bill teardown.
- **[Study 18 — Dull-Roar](../../18-dull-roar/)**: the low-volatility anomaly — the
  idiosyncratic-vol confound that loads on the same names.
- **[Study 208 — Gold-Miners](../../208-gold-miners/)**: upside-vs-downside beta as a
  *single-asset time-series* asymmetry test — a different question (one stock's beta
  shape) from this cross-sectional pricing test.
