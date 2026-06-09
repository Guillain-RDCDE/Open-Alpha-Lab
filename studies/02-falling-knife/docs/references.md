# References — Study 02 (Falling-Knife)

The claim under test, the literature on short-term reversal / overreaction it walks into, and
the statistical methods used to decide whether the bounce is real or noise.

## The claim (the folk belief we steelman and test)
- **"Buy the dip."** The widely-repeated retail maxim that a sharp one-day drop in a major
  index (−3% / −5% / −7%) is a buying opportunity — prices "snap back." No single author owns
  it; it is tested here against the strongest version (a mechanical, pre-registered buy-after-N%
  rule on the Nasdaq-100 / S&P 500) rather than a strawman.

## Short-term reversal & overreaction (is the bounce real?)
- **De Bondt, W. F. M. & Thaler, R. (1985). "Does the stock market overreact?"** *Journal of
  Finance* 40(3), 793–805. The foundational overreaction hypothesis — extreme moves partly
  reverse.
- **Lehmann, B. (1990). "Fads, martingales, and market efficiency."** *Quarterly Journal of
  Economics* 105(1), 1–28. Short-horizon return reversals at the weekly frequency.
- **Jegadeesh, N. (1990). "Evidence of predictable behavior of security returns."** *Journal of
  Finance* 45(3), 881–898. Predictable short-term (one-month) reversal in individual stocks.
- **Lo, A. W. & MacKinlay, A. C. (1990). "When are contrarian profits due to stock market
  overreaction?"** *Review of Financial Studies* 3(2), 175–205. How much "contrarian" profit is
  overreaction vs cross-autocorrelation — a caution against reading a bounce as a free edge.
- **Atilgan, Y., Bali, T. G., Demirtas, K. O. & Gunaydin, A. D. (2020). "Left-tail momentum."**
  *Journal of Financial Economics* 135(3). Extreme downside moves and their (non-)reversal — the
  index-level analogue of the buy-the-dip question.

## Why a few crashes can fake an edge (clustering, capacity, selection)
- **Mandelbrot, B. (1963). "The variation of certain speculative prices."** *Journal of
  Business* 36(4), 394–419. Volatility clustering — extreme down days arrive in bursts, so
  "events" are not independent draws (the bootstrap must respect this).
- **Sullivan, R., Timmermann, A. & White, H. (2001). "Dangers of data mining: the case of
  calendar effects in stock returns."** *Journal of Econometrics* 105(1). Why a rule that looks
  significant in-sample can be a selection artefact — the spirit of the out-of-sample flip test.

## Method (shared desk engine)
- **Newey, W. & West, K. (1987).** Autocorrelation-robust (HAC) standard errors. *Econometrica*
  55, 703–708.
- **Lo, A. (2002). "The statistics of Sharpe ratios."** *Financial Analysts Journal* 58(4).
- **Politis, D. & Romano, J. (1994). "The stationary bootstrap."** *JASA* 89(428). Block
  bootstrap for serially-dependent / clustered returns — the clustering-aware CI used here.
- Reproducibility stamp (as-of + content fingerprint): [`quantlab/repro.py`](../../../quantlab/repro.py);
  inference helpers: [`quantlab/analytics.py`](../../../quantlab/analytics.py),
  [`quantlab/stats.py`](../../../quantlab/stats.py).
