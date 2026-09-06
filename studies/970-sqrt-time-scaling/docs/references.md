# Sources & literature map — Study 970 (Root Time)

## The rule and its assumption

- **Bachelier, L. (1900), *Théorie de la spéculation*.** The original: the "law of the square
  root of time" for a random walk, thirty-five years before Kolmogorov made it rigorous.
- **Basel Committee on Banking Supervision (1996, 2019), *Amendment to the Capital Accord to
  Incorporate Market Risks* and the FRTB.** The regulatory instruction to scale a one-day VaR
  by √10 — the single most consequential use of this rule anywhere.
- **Danielsson, J. & Zigrand, J.-P. (2006), "On Time-Scaling of Risk and the Square-Root-of-Time
  Rule", *Journal of Banking & Finance* 30(10), 2701-2713.** Shows the rule *underestimates*
  risk when jumps are present, independently of autocorrelation. The complementary failure mode
  to the one measured here.

## Variance ratios

- **Lo, A. W. & MacKinlay, A. C. (1988), "Stock Market Prices Do Not Follow Random Walks:
  Evidence from a Simple Specification Test", *Review of Financial Studies* 1(1), 41-66.** The
  estimator, the bias corrections and the heteroskedasticity-robust statistic used throughout.
- **Lo, A. W. & MacKinlay, A. C. (1989), "The Size and Power of the Variance Ratio Test in
  Finite Samples", *Journal of Econometrics* 40(2), 203-238.** Why the small-sample corrections
  are not optional.
- **Poterba, J. M. & Summers, L. H. (1988), "Mean Reversion in Stock Prices: Evidence and
  Implications", *Journal of Financial Economics* 22(1), 27-59.** The long-horizon mean
  reversion claim, and the sample-size problem that dogs it.
- **Charles, A. & Darné, O. (2009), "Variance-Ratio Tests of Random Walk: An Overview",
  *Journal of Economic Surveys* 23(3), 503-527.** The modern survey, including the multiple
  variance-ratio and wild-bootstrap refinements this study does not use.

## Sharpe ratios and time scaling

- **Lo, A. W. (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal* 58(4),
  36-52.** The annualisation factor used here (`quantlab.analytics.lo_annualization_factor`):
  with autocorrelated returns the correct multiplier is not √q, and for a smoothed return
  series it can be far from it.
- **Getmansky, M., Lo, A. W. & Makarov, I. (2004), "An Econometric Model of Serial Correlation
  and Illiquidity in Hedge Fund Returns", *Journal of Financial Economics* 74(3), 529-609.**
  The extreme case: smoothed (stale) pricing manufactures autocorrelation and inflates every
  √T-scaled statistic.

## Why the equity result looks so benign

- **Fama, E. F. (1970) and Fama, E. F. (1991), "Efficient Capital Markets: II", *Journal of
  Finance* 46(5), 1575-1617.** Index returns are close to serially uncorrelated, which is why
  the rule was never seriously questioned by the people who mostly test it on indices.
- **Campbell, J. Y., Lo, A. W. & MacKinlay, A. C. (1997), *The Econometrics of Financial
  Markets*, ch. 2.** The textbook treatment of all of the above.

## Neighbours on this desk

**815-variance-ratio-reversal**, **969-log-vs-simple-returns**, **966-har-vs-garch**,
**841-overlapping-returns**, **990-var-breach-count**, **992-vol-clustering-halflife**,
**917-nav-staleness-timezone**.
