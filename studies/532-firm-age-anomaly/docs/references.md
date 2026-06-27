# References & literature map -- Study 532 (Firm-Age-Anomaly)

## The primary claim under test

- **Fama, E. F. & French, K. R. (2004).** "New Lists: Fundamentals and Survival Rates."
  *Journal of Financial Economics*, 73(2), 229--269. The founding documentation of the
  "new-list" effect: firms that newly list have lower profitability, faster asset growth, and
  notably *lower survival rates* and *lower expected returns* than seasoned firms. Young,
  recently-listed firms drag on the cross-section. This is the prediction we test: mature firms
  should out-earn young firms (old-minus-young > 0).
- **Jiang, G., Lee, C. M. C., & Zhang, Y. (2005).** "Information Uncertainty and Expected
  Returns." *Review of Accounting Studies*, 10(2--3), 185--221. Firm age is one of their
  headline information-uncertainty proxies: younger (higher-uncertainty) firms earn lower
  future returns, especially among low-momentum names. The mechanism the brief invokes for the
  firm-age sort.

## Why the effect should exist -- the theoretical backbone

- **Pastor, L. & Veronesi, P. (2003).** "Stock Valuation and Learning about Profitability."
  *Journal of Finance*, 58(5), 1749--1789. Young firms carry high uncertainty about average
  profitability; as investors learn, uncertainty (and the associated valuation premium) resolves.
  A learning-based rationale for an age-related return pattern.
- **Zhang, X. F. (2006).** "Information Uncertainty and Stock Returns." *Journal of Finance*,
  61(1), 105--137. Confirms that high information-uncertainty stocks (young, volatile, hard to
  value) earn lower returns, reinforcing the Jiang-Lee-Zhang firm-age channel.

## IPO underperformance -- the closely related literature

- **Ritter, J. R. (1991).** "The Long-Run Performance of Initial Public Offerings." *Journal of
  Finance*, 46(1), 3--27. The classic IPO long-run underperformance result: newly public firms
  underperform seasoned peers for ~3 years post-listing. Firm-age and new-list effects are the
  cross-sectional cousins of this finding.
- **Loughran, T. & Ritter, J. R. (1995).** "The New Issues Puzzle." *Journal of Finance*, 50(1),
  23--51. Documents persistent underperformance of issuing firms; central to the prior that
  young/recently-issued names are poor performers.

## Survivorship bias -- why our sign may flip

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings are correlated with poor performance; removing them biases factor returns
  upward. Crucially for *this* study: young, recently-listed firms have the *highest* delisting
  rates (Fama-French 2004), so a survivor-only young leg is the most upward-biased -- it omits
  precisely the names that make the new-list effect work, and can reverse the measured sign.
- **Brown, S. J., Goetzmann, W. N., Ibbotson, R. G., & Ross, S. A. (1992).** "Survivorship Bias
  in Performance Studies." *Review of Financial Studies*, 5(4), 553--580. The canonical statement
  of how conditioning on survival manufactures spurious performance -- here it manufactures a
  spurious *young-wins* result.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of Stock
  Returns." *Review of Financial Studies*, 31(7), 2606--2649. Many anomalies are weaker or absent
  out-of-sample once data-snooping and selection are controlled.

## Post-publication decay

- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. ~32% average attenuation of anomalies after
  publication; the new-list / firm-age effect, documented in the mid-2000s, is a candidate.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  long-run-variance estimator behind the one-sample t in
  [`strategy.summary`](../firm_age_anomaly/strategy.py).

## Related desk studies

- **[Study 219 -- IPO-Pop](../../219-ipo-pop/)**: first-day IPO returns -- the short-horizon
  sibling of the new-list effect.
- **[Study 265 -- IPO-Volume](../../265-ipo-volume/)**: IPO issuance as a market-timing signal.
- **[Study 172 -- Hundred-Minus-Age](../../172-hundred-minus-age/)**: a different "age" -- the
  investor's, not the firm's -- but the same desk infrastructure.
- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: another cross-sectional
  sort on the same survivor-basket machinery (rolling rank, equal-weight legs, HAC inference).
