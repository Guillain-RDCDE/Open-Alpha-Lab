# References & literature map -- Study 509 (Intermediate-Momentum)

## The primary claim under test

- **Novy-Marx, R. (2012).** "Is momentum really momentum?" *Journal of Financial Economics*,
  103(3), 429--453. The paper this study replicates. Novy-Marx decomposes the standard
  12-2 momentum formation window and shows the cross-sectional momentum premium is driven by
  the **intermediate** horizon -- returns from roughly twelve to seven months ago (t-12..t-7)
  -- rather than the **recent** horizon (t-6..t-2). Sorting on intermediate past performance
  earns a large, significant premium; sorting on recent past performance earns little. He
  argues this is hard to reconcile with behavioural underreaction stories, which predict the
  drift should be strongest in the most recent returns.

## The momentum effect it builds on

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65--91. The founding
  cross-sectional momentum paper -- buy past 3--12 month winners, sell losers, hold 3--12
  months. The "12-2" convention (skip the most recent month) that Novy-Marx dissects.
- **Jegadeesh, N. (1990).** "Evidence of Predictable Behavior of Security Returns." *Journal
  of Finance*, 45(3), 881--898. Documents **short-term reversal** in the most recent month --
  the reason momentum studies skip month t-1 (and why the "recent" window can be contaminated).
- **Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013).** "Value and Momentum
  Everywhere." *Journal of Finance*, 68(3), 929--985. Momentum as a pervasive cross-asset
  premium; the standard formation/holding conventions and risk-factor framing.

## Why a horizon decomposition matters -- the behavioural debate

- **Barberis, N., Shleifer, A., & Vishny, R. (1998).** "A Model of Investor Sentiment."
  *Journal of Financial Economics*, 49(3), 307--343. Underreaction model that predicts drift
  should be strongest in recent information -- the prediction Novy-Marx's intermediate result
  challenges.
- **Daniel, K., Hirshleifer, D., & Subrahmanyam, A. (1998).** "Investor Psychology and
  Security Market Under- and Overreactions." *Journal of Finance*, 53(6), 1839--1885.
  Overconfidence/self-attribution model of momentum; a different behavioural lens.
- **Goyal, A. & Wahal, S. (2015).** "Is Momentum an Echo?" *Journal of Financial and
  Quantitative Analysis*, 50(6), 1237--1262. Tests the Novy-Marx intermediate ("echo")
  result internationally; finds the echo is largely a US phenomenon and weak or absent in
  many non-US markets -- a caution on generality.

## Subsequent evidence, attenuation, and costs

- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. ~58% post-publication decay on average;
  momentum-family signals published and widely traded are prime candidates for attenuation.
- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading Costs."
  *Review of Financial Studies*, 29(1), 104--147. Momentum has high turnover and meaningful
  trading costs; net premia are materially smaller than gross, especially for small caps.
- **Hou, K., Xue, C., & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019--2133. Many momentum-adjacent anomalies replicate weakly once
  micro-caps are screened out and costs are imposed.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance -- removing them biases factor returns.
  For momentum, past *losers* (the natural short leg) are exactly the names most likely to be
  delisted, so a survivor basket understates the short leg's losses and the gross premium.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703--708. The HAC long-run variance estimator in
  [`strategy.summary`](../intermediate_momentum/strategy.py).

## Related desk studies

- **[Study 24 -- Stampede](../../24-stampede/)**: plain 12-1 cross-sectional momentum on the
  modern S&P 500 -- the parent effect this study decomposes by horizon.
- **[Study 25 -- Clean-Slate](../../25-clean-slate/)**: residual momentum -- another attempt
  to find a cleaner version of the momentum drift.
- **[Study 237 -- Residual-Momentum](../../237-residual-momentum/)**: residual-return momentum
  and whether it dodges the crashes; same desk infrastructure (rolling sort, HAC inference).
- **[Study 50 -- High-Water](../../50-high-water/)**: the 52-week-high effect -- momentum
  wearing a different hat.
