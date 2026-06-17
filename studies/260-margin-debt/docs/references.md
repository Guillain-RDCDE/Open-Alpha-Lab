# References & literature map -- Study 260 (Margin-Debt)

## The data source -- what margin debt actually is

- **FINRA, "Margin Statistics."** Monthly debit balances in customers'
  securities margin accounts, reported by FINRA member firms (the successor, from
  2010, to the NYSE member-firm series). The canonical series behind every
  "margin debt hits record" headline. https://www.finra.org/investors/margin-statistics
- **NYSE Factbook / "Securities Market Credit."** The pre-2010 member-firm margin
  debt series. The NYSE and FINRA series are spliced into one continuous monthly
  history charted by the sources below.
- **Advisor Perspectives (Jill Mislinski / Jennifer Nash), "Margin Debt and the
  Market."** The widely-cited monthly chart pack overlaying real margin debt on the
  real S&P 500. Popularized the "margin debt leads the market" visual narrative
  that this study stress-tests.
- **Yardeni Research, "Stock Market Briefing: Margin Debt."** Ed Yardeni's
  long-run margin-debt charts; another standard reference for the series.

## The claim and its critiques

- **The folklore.** "Record / fast-rising margin debt means investors are
  dangerously leveraged, so a record is a contrarian SELL signal." Repeated every
  cycle in the financial press; visually compelling because margin debt peaked
  near the 2000, 2007, and 2021 market tops.
- **Irrational Exuberance (Shiller, 2000/2015).** Robert Shiller discusses margin
  borrowing as part of the feedback loop in speculative bubbles -- a *descriptive*
  link, not a tradable timing rule.
- **Mislinski / Nash, "Margin Debt: Is It a Useful Indicator?"** The Advisor
  Perspectives caveat: margin debt is *coincident* with the market, rising and
  falling almost mechanically with prices, so the "lead" is mostly an artifact of
  the level co-moving with the index. Our regression confirms the YoY change has
  the right contrarian sign but no significant forward predictive power.
- **The mechanical-coincidence critique (general).** Margin debt is dollar-
  denominated and scales with portfolio values; when prices rise, the *value* of
  marginable collateral rises and borrowing capacity expands, so margin debt
  growth is largely a function of *past* returns. A coincident by-product of a
  trend is a poor leading indicator -- the core finding here.

## Why the headline keeps recurring -- base rates and anecdotes

- **Krueger & Kennedy (1990), "An Analysis of the Super Bowl Stock Market
  Predictor"** (Journal of Finance 45(2), 691-697). The methodological template for
  this desk's folklore studies: a binary "signal" looks predictive only until you
  test it against the *unconditional* up-rate (~74% here). Margin-debt records
  cluster in long bull markets, so a few vivid post-record crashes (2000, 2007,
  2021) anchor the story while the many record-years that kept rising are forgotten.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica). Used for the slope t-stat in the predictive regression.
- **Welch's t-test.** Welch (1947) -- unequal-variance two-sample test, used for the
  tercile and record-event group comparisons.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the base-rate trap in its
  purest form -- a binary "predictor" that just re-packages the market's upward bias.
- **[Study 120 -- Excess-CAPE-Yield](../../120-excess-cape-yield/)** and other
  valuation-timing studies: macro level indicators that look like timing signals
  but mostly trend with the market.
