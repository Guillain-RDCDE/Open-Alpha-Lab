# References & literature map — Study 198 (Cash-Holdings)

## The foundational paper

- **Palazzo, B. (2012).** *Cash holdings, risk, and expected returns.* Journal of
  Financial Economics, 104(1), 162–185. The study this desk tests directly. Palazzo
  argues that firms hoard cash because they face high external financing costs, and that
  cash holdings proxy for financial constraints. Investors demand a premium for holding
  constrained firms, producing a positive cross-sectional relationship between
  Cash-to-Assets and future stock returns. On the CRSP universe: top-minus-bottom cash
  quintile earns roughly +3 to +5%/yr on a risk-adjusted basis. The effect is
  concentrated among small, financially constrained firms.

## Replication and related work

- **Bates, T. W., Kahle, K. M., & Stulz, R. M. (2009).** *Why do U.S. firms hold so
  much more cash than they used to?* Journal of Finance, 64(5), 1985–2021. Documents
  the secular rise in corporate cash holdings from the 1980s to 2000s, driven by
  increased R&D, cash-flow volatility, and reduced dividends. Provides the empirical
  backdrop for Palazzo's premium: if cash has risen because constraints have tightened,
  the cross-sectional dispersion in cash carries more information over time.

- **Faulkender, M., & Wang, R. (2006).** *Corporate financial policy and the value of
  cash.* Journal of Finance, 61(4), 1957–1990. Estimates the marginal value of cash to
  shareholders. Finds that an extra dollar of cash is worth more for financially
  constrained firms (consistent with Palazzo's mechanism) and less for firms with high
  cash levels already (diminishing returns). The cross-sectional heterogeneity in the
  value of cash is the theoretical foundation for the premium.

- **Pinkowitz, L., Stulz, R., & Williamson, R. (2006).** *Does the contribution of
  corporate cash holdings and dividends to firm value depend on governance? A
  cross-country analysis.* Journal of Finance, 61(6), 2725–2751. Finds that cash is
  worth less in countries with weak shareholder rights (agency problem: cash is likely
  to be wasted). On US firms, where governance is stronger, the cash premium is more
  credibly a constraint proxy.

- **Gao, H., Harford, J., & Li, K. (2013).** *Determinants of corporate cash policy:
  Insights from private firms.* Journal of Financial Economics, 109(3), 623–639.
  Compares cash holdings of public vs private firms. Private (more constrained) firms
  hold less cash than public firms of similar size, suggesting cash accumulation is not
  purely a constraint response for large public firms — complicating the Palazzo story
  for S&P 500 names.

- **Denis, D. J., & Sibilkov, V. (2010).** *Financial constraints, investment, and the
  value of cash holdings.* Review of Financial Studies, 23(1), 247–269. Shows that cash
  holdings are more valuable for financially constrained firms, consistent with
  Palazzo's mechanism — but also that the constraint-premium is not identifiable from
  cash alone among large, unconstrained firms.

## Why the mechanism fails for S&P 500 large caps

- The Palazzo (2012) premium is strongest among **small firms** measured by Kaplan-Zingales
  or Whited-Wu financial constraint indices. Large S&P 500 constituents have open access
  to commercial paper markets, revolving credit lines, and investment-grade bond markets.
  For these firms, cash holdings reflect capital allocation policy (or sector mix, e.g.
  tech), not binding financial constraints.

- The high-cash quintile in the S&P 500 panel is dominated by large-cap technology and
  healthcare firms (Apple, Alphabet, Microsoft, Johnson & Johnson, Pfizer). These
  happened to be the highest-performing sector over 2008–2026 for reasons unrelated to
  their cash ratios. The apparent premium is a **latent factor exposure**, not a cash
  premium.

## Survivorship bias and data limitations

- The EDGAR cache used here covers *current* S&P 500 members only, projected backwards.
  This excludes companies that were removed from the index for distress, bankruptcy, or
  delistment — precisely the cash-burning firms that would drive the signal in the
  *wrong* direction. A high-cash firm that subsequently exhausted reserves and failed
  (high-cash → distress → delisted) is absent from the data. See **Kothari, S. P.,
  Sabino, J., & Zach, T. (2005).** *Implications of survival and data trimming for
  tests of market efficiency.* Journal of Accounting and Economics, 39(1), 129–161.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summary`](../cash_holdings/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Reporting lag discipline.** Fundamentals from fiscal year y predict returns in
  calendar year y+1 — the same conservative lag used in Studies 52, 65, 121, and 153 on
  this desk.

## Related desk studies

- **[Study 153 — Net-Operating-Assets](../../153-net-operating-assets/)**: balance-sheet
  bloat anomaly (Hirshleifer et al. 2004), same EDGAR panel, same survivorship-bias caveat.
- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: accruals anomaly (Sloan 1996),
  working-capital counterpart to cash hoarding.
- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski F-score, which uses cash
  flow from operations as one of its nine signals.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt's quality+value,
  which combines return-on-capital with earnings yield — partially correlated with
  cash-richness via the profitability channel.
