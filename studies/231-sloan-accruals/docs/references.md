# References & literature map — Study 231 (Sloan Accruals)

## The foundational paper

- **Sloan, R. G. (1996).** *Do stock prices fully reflect information in accruals and
  cash flows about future earnings?* The Accounting Review, 71(3), 289–315. The seminal
  accruals-anomaly paper. Decomposes earnings into a cash-flow component and an accruals
  component; shows that while earnings persistence is high for both components, markets
  price them equally when in fact the accruals component is less persistent. High accruals
  (earnings > cash flows) predict low future stock returns; low accruals (cash-backed
  earnings) predict high future returns. Original sample: US firms on Compustat/CRSP
  1962–1991. Documents a hedge of ~10%/yr on a value-weighted basis on the full universe.

## Replication and decay

- **Green, J., Hand, J. R. M., & Soliman, M. T. (2011).** *The supraview of return
  predictive signals.* Review of Accounting Studies, 16(3), 635–664. Documents that the
  Sloan accruals anomaly has substantially decayed in the post-2000 period as the signal
  became widely known and arbitrage capital closed the mispricing. Finds the effect
  reduced or disappeared in the 2000s for large-cap, liquid stocks.

- **Richardson, S. A., Sloan, R. G., Soliman, M. T., & Tuna, I. (2005).** *Accrual
  reliability, earnings persistence and stock prices.* Journal of Accounting and
  Economics, 39(3), 437–485. Extends Sloan (1996) to decompose accruals into working
  capital and long-term components; finds the long-term component (investment in PP&E and
  intangibles) is also negatively predictive of returns. Also clarifies that the cash-flow
  statement version of accruals (post-SFAS 95) is cleaner than the balance-sheet version.

- **Mashruwala, C., Rajgopal, S., & Shevlin, T. (2006).** *Why is the accrual anomaly
  not arbitraged away? The role of idiosyncratic risk and transaction costs.* Journal of
  Accounting and Economics, 42(1–2), 3–33. Finds the accruals anomaly concentrates in
  stocks with high idiosyncratic risk and high transaction costs — i.e., precisely those
  stocks *not* in the S&P 500. Consistent with our finding that the short-side (high-accrual
  S&P 500 names) shows negligible underperformance.

- **Lev, B., & Nissim, D. (2006).** *The persistence of the accruals anomaly.* Contemporary
  Accounting Research, 23(1), 193–226. Confirms the anomaly persists in the sample through
  the early 2000s but is concentrated in small and microcap stocks with limited institutional
  ownership, consistent with limits-to-arbitrage explanations.

## Related signals

- **Hirshleifer, D., Hou, K., Teoh, S. H., & Zhang, Y. (2004).** *Do investors overvalue
  firms with bloated balance sheets?* Journal of Accounting and Economics, 38, 297–331.
  Generalises Sloan's working-capital accrual to a balance-sheet Net Operating Assets
  (NOA) measure — covered in [Study 153 — Net-Operating-Assets](../../153-net-operating-assets/).
  NOA is the full balance-sheet expansion including long-term assets.

- **Cooper, M. J., Gulen, H., & Schill, M. J. (2008).** *Asset growth and the cross
  section of stock returns.* Journal of Finance, 63(4), 1609–1651. Documents a broad
  asset-growth effect (high growth → low returns) that encompasses the accruals signal;
  largest among small stocks, attenuated for large caps.

## Economic mechanism debate

- **Mispricing view (Sloan 1996; Xie 2001).** Investors fail to see through the lower
  persistence of accrual earnings; when accruals mean-revert, earnings disappoint and
  prices fall. Supporting evidence: post-earnings-announcement drift patterns.

- **Rational-risk view.** High accruals may proxy for high investment activity with
  higher risk (more assets deployed = more operating leverage). Fama & French (2006) argue
  investment is a priced risk factor, but cannot fully account for the magnitude.

- **Post-publication decay.** Both Green et al. (2011) and Richardson et al. (2010)
  document material decay after the Sloan paper was published and became widely followed.
  Large-cap, liquid names were among the first to have the mispricing arbitraged away.

## Survivorship bias and data limitations

- The EDGAR cache used here covers *current* S&P 500 members only, projected backwards.
  High-accrual firms that were subsequently expelled from the index (for earnings
  manipulations or poor performance) are excluded — precisely the most damaging high-accrual
  disasters. See **Kothari, S. P., Sabino, J., & Zach, T. (2005).** *Implications of
  survival and data trimming for tests of market efficiency.* Journal of Accounting and
  Economics, 39(1), 129–161.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summary`](../sloan_accruals/strategy.py).
- **Reporting lag discipline.** Fundamentals from fiscal year y predict returns in
  calendar year y+1 — the same conservative lag used in Studies 153 and 121.

## Related desk studies

- **[Study 153 — Net-Operating-Assets](../../153-net-operating-assets/)**: the Hirshleifer
  et al. (2004) balance-sheet generalisation of Sloan accruals — same EDGAR panel;
  NOA captures the entire operating-asset surplus, not just working-capital accruals.
- **[Study 122 — Gross-Profit](../../122-gross-profit/)**: Novy-Marx (2013) profitability
  factor on the same EDGAR panel; closely related to earnings quality.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt quality+value rank;
  quality screens out high-accrual, low-cash-flow names indirectly.
