# References & literature map — Study 522 (Percent Operating Accruals)

## The foundational paper

- **Hafzalla, N., Lundholm, R., & Van Winkle, E. M. (2011).** *Percent accruals.* The
  Accounting Review, 86(1), 209–236. The paper this study replicates. Takes Sloan's
  operating accrual and scales it by the **absolute value of net income** ("percent
  accruals") rather than by average total assets. Argues that percent accruals identify
  firms whose earnings are *most* and *least* reliant on accrual estimates, producing a
  cross-sectional return spread that is *larger* than the Sloan (1996) asset-scaled accrual,
  especially in the extreme deciles. Original sample: US Compustat/CRSP firms 1989–2008,
  value-weighted hedge returns. Also examines "percent total accruals" (using the broader
  Richardson et al. balance-sheet accrual) scaled by earnings.

## The signal it sharpens — Sloan accruals

- **Sloan, R. G. (1996).** *Do stock prices fully reflect information in accruals and cash
  flows about future earnings?* The Accounting Review, 71(3), 289–315. The seminal accruals
  anomaly: high-accrual (earnings > cash flows) firms earn low future returns because the
  accrual component of earnings is less persistent than the cash component and investors do
  not discount it. Scales accruals by **average total assets**. This is the desk's
  [Study 231 — Sloan-Accruals](../231-sloan-accruals/); Study 522 is the
  earnings-scaled *percent* variant HLVW propose as the sharper sort.

- **Richardson, S. A., Sloan, R. G., Soliman, M. T., & Tuna, I. (2005).** *Accrual
  reliability, earnings persistence and stock prices.* Journal of Accounting and Economics,
  39(3), 437–485. Decomposes accruals into working-capital and long-term components and
  shows the cash-flow-statement accrual (post-SFAS 95) is the cleaner measure — the basis
  for the (Net Income − Operating Cash Flow) operating accrual used here.

## Replication, decay and limits to arbitrage

- **Green, J., Hand, J. R. M., & Soliman, M. T. (2011).** *The supraview of return
  predictive signals.* Review of Accounting Studies, 16(3), 635–664. Documents substantial
  post-2000 decay of the accruals anomaly as the signal became widely known and arbitrage
  capital closed it — largest decay in liquid large caps, precisely the names in this study's
  basket. Directly explains the None verdict here.

- **Mashruwala, C., Rajgopal, S., & Shevlin, T. (2006).** *Why is the accrual anomaly not
  arbitraged away? The role of idiosyncratic risk and transaction costs.* Journal of
  Accounting and Economics, 42(1–2), 3–33. The accruals anomaly concentrates in high-
  idiosyncratic-risk, high-transaction-cost stocks — i.e. *not* S&P 500 large caps. A fixed
  large-cap survivor basket is the wrong place to find it; consistent with our flat result.

- **Lev, B., & Nissim, D. (2006).** *The persistence of the accruals anomaly.* Contemporary
  Accounting Research, 23(1), 193–226. The anomaly persists primarily in small and microcap
  stocks with limited institutional ownership — limits-to-arbitrage, not present in large caps.

## Survivorship bias

- **Kothari, S. P., Sabino, J., & Zach, T. (2005).** *Implications of survival and data
  trimming for tests of market efficiency.* Journal of Accounting and Economics, 39(1),
  129–161. Why a current-survivor basket inflates any accrual hedge: the high-accrual firms
  that collapsed and delisted (the short leg's best prey) are silently excluded. Every number
  in [docs/results.md](results.md) is therefore an upper bound.

## Method lineage (the desk's shared engine)

- **Newey, W. K., & West, K. D. (1987).** *A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.* Econometrica,
  55(3), 703–708. The HAC t-stat in [`strategy.summary`](../percent_operating_accruals/strategy.py).
- **Reporting-lag discipline.** Fundamentals from fiscal year y predict returns in calendar
  year y+1 — the same conservative one-execution-lag convention used in Studies 231 and 121.
- **Label-shuffle placebo.** Permuting the signal within each year breaks the signal→return
  link while preserving the return marginals — the desk's standard null for a cross-sectional
  sort.

## Related desk studies

- **[Study 231 — Sloan-Accruals](../231-sloan-accruals/)**: the asset-scaled accrual HLVW
  set out to beat. On a comparable survivor panel it reached HAC t = +2.73; the percent-scaled
  version here lands t = +0.84 — the direct head-to-head behind this study's 3rd axis.
- **[Study 153 — Net-Operating-Assets](../153-net-operating-assets/)**: Hirshleifer et al.
  (2004) balance-sheet generalisation of accruals — same earnings-quality family.
- **[Study 122 — Gross-Profitability](../122-gross-profitability/)**: Novy-Marx (2013) profitability factor;
  earnings-quality cousin on the same large-cap survivor universe.
