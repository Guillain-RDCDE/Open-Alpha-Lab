# References & literature map — Study 240 (Dividend-Initiation)

## The claim under test

- **The folk claim.** A company paying a dividend for the **first time** signals to the
  market that management is confident about future earnings and cash flows. Because
  dividends are sticky (costly to cut), an initiation is a credible commitment signal.
  The implication: buy firms that just initiated dividends and enjoy a re-rating as the
  market updates its expectations upward. The signalling hypothesis (Bhattacharya 1979)
  provides the theoretical underpinning for this claim.

## The academic literature on dividend initiation

- **Bhattacharya, S. (1979).** *Imperfect information, dividend policy, and 'the bird in
  the hand' fallacy.* Bell Journal of Economics 10(1): 259–270. — The foundational
  signalling paper. Dividends serve as a costly signal of future profitability because
  only firms with high future cash flows can afford the signal without cutting dividends
  later. Initiation events are particularly strong signals under this framework.

- **Miller, M. H. & Rock, K. (1985).** *Dividend policy under asymmetric information.*
  Journal of Finance 40(4): 1031–1051. — Extends the signalling framework; dividend
  changes reveal information about current earnings. Initiation of dividends is interpreted
  as management's announcement that current and future earnings are high enough to sustain
  payout.

- **Asquith, P. & Mullins, D. W. (1983).** *The impact of initiating dividend payments
  on shareholders' wealth.* Journal of Business 56(1): 77–96. — Classic event-study
  paper showing positive abnormal returns around dividend initiation announcements (~3–4%
  over days [-10, +10]). This is the short-window event-study evidence that motivates
  the claim; our study tests the LONG-HORIZON (one-year forward) effect, which is
  distinct.

- **Michaely, R., Thaler, R. H. & Womack, K. L. (1995).** *Price reactions to dividend
  initiations and omissions: overreaction or drift?* Journal of Finance 50(2): 573–608. —
  Finds a short-window reaction (+3.4% around initiation) but longer-run drift is
  weaker and depends on the sample. Initiations in the 1960s–1980s show stronger drift
  than later samples, consistent with declining information asymmetry.

- **Grullon, G., Michaely, R. & Swaminathan, B. (2002).** *Are dividend changes a sign
  of firm maturity?* Journal of Business 75(3): 387–424. — The "maturity hypothesis":
  firms that initiate dividends are signalling a transition to a lower-growth, lower-risk
  profile (maturity), not necessarily higher *future* earnings growth. If investors expect
  the growth stock to become a value stock, the re-rating may be incomplete or negative.
  Directly relevant to why our large-cap initiators (MSFT, AAPL, CSCO) showed modest
  forward returns.

- **Fama, E. F. & French, K. R. (2001).** *Disappearing dividends: changing firm
  characteristics or lower propensity to pay?* Journal of Financial Economics 60(1): 3–43.
  — Fewer firms pay dividends today than in 1978; those that initiate tend to be
  established large-cap firms signalling maturity, not small-cap growth surprises. This
  changes the signal content of initiation events in a world where dividend non-payment is
  the norm for growth firms.

- **DeAngelo, H., DeAngelo, L. & Skinner, D. J. (1992).** *Dividends and losses.* Journal
  of Finance 47(5): 1837–1863. — Examines dividend reductions and initiations around
  earnings surprises. Initiation events are most informative when accompanied by a genuine
  earnings level change; in large mature firms with predictable earnings, initiation may
  convey less new information.

## The survivorship and small-sample caveats

- **Brown, S. J., Goetzmann, W., & Ross, S. A. (1995).** *Survival.* Journal of Finance
  50(3): 853–873. — Classic survivorship-bias demonstration. In our study the entire 50-
  name universe is pre-selected for large-cap survival as of 2026; this inflates all arm
  returns but does not explain the within-basket spread direction.

- **Barber, B. M. & Lyon, J. D. (1997).** *Detecting long-run abnormal stock returns: the
  empirical power of test statistics.* Journal of Financial Economics 43(3): 341–372. —
  Long-horizon event studies (1+ year holding period) are notoriously noisy. Buy-and-hold
  abnormal returns have heavy right tails; standard t-tests are mis-sized with small event
  samples. Our n = 8 initiator-year observations are well below the sample sizes needed
  for reliable long-horizon inference.

- **Mitchell, M. L. & Stafford, E. (2000).** *Managerial decisions and long-term stock
  price performance.* Journal of Business 73(3): 287–329. — Long-run post-event returns
  are difficult to distinguish from factor-loading changes; event firms often load
  differently on size/value/momentum after the event, and failure to control for these
  produces spurious alphas.

## Competing and complementary desk studies

- **[Study 88 — Dogs-of-the-Dow](../../88-dogs-of-the-dow/)**: whether high-yield Dow
  payers beat the index — the yield (not initiation) angle. Related question: do the
  highest-yielding initiators earn a larger premium?

- **[Study 201 — Dividend-Growth](../../201-dividend-growth/)**: companies with
  consecutive dividend raise streaks. The question here is: once you have initiated, does
  *growing* the dividend add incremental return?

- **[Study 122 — Gross-Profitability](../../122-gross-profitability/)**: the Novy-Marx
  quality factor. Dividend initiation is partly a quality signal; gross profitability
  captures a related but more direct quality measure.

- **[Study 160 — Split](../../160-split/)**: stock splits as signalling events — a
  structural cousin: management voluntary signals of confidence in future appreciation.
  Compares initiation event-study evidence with the split literature.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3):
  703–708. Used in `strategy.summary` for the annual spread series.

- **yfinance total-return prices.** Auto-adjusted closes (`auto_adjust=True`) incorporate
  dividend reinvestment and split adjustments. The `.dividends` series provides the raw
  cash dividend amounts used to identify initiation years.

- **Annual rebalance convention.** First-dividend year identified from the full available
  history (no look-ahead in the classification sense, but forward return uses the year-end
  signal); position entered January 1 of *y+1*; closed December 31 of *y+1*.
