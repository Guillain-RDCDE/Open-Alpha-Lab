# Sources & literature map — Study 996 (The Palindrome Portfolio)

## Data mining in finance

- **Lo, A. W. & MacKinlay, A. C. (1990), "Data-Snooping Biases in Tests of Financial Asset
  Pricing Models", *Review of Financial Studies* 3(3), 431-467.** The founding treatment, and
  the source of the observation that the community's collective search is far larger than any
  individual paper's.
- **Sullivan, R., Timmermann, A. & White, H. (1999), "Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap", *Journal of Finance* 54(5), 1647-1691.** Runs ~8,000
  technical rules and shows how much of the best one's performance is search. The direct
  ancestor of this study's method, applied to rules with a mechanism rather than without one.
- **Sullivan, R., Timmermann, A. & White, H. (2001), "Dangers of Data Mining: The Case of
  Calendar Effects in Stock Returns", *Journal of Econometrics* 105(1), 249-286.** Calendar
  effects specifically — the closest paper to this one, and it reaches the same conclusion about
  effects that *do* have candidate mechanisms.
- **Harvey, C. R., Liu, Y. & Zhu, H. (2016), "…and the Cross-Section of Expected Returns",
  *Review of Financial Studies* 29(1), 5-68.** Argues the *t*-statistic hurdle for a new factor
  should be about 3.0 rather than 2.0, precisely because of the search that preceded it.

## The corrections

- **White, H. (2000), "A Reality Check for Data Snooping", *Econometrica* 68(5), 1097-1126.**
  The bootstrap reality check — the formal version of this study's shuffle test.
- **Benjamini, Y. & Hochberg, Y. (1995), "Controlling the False Discovery Rate", *JRSS B* 57(1),
  289-300.** The FDR procedure implemented in `multiple_testing_summary`.
- **Romano, J. P. & Wolf, M. (2005), "Stepwise Multiple Testing as Formalized Data Snooping",
  *Econometrica* 73(4), 1237-1282.** The stepwise improvement on White's reality check.
- **Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio", *Journal of Portfolio
  Management* 40(5), 94-107.** Adjusts a Sharpe for the number of trials that produced it —
  `deflated_t` is a crude cousin.
- **Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014), "Pseudo-Mathematics and
  Financial Charlatanism", *Notices of the AMS* 61(5), 458-471.** The polemical version, and the
  source of the "minimum backtest length" idea.

## Replication more broadly

- **Ioannidis, J. P. A. (2005), "Why Most Published Research Findings Are False", *PLoS Medicine*
  2(8), e124.** Outside finance, and the clearest statement of why the prior matters as much as
  the *p*-value — which is exactly why a hypothesis with no mechanism is the right calibration
  target.
- **Hou, K., Xue, C. & Zhang, L. (2020), "Replicating Anomalies", *Review of Financial Studies*
  33(5), 2019-2133.** 452 published anomalies re-tested; most do not survive.

## Neighbours on this desk

**067-monday-effect**, **552-weekend-effect**, **283-sell-in-may**, **410-santa-rally**,
**718-p-hacking-simulation**, **860-backtest-overfitting**.
