# References & literature map — Study 834 (Minimum Backtest Length / MinTRL)

## The claim, at full strength

Every backtest reports a Sharpe ratio, and almost none reports the one number that makes it
interpretable: the **length of the track record**. A Sharpe estimated over a finite history is a noisy
statistic whose standard error shrinks only as `1/√T`; the shorter the record, the wider the band, and
below a certain length *any* observed Sharpe — however gaudy — is statistically indistinguishable from
zero. Bailey, Borwein, López de Prado & Zhu turn that into a hard threshold: for a target annualised
Sharpe there is a **Minimum Track Record Length (MinTRL)** below which you cannot reject "true Sharpe
≤ 0" at your chosen confidence. Its companion, the **Probabilistic Sharpe Ratio (PSR)**, reports the
probability that the true Sharpe beats a benchmark given the observed Sharpe, the return moments, and
the length. This study makes the point undeniable by running it on a world we *built* with zero edge,
so every pretty short-sample Sharpe is provably luck.

## The source papers

- **Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014)**, *"Pseudo-Mathematics and
  Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance."* *Notices
  of the American Mathematical Society* 61(5). The polemic that names the disease: most published
  backtests are statistically empty because the researcher tried many configurations over a short
  history and reported only the luckiest, without disclosing the number of trials or the minimum
  length needed to trust the result. **The source of this study's headline claim.**
- **Bailey, D. & López de Prado, M. (2012)**, *"The Sharpe Ratio Efficient Frontier."* *Journal of
  Risk* 15(2). The paper that derives the **Probabilistic Sharpe Ratio** and the closed-form
  **MinTRL**: `MinTRL_obs = 1 + [1 − skew·sr + (kurt−1)/4·sr²]·(Z_conf/(sr − sr*))²`. The skew/kurtosis
  factor is exactly the third/fourth-moment correction implemented here.
- **Lo, A. (2002)**, *"The Statistics of Sharpe Ratios."* *Financial Analysts Journal* 58(4). The
  asymptotic distribution of the estimated Sharpe ratio, including the skewness/kurtosis terms in its
  standard error — the statistical backbone of both PSR and MinTRL.

## What we measure, and the honesty rails

- **MinTRL and PSR, closed form.** `min_trl_years` and `probabilistic_sharpe_ratio` implement the
  Bailey–López de Prado formulas directly; MinTRL is exactly the `n` that makes `PSR = conf`, which the
  tests assert to machine precision.
- **The rule of thumb.** For Gaussian daily returns MinTRL collapses to `(Z_conf/SR_ann)²` years; the
  study reports both the exact and the approximate value so the quadratic blow-up (halve the Sharpe →
  quadruple the length) is legible.
- **The moment correction is separated out.** Negative skewness and excess kurtosis *lengthen* the
  requirement; we show this at the monthly reporting frequency where the per-observation Sharpe — and
  hence the correction — is largest, using a generator whose population skew/kurtosis are **closed
  form** (a standardised negated-gamma shock), so the formula is tested against a known truth, not an
  estimate.
- **Two-sided honesty via simulation.** A Monte-Carlo over 4,000 backtests shows (a) the PSR test is
  **calibrated** on the null — it fires at its nominal 5% false-positive rate, never manufacturing a
  false edge; and (b) it is **underpowered** on genuine edge in short windows — a true Sharpe-1 is only
  ~50% detectable at its MinTRL and needs ~10.8 years for 95% power.
- **Synthetic-only, capped at NONE.** Real free data can never certify "true Sharpe = 0", so there is
  no real tape and no `REAL` stamp is possible — stated openly, as on the desk's other method demos.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent (HAC)
  covariance; the desk's `newey_west_t`, retained here for the shared inference toolkit.
- **Wilson, E. B. (1927)** — score interval for a binomial share; the error bars on every simulated
  rejection rate.
- **Harvey, C. & Liu, Y. (2015)**, *"Backtesting."* *Journal of Portfolio Management* 42(1) — the
  multiple-testing haircut to the Sharpe *t*-statistic, a complementary discipline to MinTRL (too many
  trials vs too short a record).

## Related desk studies (the dedup map — what this study is NOT)

- **[344 — Backtest-Overfitting](../../344-backtest-overfitting/)** — the *multiple-trials* pitfall:
  grid-search many rules on one dataset and the best is guaranteed to dazzle. It quantifies the trap
  with the **Deflated Sharpe Ratio** and **PBO**. Study 834 is the *single-strategy* companion: even
  with **one** honest hypothesis and **no** search, a Sharpe over a **too-short** history is
  indistinguishable from luck — the length axis, not the trial-count axis.
- **[833 — Deflated-Sharpe](../../833-deflated-sharpe/)** — the Deflated Sharpe Ratio: haircut a
  reported Sharpe for the **number of trials** and the return moments. Study 834 fixes the trial count
  at one and isolates the **track-record-length** requirement (MinTRL/PSR) — the same Bailey–López de
  Prado family, the complementary knob.
- **[345 — Survivorship](../../345-survivorship/)** — a *data-construction* bias (dead names dropped
  from the universe inflates backtest returns). Study 834 assumes clean data and asks a purely
  *statistical* question: how long must even an unbiased track record be to trust its Sharpe.

None of the siblings computes the **minimum length** a single strategy's track record must reach
before its Sharpe clears a significance bar — this study's own axis.

## Data sources

- **No real market data and no network** — a synthetic/simulation-only method demonstration. The
  deterministic seeded worlds live in [`min_backtest_length/data.py`](../min_backtest_length/data.py);
  all headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
