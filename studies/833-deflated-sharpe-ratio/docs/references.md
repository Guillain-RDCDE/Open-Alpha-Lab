# References & literature map — Study 833 (Deflated Sharpe Ratio)

## The claim under test

- **The source paper.** David H. **Bailey & Marcos López de Prado**, *"The Deflated Sharpe
  Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"* (Journal of
  Portfolio Management, 2014). Their result: when a researcher tries `N` strategy
  configurations and reports the best, the maximum sample Sharpe is biased upward, and under
  the null (no skill) its expectation **grows without bound in `N`**:

      E[max SR] ≈ √V · [ (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

  with `γ` the Euler-Mascheroni constant, `Z⁻¹` the inverse standard-normal CDF, and `V` the
  cross-trial variance of the Sharpe estimates. The **Deflated Sharpe Ratio** re-expresses the
  winner's Sharpe as the probability it exceeds that expected maximum, further correcting for
  the sample length `T` and the return skew/kurtosis (the Probabilistic Sharpe Ratio, Bailey &
  López de Prado 2012, is the `N = 1` special case). A naked Sharpe of 1.5 out of a 1,000-config
  sweep can deflate to a DSR near zero — "consistent with luck."

- **The specific test here.** We make the claim un-deniable by running the demonstration on a
  tape we *know* is empty: `N` **independent** strategies of iid, zero-drift daily returns
  (population Sharpe exactly 0). We measure the best sample Sharpe as `N` grows, compare it to
  the expected-maximum-Sharpe formula, deflate the winner, and confirm the DSR shrinks it to a
  coin flip — with an honest single strategy as the positive control the correction must spare.

## What we measure, and the honesty rails

- **Independent trials, the pure form.** The E[max] formula assumes independent trials; we use
  iid columns rather than `N` timing rules on one shared price path (which are cross-correlated
  through the common tape and shrink the *effective* trial count). The correlated crossover-grid
  version is study [344](../../344-backtest-overfitting/).
- **The Deflated Sharpe Ratio, moment-aware.** SR0 uses the empirical cross-trial SR dispersion
  (≈ the theoretical `1/(T−1)` under H0); the DSR denominator carries the Lo (2002) skew/kurtosis
  correction so heavy tails do not masquerade as significance.
- **Signal is NONE by construction.** A synthetic-only method demo: real free data can never
  *certify* zero edge, so the study is capped at NONE on the Signal axis (stated openly, like the
  desk's other method demos). The synthetic controls prove the machinery is *calibrated* — they
  are never cited to support a real-tape stamp.
- **Tradability is graded separately.** The winner is priced out-of-sample and charged a one-way
  × NAV round-trip; the in-sample dazzle collapses live — a Mirage by construction.

## Shared method citations

- **Bailey, D. & López de Prado, M. (2012)** — *The Sharpe Ratio Efficient Frontier* (the
  Probabilistic Sharpe Ratio, the DSR's `N = 1` case).
- **Bailey, Borwein, López de Prado & Zhu (2014)** — *Pseudo-Mathematics and Financial
  Charlatanism* (why an undisclosed trial count voids a backtest).
- **Lo, A. (2002)** — *The Statistics of Sharpe Ratios* (the skew/kurtosis-aware sampling
  variance of the Sharpe ratio used in the DSR denominator).
- **Harvey, C., Liu, Y. & Zhu, H. (2016)** — *…and the Cross-Section of Expected Returns* (the
  multiple-testing hurdle for the factor zoo; motivates a `t > 3` bar).
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the out-of-sample series).
- **Wilson, E. B. (1927)** — score interval for a binomial share (the hit-rate rail).

## Related desk studies (the dedup map — what this study is NOT)

- [344-backtest-overfitting](../../344-backtest-overfitting/) — applies the **same** DSR **and**
  PBO/CSCV to a grid of long/flat **moving-average crossover** rules run on a shared random-walk
  tape. Those trials are **correlated** through the common price path; this study isolates the
  DSR on **independent** trials — the clean form the E[max] formula assumes — and does *not* run
  PBO. Complementary halves of the Bailey-LdP toolkit.
- [590-sharpe-hacking](../../590-sharpe-hacking/) — inflates the **reported** Sharpe by
  **transforming the returns** (smoothing / leverage / vol-targeting), corrected by an
  autocorrelation adjustment. There the inflation comes from *measurement games on one series*;
  here it comes from *selecting the best of N series* — a different mechanism, a different cure.
- [346-multiple-testing](../../346-multiple-testing/) — turns a family of *t*-stats into
  discoveries via **Bonferroni / Holm / Benjamini-Hochberg** (FWER vs FDR) on *p*-values. This
  study works in **Sharpe/expected-maximum** space and deflates a **single selected winner**
  rather than counting how many of a battery survive a *p*-value correction.

None of the siblings demonstrate the **expected-maximum-Sharpe inflation on independent trials
and the DSR that corrects it** — this study's own axis.

## Reproducing

- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (offline, deterministic, seed 833).
