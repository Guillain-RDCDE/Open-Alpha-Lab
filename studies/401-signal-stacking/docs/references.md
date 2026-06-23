# References & literature map — Study 401 (Signal-Stacking)

## The claim under test

- **The viral pitch.** A recurring "edge the retail crowd has never seen" thread: take **K weak
  signals** — "none of which would pass a significance test alone, most hovering around 52%" —
  *standardise each into a z-score, weight by historical Sharpe, and sum into one composite
  "ζ-field."* The combination is said to "stay flat when signals disagree" and to produce a
  spectacular equity curve against buy-and-hold. The recipe has four moving parts, each rebuilt
  and named in [`strategy.py`](../signal_stacking/strategy.py): the composite z-score, the
  threshold position rule, the cost-charged backtest, and the headline curve.
- **The honest kernel and the catch.** The pitch is half-right. Combining weak-but-real,
  *decorrelated* signals genuinely raises the information coefficient — that is the legitimate
  mathematics of diversification. The omissions are the two words *real* and *decorrelated*: a
  stack of pure noise times nothing, redundant signals plateau far below the advertised boost, and
  "weighted by historical Sharpe" is a data-snooping step that manufactures the backtest.

## The √K law — the fundamental law of active management

- **Grinold & Kahn, *Active Portfolio Management* (2nd ed., 2000).** The **Fundamental Law of
  Active Management**: information ratio ≈ **IC × √breadth**, where IC is the per-bet information
  coefficient and *breadth* is the number of **independent** bets per year. Stacking K signals is
  the breadth dimension: the composite IC scales like √K **only when the bets are independent**.
  This is the exact law [`strategy.composite_ic`](../signal_stacking/strategy.py) measures
  (composite IC vs the mean single IC, against a √K reference). Grinold (1989), *The Fundamental
  Law of Active Management*, *Journal of Portfolio Management*, is the original statement.
- **Correlation is the breadth destroyer.** The √K in the law is √(*effective breadth*), and
  pairwise correlation collapses effective breadth toward 1. For K equicorrelated signals with
  pairwise correlation ρ, the variance-reduction (and hence the IC lift) is governed by
  1 / (1 + (K−1)ρ): as ρ → 1 the lift → 1 regardless of K. Clarke, de Silva & Thorley (2002),
  *Portfolio Constraints and the Fundamental Law of Active Management* (*Financial Analysts
  Journal*), formalise the *transfer coefficient* — why realised breadth is almost always far
  below the nominal count of signals. This is the knob `signal_corr` in
  [`data.synthetic_panel`](../signal_stacking/data.py).
- **Signal combination in practice.** The information-coefficient framing of multi-signal alpha is
  standard buy-side craft; z-scoring and equal- or IC-weighting of standardised signals is the
  textbook composite (e.g. Qian, Hua & Sorensen, *Quantitative Equity Portfolio Management*,
  2007). The honest version *estimates* the weights out of sample; the viral version *fits* them
  in sample, which is where the snooping enters.

## Why the "edge" needs an honest arbiter — randomization and snooping

- **Permutation / randomization test.** Because the composite is a flexible object fit to the
  data, its Sharpe must be judged against a null that breaks the signal→return link.
  [`strategy.permutation_pvalue`](../signal_stacking/strategy.py) holds the composite/position
  fixed, **shuffles the return vector** many times, and reports the fraction of reshuffled worlds
  whose Sharpe matches or beats the observed one — Fisher's randomization logic (R. A. Fisher,
  *The Design of Experiments*, 1935; Efron & Tibshirani, *An Introduction to the Bootstrap*,
  1993). On the null stack the observed Sharpe sits *inside* this distribution (p ≈ 0.38); on the
  real decorrelated edge it is far outside (p ≈ 0.000).
- **Data-snooping / selection on the signals + weights.** Selecting the best signals and weighting
  them by in-sample Sharpe is precisely the multiple-testing trap. White (2000),
  *A Reality Check for Data Snooping* (*Econometrica*), and Romano & Wolf (2005),
  *Stepwise Multiple Testing as Formalized Data Snooping*, give the formal corrections;
  Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* (*Review of Financial
  Studies*), and Bailey & López de Prado (2014), *The Deflated Sharpe Ratio* (*Journal of
  Portfolio Management*), quantify how a Sharpe *selected* from many trials must be deflated.
  [`strategy.snoop_split`](../signal_stacking/strategy.py) demonstrates the tax directly: pick on
  the first half, pay on the second — and on a real tape with no edge the in-sample winner is
  often a **pure decoy**.
- **Block-bootstrap inference.** [`strategy.block_bootstrap_ci`](../signal_stacking/strategy.py)
  and the Newey-West HAC *t* ([`strategy.hac_tstat`](../signal_stacking/strategy.py); Newey &
  West, 1987) account for autocorrelation when judging the composite's mean return — the standard
  desk idiom, so a serially-correlated daily series is not over-credited.

## Method lineage (the desk's shared engine)

- **The composite & position rule.**
  [`strategy.composite_score`](../signal_stacking/strategy.py) builds the weighted, re-standardised
  z-score; [`strategy.position_from_score`](../signal_stacking/strategy.py) trades its sign past a
  threshold (the "flat when signals disagree" middle). One execution lag is baked into the panel
  ([`data.synthetic_panel`](../signal_stacking/data.py) / `load_real`), never shifted again.
- **The √K test, permutation p, and snoop split.**
  [`strategy.composite_ic`](../signal_stacking/strategy.py),
  [`strategy.permutation_pvalue`](../signal_stacking/strategy.py), and
  [`strategy.snoop_split`](../signal_stacking/strategy.py) are the three honest arbiters; the full
  bundle is assembled by [`strategy.run_experiment`](../signal_stacking/strategy.py).
- **The deterministic synthetic panel.**
  [`data.synthetic_panel`](../signal_stacking/data.py) is analytic: every per-signal IC
  (`corr = signal_ic`) and cross-correlation (`signal_ic² + (1−signal_ic²)·signal_corr`) is known
  in closed form, so the null and the positive control are exact.

## Data sources used here

- **yfinance** daily SPY total-return closes, 1995-10-16 → 2026-06-18, cached under
  `_cache/stacking_SPY.parquet` (fingerprint `a4bfd3d8a8cd`). The synthetic panel needs no
  network. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Studies 343–350 — research-method demos](../../343-data-mining-roulette/)**: the family this
  study belongs to — [343 data-mining-roulette](../../343-data-mining-roulette/),
  [344 backtest-overfitting](../../344-backtest-overfitting/),
  [348 curve-fitting](../../348-curve-fitting/), [350 dartboard-portfolio](../../350-dartboard-portfolio/).
  Where 343 randomises the *rule* and 348 fits a curve, this study takes the *combination* claim
  head-on: the maths of why fifty weak signals are powerful only when real **and** different.
- **[Study 399 — Kalshi-Efficiency](../../399-kalshi-efficiency/)**: the same methods-demo shape —
  a machine validated on a controlled book, with the "real" tape explicitly illustrative and no
  `REAL` stamp claimed.
