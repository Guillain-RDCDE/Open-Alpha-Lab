# Sources & literature map — Study 39 (Black-Box)

## The claim's source

- **Z. Kakushadze & J. A. Serur (2018), *151 Trading Strategies*, §18.2 — "Neural networks"
  (cryptocurrency trading).** The catalogue entry for feeding a neural network price-derived features and
  letting it predict the next move. SSRN `3247865` · arXiv [1912.04492](https://arxiv.org/abs/1912.04492).
  *(Copyrighted; not redistributed.)*

## The backtest-overfitting literature — why a black box flatters itself

- **Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014), "Pseudo-Mathematics and Financial
  Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance," *Notices of the AMS*
  61(5)**, and **"The Probability of Backtest Overfitting," *Journal of Computational Finance* (2017).**
  The formal statement of the problem: with enough flexibility / trials, an in-sample Sharpe tells you
  almost nothing about out-of-sample performance. The **shuffled-label control** in
  [`extension.md`](extension.md) is a direct, intuitive instance of their point.
- **López de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley.** Chapters on
  cross-validation in finance, the **Deflated Sharpe Ratio**, and why naïve fit-and-predict on
  serially-correlated financial data manufactures spurious skill. The motivation for the walk-forward
  (out-of-sample) protocol used here.
- **White, H. (2000), "A Reality Check for Data Snooping," *Econometrica* 68(5).** The data-snooping
  correction in the shared [`quantlab/`](../../../quantlab/) engine — the same hazard, formalised.

## The trap sibling

- **Study 22 — Crystal-Ball (§8.1)**, [`../../22-crystal-ball/`](../../22-crystal-ball/). The desk's other
  *backtest-trap* study: a two-sided HP filter that silently encodes the future and manufactures a
  Sharpe-2 backtest out of a coin flip. Black-Box is its machine-learning cousin — there the leak was
  look-ahead in the feature; here it is **over-parameterisation memorising noise in-sample**. Both
  vanish the moment you compute the only number a live trader could have earned.

## On the difficulty of predicting crypto direction

- **Fama, E. (1970), "Efficient Capital Markets," *Journal of Finance* 25(2).** The weak-form efficiency
  null: past prices alone should not predict future direction profitably after costs. Our walk-forward
  result is consistent with it for daily crypto direction.

## The shared method

- **Newey-West (1987)** HAC SEs · **Lo (2002)** Sharpe inference · **White (2000)** Reality Check —
  the shared [`quantlab/`](../../../quantlab/) engine; see [`METHODOLOGY.md`](../../../METHODOLOGY.md).
