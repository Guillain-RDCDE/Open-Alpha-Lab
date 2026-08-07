# References & literature map — Study 804 (Realized-Kurtosis Premium)

## The claim under test

- **The source paper.** Diego **Amaya, Peter Christoffersen, Kris Jacobs & Aurelio
  Vasquez**, *"Does Realized Skewness Predict the Cross-Section of Equity Returns?"*
  (Journal of Financial Economics, 2015). Building weekly **realized** moments from
  high-frequency returns, their headline result is a robust **negative** realized-*skewness*
  relation. Crucially for this study, the same paper also tests realized **kurtosis** — a
  name's recent fat-tailedness (the fourth standardised moment) — and finds it a **weak /
  ambiguous** predictor: only marginally priced, with a sign that is not robust and that is
  largely **subsumed** once skewness and volatility are controlled for.
- **Why kurtosis is the ambiguous cousin.** Kurtosis measures the *symmetric* fat-tailedness
  of a return distribution, mixing the up-tail and the down-tail. Whatever premium a fat tail
  commands is mostly captured by two cleaner signals — the **asymmetry** of the tail (skewness)
  and the **size** of the tail (volatility) — leaving realized kurtosis with little independent
  predictive content. The paper's own tables show it as the weakest of the three realized
  moments, which is precisely the null-ish result we set out to reproduce honestly.
- **The specific test here.** We take the self-contained daily version: sort a liquid US
  cross-section on its **trailing realized kurtosis of daily returns** and measure the forward
  return of the equal-weight long-high-kurt / short-low-kurt book, with a Newey-West *t*, a
  permutation placebo, a two-era robustness cut, a costed timer, and a seeded synthetic positive
  control. (Daily returns are a coarser kurtosis estimator than the paper's intraday sampling,
  so the magnitudes here are conservative and the fourth-moment noise is larger.)

## What we measure, and the honesty rails

- **Realized kurtosis, no free model.** For each name, the rolling `window`-day population
  kurtosis of daily simple returns (`m4 / m2**2`, the fourth standardised central moment),
  computed **vectorised** via the raw-moment expansion (`m4 = e4 − 4·mean·e3 + 6·mean²·e2 −
  3·mean⁴`) — never `rolling.apply`, which would be orders of magnitude slower.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing kurtosis
  **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short spread —
  an overlapping-formation signal is serially correlated, so a plain *t* (or a permutation test)
  would overstate significance. A one-sample *t* and a pooled Welch *t* (high-kurt book vs
  low-kurt book) cross-check. A **1,000-permutation placebo** breaks the signal → forward-return
  link to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent, so
  the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and the
  short book pays borrow — the honest test of whether a small daily spread survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Boyer, B., Mitton, T. & Vorkink, K. (2010)** — expected idiosyncratic skewness, the
  *ex-ante* higher-moment cousin (tested in study 503); this family reads the *realized* moments
  directly off the tape instead.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [803-realized-skewness-reversal](../../803-realized-skewness-reversal/) — the **third**
  moment (asymmetry / lottery right-tail) of a name's own returns, the *strong* headline of the
  same Amaya-et-al paper. Kurtosis is the **fourth** moment (symmetric fat-tailedness) — the
  paper's *weak* sibling signal, and this study's own axis.
- [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — the **second** moment
  (dispersion) of a name's residual returns. Kurtosis is scale-free (it divides out the
  variance), so it is explicitly **not** a volatility tilt in disguise.
- [365-lottery-max-effect](../../365-lottery-max-effect/) — the single **maximum daily return**
  (MAX) over a month, a one-number extreme order statistic. Realized kurtosis is the full fourth
  moment of the distribution, not the single extreme.

None of the siblings sort on the **realized fourth moment of a name's own daily returns** — the
weak Amaya-et-al kurtosis signal — which is this study's own axis.
