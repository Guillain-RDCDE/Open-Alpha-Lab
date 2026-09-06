# Sources & literature map — Study 979 (The Prior Is the Portfolio)

## The model

- **Black, F. & Litterman, R. (1992), "Global Portfolio Optimization", *Financial Analysts
  Journal* 48(5), 28-43.** The original. Note how much of it is about the *prior* — the
  equilibrium portfolio — rather than about views.
- **He, G. & Litterman, R. (1999), "The Intuition Behind Black-Litterman Model Portfolios",
  Goldman Sachs Investment Management Research.** The paper that gives the Omega convention used
  here, and the clearest statement of the zero-view case.
- **Idzorek, T. (2007), "A Step-by-Step Guide to the Black-Litterman Model", in *Forecasting
  Expected Returns in the Financial Markets*.** The practitioner reference, including the
  confidence-based specification of Omega that this study does not use but many desks do.

## The parameters nobody derives

- **Walters, J. (2014), "The Black-Litterman Model in Detail" (SSRN 1314585).** The best survey
  of the competing conventions for `tau` and `Omega`, and of how much they disagree.
- **Meucci, A. (2010), "The Black-Litterman Approach: Original Model and Extensions", in *The
  Encyclopedia of Quantitative Finance*.** Puts the model in its Bayesian setting, which makes
  the zero-view identity obvious rather than surprising.
- **Satchell, S. & Scowcroft, A. (2000), "A Demystification of the Black-Litterman Model",
  *Journal of Asset Management* 1(2), 138-150.** Where `tau` comes from, and why it is not
  identifiable from data.

## Why the prior does the work

- **Grinold, R. C. & Kahn, R. N. (1999), *Active Portfolio Management*, 2nd ed.** Reverse
  optimisation and the risk-aversion convention `delta = 2.5` used here.
- **Jagannathan, R. & Ma, T. (2003), *Journal of Finance* 58(4), 1651-1683.** Constraints as
  shrinkage — the same phenomenon as a strong prior, arriving by another route.
- **DeMiguel, V., Garlappi, L. & Uppal, R. (2009), *Review of Financial Studies* 22(5),
  1915-1953.** 1/N: the prior that is hardest to beat, and one of the three tested here.

## Neighbours on this desk

**975-covariance-shrinkage**, **976-hierarchical-risk-parity**, **977-max-diversification**,
**978-resampled-frontier**, **171-naive-1-over-n**, **902-multi-factor-composite**,
**518-time-series-momentum**.
