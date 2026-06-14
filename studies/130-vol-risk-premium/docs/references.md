# References -- Study 130 (Vol-Risk-Premium)

## Core literature

**Bakshi, G. & Kapadia, N. (2003).** Delta-hedged gains and the negative market volatility
risk premium. *Review of Financial Studies*, 16(2), 527-566.
Introduces the variance risk premium from a delta-hedged option replication perspective;
shows that the negative VRP is a robust finding across maturities and strikes.

**Carr, P. & Wu, L. (2009).** Variance risk premiums. *Review of Financial Studies*, 22(3),
1311-1341. The definitive derivation of the VRP using model-free variance swaps (the
realised vs implied variance spread). Shows the VRP is persistently negative — i.e., IV
exceeds RV — across equity indices, currencies, and rates.

**Bollerslev, T., Tauchen, G. & Zhou, H. (2009).** Expected stock returns and variance risk
premia. *Review of Financial Studies*, 22(11), 4463-4492. Establishes that the VRP
(measured as VIX^2 minus a model-based realised variance forecast) predicts stock returns
at the 1-3 month horizon with R^2 around 5-8%, outperforming many other predictors.

**Bekaert, G. & Hoerova, M. (2014).** The VIX, the variance premium and stock market
volatility. *Journal of Econometrics*, 183(2), 181-192. Decomposes VIX^2 into the
conditional variance and a variance risk premium component; finds the VRP is the main
driver of return predictability at intermediate horizons.

**Drechsler, I. & Yaron, A. (2011).** What's vol got to do with it. *Review of Financial
Studies*, 24(1), 1-45. A general equilibrium model where the VRP arises endogenously from
jump risk in consumption growth; derives testable implications for the VRP-return
relationship.

## Methodology and related work

**Newey, W.K. & West, K.D. (1987).** A simple, positive semi-definite, heteroskedasticity
and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. The
Bartlett-kernel HAC estimator used for all t-statistics in this study.

**Martin, I. (2017).** What is the expected return on the market? *Quarterly Journal of
Economics*, 132(1), 367-433. Uses the SVIX measure (lower bound on the VRP) to derive an
equity premium lower bound; shows the VRP has a tight theoretical link to expected returns.

**Londono, J.M. (2015).** The variance risk premium around the world. *International Finance
Discussion Papers* 1254, Board of Governors of the Federal Reserve System. Documents the
VRP internationally; confirms it is a global phenomenon.

**Eraker, B. (2004).** Do stock prices and volatility jump? Reconciling evidence from spot
and option prices. *Journal of Finance*, 59(3), 1367-1403. Disentangles jump and diffusion
components of the VRP; key for understanding the negative skew of short-vol payoffs.

**Politis, D.N. & Romano, J.P. (1994).** The stationary bootstrap. *Journal of the American
Statistical Association*, 89(428), 1303-1313. Block-bootstrap method used here to compute
confidence intervals on the Sharpe ratio.

## Practical and adjacent studies

This study is distinct from Study 63 (Free-Fall), which shorts the VIX ETP VIXY — an
indirect, roll-cost-laden exposure. Here we measure the raw IV-RV spread from the index
itself (^VIX and SPY), with no roll cost or ETP tracking-error confound.

See also Study 111 (VIX-Term-Structure) for the related VIX/VIX3M slope signal (contango
vs backwardation), and Study 86 (Tail-Radar) for the left-tail event risk that the
short-vol payoff is exposed to.
