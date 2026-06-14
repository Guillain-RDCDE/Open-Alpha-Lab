# References — Study 157 (Kelly-Sizing)

## Foundational Theory

1. **Kelly, J.L. (1956).** "A New Interpretation of Information Rate." *Bell System Technical
   Journal*, 35(4), 917–926. The original paper deriving the growth-optimal betting fraction
   as `f* = edge / odds` for binary bets; generalises to `f* = mu/sigma^2` for continuous
   Gaussian returns. Kelly showed this fraction maximises the long-run geometric growth rate
   (the expected log wealth).

2. **Breiman, L. (1961).** "Optimal Gambling Systems for Favorable Games." *Proceedings of the
   Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 65–78. Proved
   that the Kelly strategy asymptotically maximises the geometric growth rate among all
   non-anticipating strategies, and that it outperforms any other strategy with probability 1
   in the long run.

3. **Thorp, E.O. (1971).** "Portfolio Choice and the Kelly Criterion." *Proceedings of the
   Business and Economics Section of the American Statistical Association*, 215–224. First
   systematic application of Kelly to securities markets, by the author of *Beat the Dealer*
   and founder of Princeton/Newport Partners.

## The Drawdown Problem and Fractional Kelly

4. **MacLean, L.C., Thorp, E.O., and Ziemba, W.T. (2010).** "Good and Bad Properties of the
   Kelly Criterion." *SIAM Journal on Financial Mathematics*, 1(1), 752–768. The canonical
   modern treatment: Kelly maximises long-run growth but produces extreme drawdowns (>50%
   is common, >90% is possible under estimation error). Half-Kelly halves growth, quarters
   variance — the standard practitioner compromise.

5. **Vince, R. (1990).** *Portfolio Management Formulas*. Wiley. Popularised "Optimal-f"
   (the Kelly fraction applied to trading), emphasised the catastrophic drawdowns of
   full-Kelly in practice.

## Estimation Risk — The Core Problem

6. **Chopra, V.K. and Ziemba, W.T. (1993).** "The Effect of Errors in Means, Variances, and
   Covariances on Optimal Portfolio Choice." *Journal of Portfolio Management*, 19(2), 6–11.
   Errors in the mean return estimate are 11x more damaging to portfolio optimality than
   errors in variance. Since Kelly is `f* = mu/sigma^2`, imprecise mu is fatal.

7. **Hakansson, N.H. (1971).** "Capital Growth and the Mean-Variance Approach to Portfolio
   Selection." *Journal of Financial and Quantitative Analysis*, 6(1), 517–557. Early analysis
   of the gap between theoretical Kelly optimality and practical estimation-contaminated Kelly.

## Log-Utility and the Growth-Optimality Framework

8. **Merton, R.C. (1969).** "Lifetime Portfolio Selection Under Uncertainty: The Continuous-Time
   Case." *Review of Economics and Statistics*, 51(3), 247–257. Derives the optimal consumption
   and investment policy for an investor with constant relative risk aversion; the log-utility
   special case yields the Kelly fraction.

9. **Cover, T.M. and Thomas, J.A. (1991).** *Elements of Information Theory*. Wiley. Chapter 6
   shows the deep connection between the Kelly criterion and the channel capacity / mutual
   information framework — Kelly growth rate = entropy rate of the bet sequence.

## Practical Applications and Post-Publication Evidence

10. **Thorp, E.O. (2006).** "The Kelly Criterion in Blackjack, Sports Betting, and the Stock
    Market." In *Handbook of Asset and Liability Management* (eds. Zenios and Ziemba). Elsevier.
    Reviews three decades of practical experience; concludes that fractional Kelly (typically
    one-quarter to one-half) is essential in real markets where parameter uncertainty is large.

11. **Lo, A.W. and MacKinlay, A.C. (1990).** "When Are Contrarian Profits Due to Stock Market
    Overreaction?" *Review of Financial Studies*, 3(2), 175–205. Illustrates the magnitude of
    estimation error in short-horizon return parameters — directly relevant to the Kelly
    fraction's unreliability at short windows.
