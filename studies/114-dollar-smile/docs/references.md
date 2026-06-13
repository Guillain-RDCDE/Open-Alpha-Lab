# References — Study 114 (Dollar-Smile)

## Primary claim

The "dollar-smile" or "strong dollar is bad for stocks" narrative is a long-standing
piece of macro conventional wisdom: a rising US Dollar Index implies tighter financial
conditions globally, higher dollar-denominated debt burdens for EM borrowers, and
repatriation-led drag on US multinational earnings — all supposedly bad for equities,
and especially for EM.

## Source of the claim

1. **Gundlach, J.** — "The Dollar Smile" framework popularised in macro commentary
   (DoubleLine Capital, 2014-2018); the idea that dollar strength and equity weakness
   co-move.
2. **BIS Quarterly Review** (various years) — the "dollar as global risk factor" theme:
   Bruno & Shin (2015), "Capital Flows and the Risk-Taking Channel of Monetary Policy",
   *Journal of Monetary Economics*, 71, 119-132. Documents the dollar's role in global
   financial conditions.
3. **Avdjiev, S., Du, W., Koch, C., & Shin, H. S.** (2019). "The dollar, bank leverage,
   and deviations from covered interest parity." *American Economic Review: Insights*, 1(2),
   193-208. Shows the dollar as a barometer of global risk appetite (contemporaneous).

## Contemporaneous vs predictive distinction

4. **Goyal, A., & Welch, I.** (2008). "A comprehensive look at the empirical performance
   of equity premium prediction." *Review of Financial Studies*, 21(4), 1455-1508.
   The standard out-of-sample R² benchmark; negative OOS R² means the model is worse
   than the historical mean. This study applies it directly.
5. **Campbell, J. Y., & Thompson, S. B.** (2008). "Predicting excess stock returns
   out of sample: Can anything beat the historical average?" *Review of Financial
   Studies*, 21(4), 1509-1531. Related OOS evaluation framework.

## Method lineage

6. **Newey, W. K., & West, K. D.** (1987). "A simple, positive semi-definite,
   heteroskedasticity and autocorrelation consistent covariance matrix." *Econometrica*,
   55(3), 703-708. The HAC standard error estimator used for all t-statistics.
7. **Diebold, F. X., & Mariano, R. S.** (1995). "Comparing predictive accuracy."
   *Journal of Business & Economic Statistics*, 13(3), 253-263. The DM test for
   comparing forecast accuracy.
8. **Koijen, R. S. J., & Van Nieuwerburgh, S.** (2011). "Predictability of returns and
   cash flows." *Annual Review of Financial Economics*, 3, 467-491. Survey of the limits
   of predictability; the null of no predictability is the right benchmark.

## Dollar and equity / EM relationships

9. **Lustig, H., & Verdelhan, A.** (2007). "The cross section of foreign currency risk
   premia and consumption growth risk." *American Economic Review*, 97(1), 89-117.
   Dollar risk premia and the dollar as a global risk factor.
10. **Ilzetzki, E., Reinhart, C. M., & Rogoff, K. S.** (2019). "Exchange arrangements
    entering the 21st century: Which anchor will hold?" *Quarterly Journal of Economics*,
    134(2), 599-646. The dollar's special role in global finance.

## Data sources

11. **ICE US Dollar Index (DX-Y.NYB)** — via Yahoo Finance. The trade-weighted nominal
    US Dollar Index (DXY), 2004-01-01 to 2026-06-12.
12. **SPDR S&P 500 ETF (SPY)** — via Yahoo Finance, adjusted close. 2004-01-01 onward.
13. **iShares MSCI Emerging Markets ETF (EEM)** — via Yahoo Finance, adjusted close.
    EEM inception 2003-04-14; we use 2004-01-01 for a clean overlap with the DXY.

## Related desk studies

- **Study 85 (Dr-Copper)** — the closest analogue: a cross-asset ratio (copper/gold) as
  an equity and yield forecaster. Same methodology (IS, OOS R², DM test), same verdict:
  contemporaneous link confirmed, predictive link absent.
- **Study 16 (Storm-Shy)** — macro regime timing; dollar-strength as a risk-off regime
  signal (related but distinct from the price-level regression tested here).
- **Study 68 (All-Weather)** — asset-allocation across regimes; the dollar's role in
  multi-asset diversification.
