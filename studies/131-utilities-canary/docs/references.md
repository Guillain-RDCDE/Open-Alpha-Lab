# References — Study 131 (Utilities-Canary)

## The folk claim

The "utilities canary" is a recurring theme in retail and professional market commentary.
The intuition: utilities (XLU) are rate-sensitive, low-beta, and "defensive" — investors
rotate into them when they anticipate risk-off environments. If that rotation is visible
in relative strength before the broad market rolls over, XLU rising vs SPY could be a
leading indicator of drawdowns.

## Academic and practitioner foundations

**Fama, E. F., & French, K. R. (1993).** Common risk factors in the returns on stocks and bonds.
*Journal of Financial Economics*, 33(1), 3–56.
The benchmark for understanding sector return differences as risk factor exposures rather than
directional signals.

**Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013).** Value and momentum everywhere.
*Journal of Finance*, 68(3), 929–985.
Establishes that momentum effects are present in many asset classes including sectors, but at
multi-month horizons — not the short-term relative-strength reversals the canary uses.

**Baker, M., & Wurgler, J. (2007).** Investor sentiment in the stock market.
*Journal of Economic Perspectives*, 21(2), 129–152.
Frames defensive-sector outperformance as a sentiment indicator; the relationship between
sentiment and subsequent returns is imprecise at actionable horizons.

**Hong, H., Torous, W., & Valkanov, R. (2007).** Do industries lead stock markets?
*Journal of Financial Economics*, 83(2), 367–396.
Formal test of sector lead-lag relationships. Some sectors contain leading information for
the aggregate market, but utilities are among the weaker predictors.

**Goetzmann, W., & Shiller, R. (2014).** Cyclically adjusted price-earnings ratios as
predictors of returns. Variants use macro-regime variables including sector RS signals as
business-cycle indicators. The documented predictive power operates at 5-10 year horizons,
not at the 1-21 day horizons the canary is typically quoted for.

**Rapach, D. E., Strauss, J. K., & Zhou, G. (2010).** Out-of-sample equity premium prediction:
Combination forecasts and links to the real economy.
*Review of Financial Studies*, 23(2), 821–862.
Demonstrates that macro-linked variables can predict market returns out-of-sample at monthly
horizons, but the effect sizes are small (R2 of ~1%) and require explicit statistical testing —
exactly what the canary folk signal skips.

## Data sources

- **Yahoo Finance** (via `yfinance`): XLU and SPY daily adjusted closes 1998-12-22 onward.
  XLU (Utilities Select Sector SPDR Fund) launched 1998-12-22; data available from inception.
  Auto-adjusted prices account for dividends and splits.
- Sector SPDR inception date: [https://www.sectorspdr.com/sectorspdr/sector/xlu](https://www.sectorspdr.com/sectorspdr/sector/xlu)

## Why the result is WEAK / MIRAGE

The folk signal conflates two effects:

1. **Coincident correlation.** XLU outperforms during risk-off periods (flight to quality,
   rate cuts in recessions). The *correlation* with concurrent SPY weakness is genuine.

2. **Leading indicator.** A much harder claim: that the RS rotation *precedes* SPY weakness
   enough to be acted on profitably. This study finds t = 1.58 at the 21-day horizon — below
   the bar of 2 required for a REAL stamp.

The regime-Sharpe split (0.70 calm vs 0.22 alert) reflects the well-known fact that
high-volatility environments (where XLU outperforms) have lower risk-adjusted SPY returns.
This is a risk-factor statement, not an alpha statement.
