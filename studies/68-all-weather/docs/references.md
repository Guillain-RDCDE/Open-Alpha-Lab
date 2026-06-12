# References & literature map — Study 68 (All-Weather)

## The idea and its source

- **Bridgewater Associates / Dalio, R.** — the "All Weather" portfolio: balance risk (not dollars)
  across assets that do well in different growth/inflation regimes.
- **Qian, E. (2005).** *Risk Parity Portfolios.* PanAgora — the formal inverse-volatility / equal
  risk-contribution construction.
- **Asness, C., Frazzini, A., & Pedersen, L. (2012).** *Leverage Aversion and Risk Parity.* Financial
  Analysts Journal — why risk parity works (levering low-vol assets) and why it needs leverage.

## The caveats

- **Anderson, Bianchi & Goldberg (2012).** *Will My Risk Parity Strategy Outperform?* FAJ — once you
  account for leverage cost and turnover, the historical edge shrinks.
- **Bhansali, V. (2011).** on risk parity's hidden dependence on the bond bull and on financing.
- **Open-Alpha-Lab** kin: [69 Safe-Haven](../../69-safe-haven/) (gold as a diversifier), [70
  Digital-Gold](../../70-digital-gold/) (bitcoin as a diversifier), and the cross-asset carry/diversification
  studies in the desk.

## Data

- **Yahoo! Finance** — SPY (US equities), IEF (7–10y Treasuries), GLD (gold), DBC (broad commodities),
  daily total returns, 2006–2026. Risk parity is built **unlevered** (inverse-vol, monthly rebalance);
  the leverage discussion is analytic. The offline synthetic world sets assets to a common Sharpe with a
  tunable vol spread (and a null where vols are equal).

*A cross-asset allocation companion to the diversifier studies [69 Safe-Haven](../../69-safe-haven/) and
[70 Digital-Gold](../../70-digital-gold/).*
