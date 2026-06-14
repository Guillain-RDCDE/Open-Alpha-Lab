# References & literature map — Study 145 (Home-Bias)

## The claim under test

The received wisdom of portfolio construction: US investors concentrate ~50–70% of
equity holdings in the US market (their "home bias"), and the standard remedy is to
"diversify internationally" by adding developed (EFA/MSCI EAFE) and emerging market
(EEM/MSCI EM) equities. The expected benefit — articulated since Grubel (1968) and
formalized in Levy & Sarnat (1970) — is lower portfolio variance via the low
correlation between markets, improving the Sharpe ratio without sacrificing return.
We steelman this as: *a static 60/25/15 SPY/EFA/EEM blend, rebalanced annually,
earns a higher annualised Sharpe ratio than 100% SPY over the 2003-2026 sample.*

## Theoretical foundation

- **Grubel, H.G. (1968).** *Internationally Diversified Portfolios.* American Economic
  Review 58(5). The original case for international diversification based on
  correlation theory: lower cross-country correlations translate to mean-variance
  efficient frontiers that dominate domestic-only portfolios.
- **Levy, H. & Sarnat, M. (1970).** *International Diversification of Investment
  Portfolios.* American Economic Review 60(4). Quantified the benefit using
  pre-1970 data — a period of genuinely low international correlations.
- **Markowitz, H. (1952).** *Portfolio Selection.* Journal of Finance 7(1).
  The theoretical grounding: in a Gaussian world, diversification improves
  Sharpe only if new assets have correlations < 1.0 and competitive Sharpe.
- **Solnik, B. (1974).** *Why Not Diversify Internationally Rather Than Domestically?*
  Financial Analysts Journal. Extended the Markowitz argument to international
  markets and estimated large diversification gains from holding European equities.

## Why the claim is harder than it looks — rising correlations

- **Quinn, D.P. & Voth, H.-J. (2008).** *A Century of Global Equity Market
  Correlations.* American Economic Review 98(2). Documents the long-run trend:
  international equity correlations have risen substantially since the 1970s,
  driven by capital market integration and financial globalisation.
- **Longin, F. & Solnik, B. (2001).** *Extreme Correlation of International Equity
  Markets.* Journal of Finance 56(2). Correlation between equity markets is
  **not constant**: it rises dramatically during bear markets and crashes —
  exactly when diversification is most desired. The 2008 and 2020 crises saw
  cross-market correlations approach 0.95+.
- **Ammer, J. & Mei, J. (1996).** *Measuring International Economic Linkages with
  Stock Market Data.* Journal of Finance 51(5). Integration has strengthened the
  common factor in equity returns across developed markets.
- **De Santis, G. & Gerard, B. (1997).** *International Asset Pricing and Portfolio
  Diversification with Time-Varying Risk.* Journal of Finance 52(5). Shows that
  the gain from international diversification has declined as markets integrated.

## US equity outperformance (the practical killer)

- **Dimson, E., Marsh, P. & Staunton, M. (2002, updated annually).** *Triumph of
  the Optimists: 101 Years of Global Investment Returns.* Princeton University Press.
  Long-run evidence on cross-country equity returns; the US premium vs rest-of-world
  is real but not historically extreme. The post-2009 US outperformance is unusual.
- **Rouwenhorst, K.G. (1999).** *Local Return Factors and Turnover in Emerging Stock
  Markets.* Journal of Finance 54(4). Emerging market equity is high-vol; the
  Sharpe ratio on EEM is materially lower than SPY in 2003-2026.
- **Vanguard (2022).** *Global Equity Investing: The Benefits of Diversification and
  Sizing Your Allocation.* Vanguard Research. Acknowledges the post-2009 US
  outperformance while arguing for a forward-looking case for diversification.
  The desk's position: a historical test must report what happened, not what
  theory predicts will happen next.

## Home bias literature

- **French, K.R. & Poterba, J.M. (1991).** *Investor Diversification and International
  Equity Markets.* American Economic Review 81(2). The original quantification of the
  "home bias puzzle": US investors held ~94% domestic equity despite theory predicting
  international diversification.
- **Obstfeld, M. & Rogoff, K. (2000).** *The Six Major Puzzles in International
  Macroeconomics: Is There a Common Cause?* NBER Macroeconomics Annual. The home
  bias puzzle remains one of international finance's major open questions.
- **Sercu, P. & Vanpée, R. (2007).** *Home Bias in International Equity Portfolios:
  A Review.* Review of Finance. Survey of explanations: transaction costs,
  information asymmetry, implicit hedging of domestic human capital, familiarity
  bias. This study addresses none of these — we purely test whether the standard
  "diversify globally" advice improved outcomes for a US investor in 2003-2026.

## Inference and methodology

- **Politis, D.N. & Romano, J.P. (1994).** *The Stationary Bootstrap.* JASA 89(428).
  The circular block bootstrap for the Sharpe CI — [`strategy.sharpe_diff_bootstrap`].
- **Newey, W.K. & West, K.D. (1987).** *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica 55(3). HAC standard errors — [`strategy.mean_excess_hac`].
- **Lo, A.W. (2002).** *The Statistics of Sharpe Ratios.* Financial Analysts Journal.
  Annualisation and the distribution of the Sharpe estimator.

## Related desk studies

- **[Study 68 — All-Weather](../../68-all-weather/)**: risk parity (inverse-vol weights
  across SPY/IEF/GLD/DBC) vs equal-weight and 60/40. Same allocation-vs-US-only frame
  but with a multi-asset panel including bonds and gold.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: rebalancing bonus —
  does calendar rebalancing itself generate alpha?
- **[Study 31 — Trade-Winds](../../31-trade-winds/)**: global equity momentum across
  countries — a related international-equity study with a different angle.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)**: the "free rebalancing"
  bonus claim — does monthly vs annual rebalancing frequency materially change outcomes.
