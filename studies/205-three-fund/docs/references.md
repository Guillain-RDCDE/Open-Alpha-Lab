# References & literature map — Study 205 (Three-Fund)

## The claim under test

The **Bogleheads three-fund portfolio**: allocate broadly across US stocks, international
stocks, and bonds (e.g. VTI / VXUS / BND or SPY / EFA / BND) at roughly global market-cap
weights. The hypothesis is that this low-cost, globally-diversified passive portfolio
achieves *better risk-adjusted returns* than simpler alternatives (100% US equities, 60/40)
through international diversification and bond smoothing, without requiring active management.

The steelman: mean-variance theory predicts a globally-diversified portfolio lies on or
near the efficient frontier, while a US-only portfolio is an *undiversified* concentrated bet
on one country. The bond sleeve reduces volatility and drawdown via negative equity/bond
correlation. All three benefits are theoretically grounded.

## The foundational literature on diversification and the global portfolio

- **Markowitz, H.M. (1952).** *Portfolio Selection.* Journal of Finance 7(1), 77–91.
  The foundational result: diversification reduces portfolio variance without sacrificing
  expected return when assets are imperfectly correlated. The three-fund portfolio is
  precisely this principle applied globally at minimal cost.
- **Sharpe, W.F. (1964).** *Capital Asset Prices: A Theory of Market Equilibrium under
  Conditions of Risk.* Journal of Finance 19(3), 425–442. Under CAPM, the efficient
  portfolio is the *world market portfolio* — a global market-cap-weighted allocation that
  the three-fund approximates.
- **Solnik, B. (1974).** *Why Not Diversify Internationally Rather than Domestically?*
  Financial Analysts Journal 30(4), 48–54. The seminal paper on international diversification:
  correlations between national markets are lower than within-market correlations, so
  international diversification expands the efficient frontier. The three-fund's rationale
  stems directly from this result.
- **Dimson, E., Marsh, P., & Staunton, M. (2002).** *Triumph of the Optimists.* Princeton
  University Press. Long-run evidence (1900–2000) that diversification across countries
  reduces risk even when individual countries fail; the US has been an exceptional winner
  and survivorship bias inflates the impression of US stock market returns.

## The home-bias debate — why people don't go global, and whether they should

- **French, K.R., & Poterba, J.M. (1991).** *Investor Diversification and International
  Equity Markets.* American Economic Review 81(2), 222–226. Documents the massive
  *home bias* — investors hold far more domestic equity than global market weights imply
  they should. The three-fund is precisely the corrective.
- **Coval, J.D., & Moskowitz, T.J. (1999).** *Home Bias at Home: Local Equity Preference
  in Domestic Portfolios.* Journal of Finance 54(6), 2045–2073. Extends home bias to
  even local geographic concentration within markets.
- **Vanguard Research (2012–2023).** Multiple white papers on international diversification
  and the role of foreign equity in a long-term portfolio. Vanguard's own recommendation
  is 40% international equity, close to the three-fund's 30–33% allocation. Available at
  Vanguard.com/research.
- **Philips, C.B., Kinniry, F.M., & Schlanger, T. (2012).** *The Case for Index-Fund Investing.*
  Vanguard Investment Counseling & Research. The cost argument for passive investing: even a
  small return premium from international diversification is erased if the vehicle is
  expensive; low-cost ETFs (VTI expense ratio 0.03%, VXUS 0.07%) are the study's assumed
  vehicle.

## The international underperformance debate (the real-world finding)

- **Fama, E.F., & French, K.R. (2012).** *Size, Value, and Momentum in International Stock
  Returns.* Journal of Financial Economics 105(3), 457–472. Value and size premia exist
  internationally, but they are smaller and less reliable than in the US, providing
  partial theoretical backing for US-heavy allocation.
- **MSCI (2023).** *ACWI IMI — All Country World Investable Market Index.* As of end-2023,
  the US represents ≈ 63% of global market cap; international (ex-US) ≈ 37%. The three-fund's
  30% VXUS weight is a mild US overweight relative to global cap-weights.
- **Sharpe, W.F. (1991).** *The Arithmetic of Active Management.* Financial Analysts Journal
  47(1), 7–9. Before costs, the average dollar in international markets earns the
  international market return. Under-performance by international equities (VXUS vs VTI,
  2011–2026) is simply the realised gap between market returns, not a structural failure.

## The 60/40 baseline literature

- **Swensen, D.F. (2000).** *Pioneering Portfolio Management.* Free Press. The classic
  institutional asset allocation text; endorses diversified multi-asset portfolios. The
  Yale endowment model is a more complex version of the three-fund idea.
- **Asness, C.S., Frazzini, A., & Pedersen, L.H. (2012).** *Leverage Aversion and Risk
  Parity.* Financial Analysts Journal 68(1), 47–59. The case for risk-parity weighting
  (equal risk contribution) rather than market-cap weighting; the traditional 60/40 is
  equity-dominated despite appearing "balanced."
- **Arnott, R., & Bernstein, P. (2002).** *What Risk Premium Is "Normal"?* Financial
  Analysts Journal 58(2), 64–85. Historical equity risk premia may be lower going forward;
  high equity allocations carry more risk than their historical track records suggest.

## Inference method lineage

- **Newey, W.K., & West, K.D. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3), 703–708. HAC t-stat
  on annual return differences; used here for the pairwise comparisons.
- **Politis, D.N., & Romano, J.P. (1994).** *The Stationary Bootstrap.* Journal of the
  American Statistical Association 89(428), 1303–1313. Circular block-bootstrap Sharpe CI,
  via [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Lo, A.W. (2002).** *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4),
  36–52. Annualised Sharpe with delta-method SE.
- **Miller, R.G. (1981).** *Simultaneous Statistical Inference.* 2nd ed., Springer. Bonferroni
  multiple-comparisons correction across 3 pairwise hypotheses.

## Related desk studies

- **[Study 68 — All-Weather](../../68-all-weather/)**: the Bridgewater risk-parity allocation —
  same family (passive multi-asset), different weighting philosophy.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: rebalancing timing and frequency —
  how much does when you rebalance matter?
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: Harry Browne's
  25/25/25/25 four-asset allocation — another canonical lazy alternative.
- **[Study 171 — Naive-1-Over-N](../../171-naive-1-over-n/)**: DeMiguel et al. on 1/N equal
  weight vs Markowitz optimisers — the optimisation question one level up from this study.
- **[Study 145 — Home-Bias](../../145-home-bias/)**: the companion study on international
  equity directly; the international underperformance finding here is a partial confirmation
  of that study's scope.
