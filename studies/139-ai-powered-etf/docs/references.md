# References & literature map — Study 139 (AI-Powered-ETF)

## The claim under test

- **The fund's own marketing.** EquBot LLC and ETF Managers Group (2017), *AIEQ: AI
  Powered Equity ETF prospectus and marketing materials* — "harnessing the power of
  artificial intelligence to consistently outperform the market." The fund uses IBM
  Watson to analyse fundamentals, sentiment, and market data, making it the canonical
  real-world test of whether off-the-shelf AI translates into investable alpha.
- **Media coverage at launch.** *Bloomberg* (Oct 2017), *CNBC* (Oct 2017), *Forbes*
  (Oct 2017) — widespread coverage positioning AIEQ as a landmark: "the first AI-run
  ETF," "Watson picks your stocks," promising a new paradigm for active management.

## The prior academic evidence — why the claim is hard to sustain

- **Fama (1970).** *Efficient Capital Markets: A Review of Theory and Empirical Work*,
  Journal of Finance 25(2). The null hypothesis: in a semi-strong efficient market,
  no public information (including AI-processed fundamentals and news) delivers
  persistent alpha. Active management must beat this bar.
- **Grossman & Stiglitz (1980).** *On the Impossibility of Informationally Efficient
  Markets*, American Economic Review 70(3). AI requires an edge in information
  *processing* speed or depth; if many players use similar AI tools, the edge
  arbitrages away quickly — consistent with AIEQ's long-run underperformance.
- **Fama & French (2010).** *Luck versus Skill in the Cross-Section of Mutual Fund
  Returns*, Journal of Finance 65(5). Only a tiny fraction of active funds generate
  statistically significant alpha after fees; the expected outcome for any new active
  fund (AI or not) is negative alpha net of costs.
- **McLean & Pontiff (2016).** *Does Academic Research Destroy Stock Return
  Predictability?*, Journal of Finance 71(1). Published anomalies decay post-
  publication as arbitrageurs exploit them; AI picking from known factor signals
  faces the same headwind.

## The AI-in-finance evidence specifically

- **Gu, Kelly & Xiu (2020).** *Empirical Asset Pricing via Machine Learning*, Review
  of Financial Studies 33(5). ML methods can improve return predictions in-sample and
  modestly out-of-sample — but the evidence requires careful walk-forward validation
  and the out-of-sample gains are modest after transactions costs.
- **Lopez de Prado (2018).** *Advances in Financial Machine Learning*, Wiley. Documents
  the overfitting risks in financial ML; most in-sample gains do not survive live
  trading — a cautionary frame for any AI fund.
- **DeMiguel, Garlappi & Uppal (2009).** *Optimal Versus Naive Diversification*, Review
  of Financial Studies 22(5). The 1/N rule beats many sophisticated allocation models
  out of sample — the baseline for any AI-powered selection strategy to beat.

## Expense ratio and cost drag

- **Sharpe (1991).** *The Arithmetic of Active Management*, Financial Analysts Journal
  47(1). Before costs, active management is a zero-sum game. After costs (AIEQ 0.75%
  vs SPY 0.0945%), the average active fund *must* underperform. The AI wrapper does
  not change this arithmetic.
- **French (2008).** *The Cost of Active Investing*, Journal of Finance 63(4). Estimates
  the annual cost of active management at ~0.67% of total stock-market value in the US;
  the added 0.65pp expense drag of AIEQ vs SPY maps directly onto this literature.

## Method lineage (the desk's shared engine)

- **Jensen (1968).** *The Performance of Mutual Funds in the Period 1945-1964*, Journal
  of Finance 23(2). Jensen's alpha as the CAPM intercept — the standard measure used here.
- **Newey & West (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3). HAC t-stat via
  Bartlett kernel, used in [`strategy._hac_ols_tstat_alpha`](../ai_powered_etf/strategy.py).
- **Lo (2002).** *The Statistics of Sharpe Ratios*, Financial Analysts Journal 58(4).
  Uncertainty on Sharpe ratios from finite samples — motivates the bootstrap CI.
- **Politis & Romano (1994).** *The Stationary Bootstrap*, JASA 89. Circular block
  bootstrap for Sharpe CIs, via [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Related desk studies

- **[Study 12 — Paper-Prophet](../../12-paper-prophet/)**: prophet/ML forecasting of
  prices — the same ML-in-finance scepticism applied to a pure forecasting task.
- **[Study 39 — Black-Box](../../39-black-box/)**: RandomForest stock picker, strict
  walk-forward — the desk's own ML-alpha attempt, with the same honest verdict.
- **[Study 65 — Scorecard](../../65-scorecard/)**: fundamental factor cross-section
  on EDGAR data, showing the limits of systematic fundamental selection.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt's mechanical
  value+quality screener — a simpler version of what AIEQ claims to automate.
