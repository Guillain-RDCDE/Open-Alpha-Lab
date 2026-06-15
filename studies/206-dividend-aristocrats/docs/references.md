# References & literature map — Study 206 (Dividend-Aristocrats)

## The claim under test

The S&P 500 Dividend Aristocrats Index tracks S&P 500 members that have raised their
dividend for at least 25 consecutive years.  ProShares launched NOBL (the ETF tracking
this index) in October 2013.  The investment thesis rests on two pillars: (1) a
**quality/dividend-growth premium** — companies with 25+ years of uninterrupted dividend
growth exhibit superior financial discipline, pricing power, and earnings stability that
should translate into better risk-adjusted returns; (2) a **crash shield** — such companies
cut dividends last (or never) in recessions, making the portfolio more defensive.

## Why the claim is plausible — the real effects it leans on

- **Quality factor evidence.** Asness, Frazzini & Pedersen (2019), *Quality Minus Junk*
  (Review of Accounting Studies), document a long-running quality factor premium in US and
  global equities — stocks with higher profitability, growth, safety, and payout ratios
  earn risk-adjusted returns above the market.  Dividend-growth consistency is a strong
  proxy for this quality composite.
- **Dividend-growth anomaly.** Litzenberger & Ramaswamy (1979), *The Effect of Personal
  Taxes and Dividends on Capital Asset Prices* (Journal of Financial Economics), and
  subsequent work by Arnott & Asness (2003), *Surprise! Higher Dividends = Higher Earnings
  Growth* (Financial Analysts Journal), argue that sustainable dividend growth signals
  management confidence in future cash flows — an information signal not fully in price.
- **Low-volatility anomaly.** Baker, Bradley & Wurgler (2011), *Benchmarks as Limits to
  Arbitrage: Understanding the Low-Volatility Anomaly* (Financial Analysts Journal), and
  Frazzini & Pedersen (2014), *Betting Against Beta* (Journal of Financial Economics),
  show that low-beta stocks have historically earned returns per unit of risk above what
  CAPM predicts.  Dividend Aristocrats carry a beta of ~0.81 (see results.md), so they
  straddle the low-vol and quality literatures simultaneously.

## Why the backtest disappoints — what the real tape shows

- **Factor crowding and valuation.** Harvey, Liu & Zhu (2016), *... and the Cross-Section
  of Expected Returns* (Review of Financial Studies), caution that factor premiums erode
  post-discovery.  Defensive/quality factor valuations ran expensive through 2017–2019,
  compressing future returns even if the underlying earnings quality was real.
- **Growth stocks dominate the benchmark.** The S&P 500 is increasingly concentrated in
  mega-cap technology companies.  None of them are Dividend Aristocrats (they either
  don't pay dividends or have not done so for 25 years).  When growth leads — as in
  2023–2024 — any quality/value tilt lags mechanically.  Arnott, Beck & Kalesnik (2016),
  *Timing "Smart Beta" Strategies? Of Course! Buy Low, Sell High!* (Financial Analysts
  Journal), show that factor tilts cycle and can underperform for multi-year stretches.
- **Beta-adjusted returns.** With beta = 0.81, NOBL's lower CAGR partly reflects lower
  market exposure — not a quality discount.  The OLS alpha net of beta is −0.96%/yr
  (t = −0.45): no detectable market-neutral premium.
- **Expense ratio headwind.** At 0.35%/yr vs 0.0945%/yr for SPY, NOBL must earn a gross
  alpha just to break even on fees — a hurdle it has not cleared.

## The crash-shield puzzle

- **Low dividend cut in recessions does not guarantee price resilience.** Dividends are
  paid from earnings, but stock prices are also driven by multiple expansion/compression
  and sector rotation.  During COVID-19 (2020), value/dividend stocks were hit hard
  relative to growth stocks, contradicting the defensive narrative.  During the 2022
  rate-shock bear market NOBL did outperform (+13.3% excess), consistent with value
  defensives, but recovered only partially.
- **Definition-of-done consistency.** Novy-Marx (2013), *The Other Side of Value: The
  Gross Profitability Premium* (Journal of Financial Economics), shows that quality as
  measured by profitability interacts with value: pure quality without a valuation
  anchor may not protect in "quality-at-any-price" episodes.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Sharpe bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **OLS alpha/beta.** Standard OLS decomposition, analogous to
  [`quantlab.stats.beta_decomposition`](../../../quantlab/stats.py).

## Data sources used here

- **Yahoo Finance (via yfinance)** — daily adjusted total-return closes for NOBL and SPY,
  2013-10-10 to 2026-06-15, `auto_adjust=True`.  NOBL was launched 2013-10-09; the full
  live history is used.  Adjusted closes fold in dividends automatically: critical for a
  dividend-focused comparison.

## Related desk studies

- **[Study 88 — Dogs-of-the-Dow](../../88-dogs-of-the-dow/)**: high-dividend-yield
  selection (distinct from dividend-growth selection tested here).
- **[Study 201 — Dividend-Growth](../../201-dividend-growth/)**: a factor approach to
  dividend-growth stocks distinct from the Aristocrats rule.
- **[Study 171 — Naive-1-Over-N](../../171-naive-1-over-n/)**: whether equal-weight beats
  optimised portfolios — the same "simple rule vs benchmark" structure as this study.
- **[Study 110 — Faber-Timing](../../110-faber-timing/)**: the SMA timing rule for SPY
  — another defensive strategy that protects on risk at a cost of return, mirroring NOBL's
  Sharpe/CAGR tradeoff here.
