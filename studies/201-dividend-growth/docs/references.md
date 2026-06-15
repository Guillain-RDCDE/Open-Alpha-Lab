# References & literature map — Study 201 (Dividend-Growth)

## The claim under test

- **The folk strategy.** "Dividend growers" — companies with multi-year records of
  consecutive dividend raises — are marketed as superior long-run equity investments.
  The Dividend Aristocrats (S&P 500 names with 25+ years of raises) and Dividend Kings
  (50+ years) are institutionalized as ETFs and widely recommended by retail advisors.
  The investment thesis is: *consistent dividend growth signals financial quality, capital
  discipline, and durable earnings power, so growers should earn superior risk-adjusted
  forward returns vs high-yield non-growers and vs equal-weight baskets.*
  The standard product is ProShares NOBL ETF (launched 2013), which tracks the S&P 500
  Dividend Aristocrats Index.

## The academic literature on dividend policy and returns

- **Miller & Modigliani (1961).** *Dividend policy, growth, and the valuation of shares.*
  Journal of Business 34(4): 411–433. — The theoretical null: under perfect markets,
  dividend policy is irrelevant for firm value. This is the starting point that makes any
  "dividend growers outperform" claim require careful empirical justification.

- **Litzenberger & Ramaswamy (1979).** *The effect of personal taxes and dividends on
  capital asset prices.* Journal of Financial Economics 7(2): 163–195. — Early empirical
  evidence of a dividend yield effect; high-yield stocks had higher gross returns in a
  tax-penalized world. A motivation for the yield (not growth) angle.

- **Arnott & Asness (2003).** *Surprise! Higher dividends = higher earnings growth.*
  Financial Analysts Journal 59(1): 70–87. — Contrary to intuition, high dividend payout
  ratios historically *predicted higher* subsequent earnings growth. Suggests dividend
  policy signals optimism about future earnings, loosely supporting the grower thesis.

- **Fama & French (1992, 1993).** *The cross-section of expected stock returns.* Journal of
  Finance 47(2): 427–465. *Common risk factors in the returns on stocks and bonds.* Journal
  of Financial Economics 33(1): 3–56. — The size and value factors; dividend yield is
  correlated with the book-to-market (value) factor, which complicates isolating a pure
  dividend-growth effect from the value and quality tilts it naturally carries.

- **Novy-Marx (2013).** *The other side of value: the gross profitability premium.*
  Journal of Financial Economics 108(1): 1–28. — Gross profitability explains much of
  what appears to be a quality/dividend-growth premium. High-quality growers are also
  high-profitability firms; the factor returns are partly the same signal.

## The survivorship bias literature

- **Brown, Goetzmann & Ross (1995).** *Survival.* Journal of Finance 50(3): 853–873. —
  Classic demonstration that survivorship bias inflates measured average returns in
  portfolio studies. Directly applicable here: the 40-name universe is composed of
  survivors, so measured returns are upward-biased relative to what a real live strategy
  would have earned.

- **Elton, Gruber & Blake (1996).** *Survivorship bias and mutual fund performance.*
  Review of Financial Studies 9(4): 1097–1120. — Quantifies survivorship bias in
  fund studies; the bias is typically 0.5–2%/yr in annual return panels.

## Related factor research

- **Novy-Marx & Velikov (2016).** *A taxonomy of anomalies and their trading costs.*
  Review of Financial Studies 29(1): 104–147. — Systematic mapping of how many
  published anomalies survive realistic trading costs. Annual-rebalance strategies
  (like dividend growers) tend to have low direct costs but are exposed to the
  forward-looking universe selection bias identified in Brown et al. (1995).

- **McLean & Pontiff (2016).** *Does academic publication erode stock return predictability?*
  Journal of Finance 71(1): 5–32. — Post-publication return decay: anomalies lose roughly
  26% of their pre-publication alpha after being documented. The dividend-growth premium,
  packaged as Aristocrats ETFs since 2013, has had a decade of large-scale institutional
  implementation, consistent with at least partial arbitrage.

## Competing and complementary desk studies

- **[Study 88 — Dogs-of-the-Dow](../../88-dogs-of-the-dow/)**: the *yield* angle —
  whether the 10 highest-yielding Dow components beat the index. Complementary to Study
  201's *growth* angle. Arnott & Asness's result suggests yield and growth may point in
  different directions.

- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski's F-score — a composite
  nine-signal quality score from EDGAR fundamentals. Dividend consistency is a proxy for
  many of the same signals (profitability, leverage, cash flow), tested more directly.

- **[Study 122 — Gross-Profitability](../../122-gross-profitability/)**: the Novy-Marx
  quality factor. Tests whether the quality tilt embedded in dividend growth is better
  captured directly by gross profitability.

- **[Study 153 — Net-Operating-Assets](../../153-net-operating-assets/)**: the accruals
  factor — high accruals (aggressive accounting) firms underperform; dividend-paying firms
  tend to have lower accruals, providing another channel for the dividend-quality link.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3):
  703–708. Used in `strategy.summary` for the annual spread series.

- **yfinance total-return prices.** Auto-adjusted closes (`auto_adjust=True`) incorporate
  dividend reinvestment and split adjustments, providing a total-return time series.
  Method documented in the yfinance library (https://github.com/ranaroussi/yfinance).

- **Annual rebalance convention.** Dividends summed over the calendar year, streak computed
  through December 31, forward return = next calendar year — ensures no look-ahead.
