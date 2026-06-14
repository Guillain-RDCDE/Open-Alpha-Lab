# References & literature map — Study 146 (Country-Momentum)

## The claim under test

- **Cross-country momentum.** Each month, rank single-country equity ETFs by their trailing
  12-1 month total return and go long the top-K (typically 3–5 countries). The idea is that
  country-level momentum — documented by Asness, Liew & Stevens (1997) and consistent with
  the broader Jegadeesh & Titman (1993) anomaly — should persist at the country level long
  enough for a monthly rebalancing strategy to harvest it. The rule is explicitly in strategy
  §3.9 of Kakushadze & Serur (2018), *151 Trading Strategies* (Springer).

## Foundational cross-sectional momentum literature

- **Jegadeesh, N. & Titman, S. (1993).** *Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency.* Journal of Finance 48(1), 65–91. The canonical
  paper: US stocks ranked on 3–12 month past returns continue outperforming for up to 12
  months — "the momentum anomaly." All cross-country momentum work descends from this finding.
- **Asness, C. S., Liew, J. M. & Stevens, R. L. (1997).** *Parallels Between the Cross-
  Sectional Predictability of Stock and Country Returns.* Journal of Portfolio Management
  23(3), 79–87. The direct parent of this study: documents that the Jegadeesh–Titman momentum
  premium extends to country equity indices, with winner countries outperforming loser countries
  at 6–12 month horizons.
- **Rouwenhorst, K. G. (1998).** *International Momentum Strategies.* Journal of Finance 53(1),
  267–284. Confirms cross-sectional stock momentum in 12 European countries; the premium is
  similar in magnitude to the US, suggesting it is not a US data artefact.

## Why the premium might be real

- **Underreaction / slow information diffusion.** Barberis, Shleifer & Vishny (1998), *A Model
  of Investor Sentiment*, Journal of Financial Economics — investors underreact to good news,
  leading to gradual price adjustment; country-level macroeconomic news diffuses even more slowly
  than stock-specific news.
- **Herding and trend-chasing.** Daniel, Hirshleifer & Subrahmanyam (1998), *Investor
  Psychology and Security Market Under- and Overreactions*, Journal of Finance — overconfidence
  and self-attribution bias amplify initial price moves, producing momentum.

## Why the premium is thin in the real data and post-2010

- **Post-publication decay.** McLean, R. D. & Pontiff, J. (2016). *Does Academic Research
  Destroy Stock Return Predictability?* Journal of Finance 71(1), 5–32. Anomaly returns drop
  by ~32% post-publication on average; country momentum was documented in the late 1990s and
  has been shrinking since.
- **Momentum crashes.** Barroso, P. & Santa-Clara, P. (2015). *Momentum Has Its Moments.*
  Journal of Financial Economics 116(1), 111–120. Momentum strategies are subject to severe,
  sudden crashes (our data: max drawdown −62.3%). A naive long-only top-K country rotation
  inherits all equity beta and all momentum-crash risk.
- **Cost sensitivity.** Novy-Marx, R. & Velikov, M. (2016). *A Taxonomy of Anomalies and Their
  Trading Costs.* Review of Financial Studies 29(1), 104–147. Even at monthly rebalance
  frequency, ETF transaction costs (bid-ask 3–10 bps round-trip) are sufficient to erode the
  thin active return above a simple equal-weight basket.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica — used
  via [`strategy.summarize`](../country_momentum/strategy.py) for the per-period return mean.
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **12-1 momentum score.** Skip-1 month convention follows the standard in the literature
  (Jegadeesh & Titman 1993) to avoid contamination from the 1-month reversal effect (De Bondt
  & Thaler 1985, *Does the Stock Market Overreact?*, Journal of Finance).

## Related desk studies

- **[Study 24 — Stampede](../../24-stampede/)**: the same 12-1 momentum strategy applied to
  individual US stocks within the S&P 500 — the cross-sectional (stock-level) version of this
  idea, with the same methodology and a similar verdict (WEAK/FRAGILE).
- **[Study 31 — Trade-Winds](../../31-trade-winds/)**: time-series momentum applied to a
  diversified futures basket — a related but structurally different momentum signal (trend of
  an asset vs itself, not vs peers).
- **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)** and **[Study 67 — Fed-Drift](../../67-fed-drift/)**:
  other global-macro rotational strategies, for comparison on what a real macro signal looks like.
