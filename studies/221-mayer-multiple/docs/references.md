# References & literature map — Study 221 (Mayer-Multiple)

## The claim under test

- **Trace Mayer (2017-2018).** "The Mayer Multiple." Popularised by Trace Mayer (Bitcoin
  advocate and podcast host) in 2017–2018, the Mayer Multiple is defined as price /
  200-day simple moving average. A threshold of 2.4 was proposed as historically associated
  with overvaluation; buying below 1.0 was framed as accumulating at a discount. The
  metric was disseminated via the *Bitcoin Knowledge Podcast*, CoinMetrics, and numerous
  retail crypto media outlets. No peer-reviewed research underpins the specific threshold
  choices; they were derived from visual inspection of BTC's cycle history through 2017
  and have not been validated on subsequent data.
- **Glassnode (2021).** "Mayer Multiple." Glassnode on-chain metrics documentation.
  Standard reference for the metric's definition and historical distribution. Available at
  https://academy.glassnode.com/metrics/market/mayer-multiple. The distribution data
  confirms that MM < 1.0 was historically rare during bull markets, which is the source
  of the "cheap" framing.

## The simple moving average as a valuation anchor

- **Faber, M. T. (2007).** "A Quantitative Approach to Tactical Asset Allocation."
  *The Journal of Wealth Management*, 9(4), 69–79. The seminal tactical-allocation paper
  that popularised the 200-day moving average rule (buy when price > 10-month SMA, sell
  otherwise) across traditional asset classes. The Mayer Multiple inverts this logic for
  Bitcoin, labelling below-SMA as "cheap" rather than a momentum-negative regime.
- **Kilgallen, T. (2012).** "Testing the Simple Moving Average across Commodities, Global
  Stock Indices, and Currencies." *The Journal of Wealth Management*, 15(1), 82–100.
  Broad evidence that simple MA crossovers have trend-following (not contrarian) properties
  across most asset classes — consistent with our finding that MM > 1 predicts higher
  30-day forward returns for BTC.
- **Han, Y., Zhou, G., & Zhu, Y. (2016).** "Taming the Factor Zoo: A Test of New Factors."
  *Journal of Finance* — the broader statistical testing framework applied to signal
  validation; the Mayer Multiple's threshold choice is a case of implicit data mining on
  a short BTC history.

## Bitcoin as a momentum asset

- **Liu, Y., & Tsyvinski, A. (2021).** "Risks and Returns of Cryptocurrency."
  *Review of Financial Studies*, 34(6), 2689–2727. Documents that cryptocurrency returns
  exhibit strong momentum (time-series and cross-sectional). Consistent with our decile
  analysis showing higher Mayer Multiples predict higher subsequent 30-day returns rather
  than lower (the opposite of the mean-reversion claim).
- **Lintilhac, P., & Tourin, A. (2017).** "Model-based Pairs Trading in the Bitcoin
  Markets." *Quantitative Finance*, 17(5), 703–716. Illustrates the regime-dependence of
  BTC returns; mean-reversion strategies require precise regime identification that the
  Mayer Multiple does not provide.

## The 200-day SMA as a trend filter (not a valuation signal)

- **Antonacci, G. (2012).** "Risk Premia Harvesting Through Dual Momentum." *Portfolio
  Management Consultants*. Available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2042750.
  Documents the risk-reduction role of the 200-day SMA as a trend filter (exit below,
  stay above) — the opposite of the Mayer Multiple's "buy below, sell above" framing.
- **Barroso, P., & Santa-Clara, P. (2015).** "Momentum Has Its Moments." *Journal of
  Financial Economics*, 116(1), 111–120. Crash-risk management in momentum strategies;
  relevant to why mean-reversion strategies on trend assets underperform.

## Statistical methodology

- **Harvey, C., Liu, Y., & Zhu, H. (2016).** "...and the Cross-Section of Expected
  Returns." *Review of Financial Studies*, 29(1), 5–68. The canonical reference on the
  multiple-testing problem in financial research. The Mayer Multiple's threshold (2.4)
  was chosen by inspecting the same historical data used to evaluate it; the paper's
  framework implies the effective t-stat bar is much higher than 2.0.
- **Newey, W. K., & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703–708. HAC standard error estimation used in `strategy.summarize` and
  `band_tstat`; handles serial correlation in daily log-returns and overlapping 30-day
  windows.
- **Pardo, R. (2008).** *The Evaluation and Optimization of Trading Strategies*. Wiley.
  Standard practitioner reference for the dangers of in-sample threshold optimisation.

## Related desk studies

- **[Study 84 — Moon-Math](../../84-moon-math/)**: BTC Stock-to-Flow model — another
  valuation-style BTC signal, same spurious-regression diagnosis.
- **[Study 103 — Turtle-Trading](../../103-turtle-trading/)**: systematic trend-following
  using breakouts — the momentum side of the same coin.
- **[Study 174 — Bitcoin-Rainbow](../../174-bitcoin-rainbow/)**: the Bitcoin Rainbow Chart
  (log-time regression) — closely related valuation-band framework, same look-ahead
  anatomy exposed.
- **[Study 106 — Faber-Timing](../../106-faber-timing/)**: Faber's 10-month SMA timing
  rule — the trend-following use of moving averages, which the Mayer Multiple inverts.
