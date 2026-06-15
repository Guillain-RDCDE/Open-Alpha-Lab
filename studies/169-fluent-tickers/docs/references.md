# References & literature map — Study 169 (Fluent-Tickers)

## The claim under test

- **Alter, A. L. & Oppenheimer, D. M. (2006).** *Predicting short-term stock fluctuations
  by using processing fluency.* **PNAS**, 103(24), 9369–9372.
  The canonical source: using human-rated pronounceability scores for 89 NYSE listings
  (1990–2004), fluent-ticker IPOs earned ~2.1% more on their first trading day than
  disfluent ones; the effect persisted over about 12 months.  The proposed mechanism is
  **processing-fluency bias**: things that feel cognitively easy to process are judged as
  better, truer, and more valuable (the "What is easy is good" heuristic).

## The cognitive mechanism

- **Alter, A. L. & Oppenheimer, D. M. (2009).** *Uniting the tribes of fluency to form
  a metacognitive nation.* **Personality and Social Psychology Review**, 13(3), 219–235.
  Review of the fluency literature: fluent stimuli are more frequently judged as true,
  familiar, more aesthetically pleasing, and safer investments.  The stock-ticker result
  is one application of a general cognitive heuristic.

- **Reber, R., Winkielman, P. & Schwarz, N. (1998).** *Effects of perceptual fluency on
  affective judgments.* **Psychological Science**, 9(1), 45–48.
  Foundational paper on fluency-as-affect: stimuli that are easier to process trigger
  more positive affect, which is misattributed to the quality of the stimulus.

- **Laham, S. M., Koval, P. & Alter, A. L. (2012).** *The name-pronunciation effect:
  Why people like Mr. Smith more than Mr. Colquhoun.* **Journal of Experimental Social
  Psychology**, 48(3), 752–756.
  Related bias: easier-to-pronounce surnames are judged more positively; the same
  mechanism drives the ticker fluency hypothesis.

## Evidence against generalisation

- **Head, A., Smith, G. & Wilson, J. (2009).** *Would a stock by any other ticker smell
  as sweet?* **The Quarterly Review of Economics and Finance**, 49(2), 551–561.
  Replication attempt on a broader sample: the fluency-return link is weaker and
  less consistent than the original paper suggested; effect size shrinks substantially
  out of the original 1990–2004 window.

- **Jacowitz, K. E. & Kahneman, D. (1995).** *Measures of anchoring in estimation tasks.*
  **Personality and Social Psychology Bulletin**, 21(11), 1161–1166.
  General anchoring / accessibility literature: the ease-of-processing heuristic applies
  across many domains but is not always tradable.

## Small-sample and multiple-comparisons reckoning

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** *...and the cross-section of expected
  returns.* **Review of Financial Studies**, 29(1), 5–68.
  The hurdle for claiming a new factor/anomaly: with hundreds of empirical tests in the
  literature, a single t-stat of 2 is not sufficient — the bar should be 3.0 or higher
  to account for publication bias and data mining.  At n=89 IPOs, the original paper's
  significant result must be interpreted cautiously.

- **McLean, R. D. & Pontiff, J. (2016).** *Does academic research destroy stock return
  predictability?* **Journal of Finance**, 71(1), 5–32.
  Post-publication return decay: many documented anomalies shrink or disappear once
  they are known; the fluency effect, if real, would be arbitraged away by algorithmic
  liquidity providers who recognise that ticker names are public information.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** *A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.* **Econometrica**,
  55(3), 703–708.  The HAC long-run variance estimator behind `strategy.summarize`.

- **Lo, A. W. (2002).** *The statistics of Sharpe ratios.* **Financial Analysts Journal**,
  58(4), 36–52.  Annualised Sharpe standard error and t-stat; used in
  `quantlab.analytics.sharpe_with_se`.

## Survivorship-bias commentary

- **Banz, R. W. & Breen, W. J. (1986).** *Sample-dependent results using accounting and
  market data: Some evidence.* **Journal of Finance**, 41(4), 779–793.
  Classic demonstration that survivorship-biased backtests overstate returns; our
  S&P 500 survivor panel is explicitly named as an upper bound on any real edge.

## Related desk studies

- **[Study 04 — Social-Oracle](../../04-social-oracle/)**: retail investor attention
  bias (social media mentions) — the same family of behavioural anomalies driven by
  cognitive accessibility, tested with a similar null-first protocol.
- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: Japanese candlestick pattern
  recognition — another folklore/folk-signal teardown using the same random-baseline
  methodology.
- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: small-sample reckoning
  (presidential cycle effect) — directly comparable cautionary tale about small-n
  pattern identification.
