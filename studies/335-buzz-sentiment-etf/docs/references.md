# References & literature map — Study 335 (Buzz-Sentiment-ETF)

## The product under test

- **VanEck Social Sentiment ETF (BUZZ).** Launched 2021-03-04, tracking the
  *BUZZ NextGen AI US Sentiment Leaders Index*. The marketed thesis: a natural-
  language-processing / "AI" engine reads social media, news and other online
  sources to score the bullishness around US large caps, and the fund holds the
  75 most positively-discussed names, reweighted monthly. The implicit claim a
  buyer is sold is that crowd sentiment, read by machine, **picks
  market-beating winners**. (VanEck product page; index methodology, BUZZ
  Holdings DXJ / Periscope Capital.) The ETF drew attention when Dave Portnoy
  (Barstool Sports) promoted it at launch, spiking inflows.
- **Why this is distinct from the raw-signal studies.** This desk has already
  torn down the *underlying signals*: [Study 254 — WSB-Mentions](../../254-wsb-mentions/)
  (does r/WallStreetBets mention-count predict the move?) and
  [Study 256 — Twitter-Mood](../../256-twitter-mood/) (does aggregate Twitter
  mood lead the market?). Both stamped None/Mirage. Study 335 asks the **product**
  question instead: even granting that the signal is whatever it is, does the
  *tradeable wrapper a retail buyer can actually click* deliver alpha over the
  market? It is an ETF-vs-benchmark performance-attribution study, not a
  lead-lag signal study.

## The right way to grade an "active" ETF — alpha, not return

- **Jensen's alpha / CAPM.** Jensen (1968), *The Performance of Mutual Funds in
  the Period 1945–1964* (Journal of Finance) — the intercept of a fund's excess
  return regressed on the market's excess return is the canonical measure of
  manager skill. A high-beta basket out-returning the index in a bull market is
  beta, not alpha; only the intercept counts. This is exactly
  [`strategy.capm_alpha`](../buzz_sentiment_etf/strategy.py).
- **Persistence (and non-persistence) of fund outperformance.** Carhart (1997),
  *On Persistence in Mutual Fund Performance* (Journal of Finance) — most apparent
  outperformance is factor exposure or luck, not durable skill. Fama & French
  (2010), *Luck versus Skill in the Cross-Section of Mutual Fund Returns* — in
  aggregate, active funds do not beat their benchmarks net of costs.
- **Thematic / "story" ETFs underperform.** Ben-David, Franzoni, Kim & Moussawi
  (2023), *Competition for Attention in the ETF Space* (Review of Financial
  Studies) — specialised, narrative-driven ETFs launched near peak hype tend to
  *underperform* the broad market afterward; the attention that sells them is a
  contrarian tell. BUZZ, a sentiment-of-the-crowd fund sold on social-media buzz,
  is almost a caricature of this pattern.

## The sentiment literature it leans on (and its limits)

- **Investor attention & returns.** Da, Engelberg & Gao (2011), *In Search of
  Attention* (Journal of Finance) — Google search volume as attention; a transient
  price-pressure effect that reverses, not durable alpha. Tetlock (2007),
  *Giving Content to Investor Sentiment* (Journal of Finance) — media pessimism
  predicts short-horizon, reverting pressure. The honest reading: sentiment moves
  prices *transiently*; harvesting it net of costs in a monthly-rebalanced 75-name
  basket is a much taller order than the headline suggests.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica) — the HAC standard errors on the CAPM intercept and on
  the active return ([`strategy.hac_tstat`, `strategy.capm_alpha`](../buzz_sentiment_etf/strategy.py)).
- **Block bootstrap.** Politis & Romano (1992/1994), the stationary/circular block
  bootstrap — resampling contiguous blocks so the contemporaneous beta and the
  return autocorrelation survive the resample
  ([`strategy.block_bootstrap_alpha_ci`](../buzz_sentiment_etf/strategy.py)).
- **Sharpe / information ratio.** Sharpe (1966, 1994); Grinold & Kahn,
  *Active Portfolio Management* — the information ratio of the active (fund minus
  benchmark) return is the standard skill yardstick for an active product.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted (dividends +
  splits) close — a total-return-style series for BUZZ and SPY. Common window
  2021-03-04 → 2026-05-29 (BUZZ's full live history; the partial June 2026 month
  is dropped). All headline numbers are pinned with an as-of date and content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core
  and the test-suite run on the deterministic
  [`data.synthetic_pair`](../buzz_sentiment_etf/data.py) generator, never the
  network.

## Related desk studies

- **[Study 254 — WSB-Mentions](../../254-wsb-mentions/)** and
  **[Study 256 — Twitter-Mood](../../256-twitter-mood/)**: the raw social-sentiment
  signals this fund packages — both None/Mirage. Study 335 is the tradeable-wrapper
  cousin: same family (sentiment/crowds), same verdict, but the unit of analysis
  is a real ETF a buyer can hold, graded on alpha vs SPY.
- **[Study 252 — Search-Trends](../../252-google-trends/)** and
  **[Study 255 — Fear-Greed](../../255-fear-greed-index/)**: the attention/sentiment
  timing family. Sentiment is a transient price-pressure story, not a durable edge —
  consistent with BUZZ's missing alpha.
