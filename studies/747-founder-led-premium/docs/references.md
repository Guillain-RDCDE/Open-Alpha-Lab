# References & literature map — Study 747 (Founder-Led-Premium)

## The claim under test

- **The founder thesis (finance).** Fahlenbrach, Rüdiger (2009), *Founder-CEOs, Investment
  Decisions, and Stock Market Performance* (Journal of Financial and Quantitative Analysis
  44:2). The canonical citation: an equal-weighted portfolio of S&P 500 **founder-CEO** firms
  earns a positive, statistically significant **abnormal return** (~+8.3%/yr on a
  four-factor benchmark over 1993–2002), and founder-CEOs invest more in R&D and make more
  focused acquisitions. This is the paper the folklore leans on, and the strongest form of
  the claim we steelman.
- **The founder thesis (management).** Zook, Chris & Allen, James (2016), *The Founder's
  Mentality: How to Overcome the Predictable Crises of Growth* (Bain & Company / HBR Press) —
  the popularisation ("insurgent mission, owner's mindset, obsession with the front line")
  that turned the finding into a management-consulting franchise and a VC talking point.
- **Corroborating strands.** Adams, Almeida & Ferreira (2009), *Understanding the
  relationship between founder-CEOs and firm performance* (Journal of Empirical Finance);
  Villalonga & Amit (2006), *How do family ownership, control and management affect firm
  value?* (JFE) — family/founder control and valuation. These report *conditional* and
  *heterogeneous* effects, not a clean tradable premium — the nuance the folklore drops.
- **The believers' one-liner.** *"Founders have skin in the game and a long horizon, so
  founder-led companies outperform"* — a claim of a harvestable, repeatable **characteristic
  premium**. We test whether a founder-minus-professional long/short has a market-model alpha
  that clears *t* = 2 and survives the obvious confounds.

## Why the naive test is a trap — the confounds

- **Alpha vs beta (risk premium you were always paid for).** Sharpe (1964) / Lintner (1965)
  CAPM and Jensen, Michael (1968), *The Performance of Mutual Funds in the Period 1945–1964*
  (Journal of Finance) — the market-model **alpha** as the abnormal return net of market
  exposure. Founder firms skew young, tech, high-beta; in a bull market that beta alone beats
  low-beta incumbents, and beta is not a premium (an index fund sells it for basis points).
  A four-/five-factor version (Fama & French 1993, 2015; Carhart 1997) would additionally
  net out size, value, profitability and momentum — beyond this study's single-factor scope,
  and flagged in *Going further*.
- **Survivorship / selection.** Brown, Goetzmann, Ibbotson & Ross (1992), *Survivorship Bias
  in Performance Studies* (Review of Financial Studies) — a backward-looking "winners" panel
  manufactures a premium out of nothing. A basket of the founder firms one *remembers in
  2024* is exactly this: the founder-run flame-outs (Theranos never listed; WeWork; a
  graveyard of de-SPACs) cannot enter, so the sample is biased *for* the claim.
- **Data-snooping / concentration.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns* (RFS), and Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
  (Journal of Portfolio Management) — with a handful of names, one or two extreme winners
  (NVDA, TSLA) can carry an entire "factor"; the honest test is whether it survives dropping
  the top name (a jackknife) and a label permutation (a placebo).

## The event-/portfolio-study & inference method (shared engine)

- **Long/short characteristic sort + market-model alpha.** The standard cross-sectional
  factor construction: form two baskets on a characteristic, take the spread, regress on the
  market, read the intercept (Fama & French 1993; MacKinlay 1997, *Event Studies in
  Economics and Finance*, JEL, for the abnormal-return framing).
- **Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*** (Econometrica) — the HAC standard error on
  the alpha (Bartlett kernel, rule-of-thumb lag `floor(4·(n/100)^(2/9))`). Monthly strategy
  returns are mildly autocorrelated and heteroskedastic; the naive *t* overstates
  significance.
- **Randomisation / placebo inference.** Fisher's permutation logic; Efron & Tibshirani
  (1993), *An Introduction to the Bootstrap* — the label-shuffle null that isolates the
  founder *tag* from basket-membership luck.

## Method lineage (this study's code)

- **Basket construction + long/short.** [`strategy.basket_returns`](../founder_led_premium/strategy.py)
  and [`strategy.long_short`](../founder_led_premium/strategy.py) — equal-weighted monthly
  baskets (a delisted name drops from that month's average) and their difference.
- **CAPM alpha + Newey-West HAC *t*.** [`strategy.capm_alpha`](../founder_led_premium/strategy.py)
  (HAC covariance of the OLS coefficients) and [`strategy.hac_mean_t`](../founder_led_premium/strategy.py)
  — the Signal-axis tests; mirrors [`quantlab/analytics.py`](../../../quantlab/analytics.py)
  `mean_tstat_hac`.
- **Jackknife + placebo.** [`strategy.jackknife_alpha`](../founder_led_premium/strategy.py)
  (drop-one concentration test) and [`strategy.placebo_alpha_dist`](../founder_led_premium/strategy.py)
  (random-label null).
- **Costs + borrow.** [`strategy.net_of_costs`](../founder_led_premium/strategy.py) — one-way
  turnover both legs + a short-borrow charge; gross vs net.
- **Deterministic synthetic control.**
  [`data.synthetic_baskets`](../founder_led_premium/data.py) plants a known founder alpha; the
  offline core runs with no network. The control confirms the engine recovers a planted edge
  **and** does not fabricate one (nor mistake beta for alpha) when the truth is zero.

## Data sources used here

- **Hardcoded baskets** (`founder_led_premium.data.FOUNDER` / `PRO`): a transparent,
  hindsight-labelled stand-in for a survivorship-clean founder panel (Fahlenbrach's
  hand-collected S&P 500 founder-CEO set is not redistributable). Membership frozen at the
  2016-01 formation; the founder tag is the believers' own framing and is deliberately
  editable so a reader can re-tag and re-run.
- **yfinance** monthly adjusted closes for each basket name + SPY, cached under `_cache/`
  (adjusted = total-return proxy). `SQ` (reticker to XYZ) and `FIT` (delisted, acquired by
  Google 2021) drop out for lack of a current series — a literal illustration of the
  survivorship named on the Signal axis. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 391 — CEO-Turnover](../391-ceo-turnover/)**: the same abnormal-return + HAC +
  synthetic-control machinery on *replacing* a CEO — another small-sample, hindsight-shaped
  corporate-leadership mirage (there the trap is timing; here it is survivorship + beta).
- **[Study 389 — Name-Change-Effect](../389-name-change-effect/)**: another hardcoded,
  transparent, selection-on-anecdotes corporate cross-section where the loud examples are the
  ones that delisted.
