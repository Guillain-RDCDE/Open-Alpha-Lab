# References & literature map — Study 170 (Alphabetical-Bias)

## The claim under test

- **Jacobs & Hillert (2015).** "Alphabetic Bias, Investor Recognition, and Trading
  Behavior," *Journal of Financial Economics* 145(2), 2022 (working paper version
  circulated 2015 — sometimes cited as Hillert, Jacobs & Müller 2014 in SSRN form).
  The paper finds that stocks appearing higher in alphabetically sorted broker lists
  receive more analyst coverage, higher retail investor recognition, more trading
  activity and greater media attention — and that this attention effect is strongest
  for smaller, less-followed stocks. The *return* implication is more ambiguous in
  the paper: the authors note that attention may compress required returns (a
  familiarity premium) or have no net return effect at all if markets are
  semi-strong efficient. This study takes the return hypothesis seriously and tests
  it on the S&P 500 cross-section.

## Why the attention mechanism is coherent

- **Barber & Odean (2008).** "All That Glitters: The Effect of Attention and News
  on the Buying Behavior of Individual and Institutional Investors," *Review of
  Financial Studies* 21(2), 785–818. Documents that individual investors
  disproportionately buy stocks that have recently caught their attention (news,
  high volume, extreme returns). A stock sitting at the top of an alphabetical list
  is structurally more salient — this is the same channel.
- **Merton (1987).** "A Simple Model of Capital Market Equilibrium with Incomplete
  Information," *Journal of Finance* 42(3), 483–510. The classic model in which
  investor recognition affects required returns: a stock known to fewer investors
  commands a higher expected return as compensation for idiosyncratic risk not
  diversified away in the investors' held portfolios. Alphabetical salience is a
  frictionless proxy for recognition — investors "know" early-alphabet names better.
- **Seasholes & Wu (2007).** "Predictable Behavior, Profits, and Attention,"
  *Journal of Empirical Finance* 14(5), 590–610. Documents attention-driven buying
  at the open after a stock hits its daily upper price limit, consistent with retail
  attention being a distinct and patterned force.

## Why the return edge likely does not exist (or is priced away)

- **Fama (1970).** "Efficient Capital Markets: A Review of Theory and Empirical
  Work," *Journal of Finance* 25(2), 383–417. In a semi-strong efficient market,
  the alphabetical-position signal is entirely public and mechanical — there is no
  reason it should earn a persistent return. If the attention premium were real,
  institutions would arb it away by going long Z and short A.
- **McLean & Pontiff (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance* 71(1), 5–32. Documents that anomaly
  returns decay significantly after academic publication. Even if an alphabetical
  premium existed pre-publication, post-Jacobs & Hillert it should be weaker.
- **Multiple-comparisons danger.** Testing 26 individual letters plus the main
  group hypothesis (27 total) at a 5% nominal level is expected to produce ~1.35
  false positives. We apply a Bonferroni correction (|t| ≥ 2.897) and find no
  letters survive, consistent with the null of no letter-level return edge.
  See Harvey, Liu & Zhu (2016), "… and the Cross-Section of Expected Returns,"
  *Review of Financial Studies* 29(1), 5–68, on the multiple-comparisons crisis
  in the anomalies literature.

## Related bias: ticker fluency (different channel, similar family)

- **Alter & Oppenheimer (2006).** "Predicting Short-Term Stock Fluctuations by
  Using Processing Fluency," *Proceedings of the National Academy of Sciences*
  103(24), 9,369–9,372. Easy-to-pronounce ticker symbols (e.g. "KAR" vs "XPNSR")
  earned higher returns in the first days after IPO, attributable to processing
  fluency increasing familiarity and demand. This is a *distinct* channel from
  alphabetical position — it is about name quality, not list rank — but it is in
  the same family of attention/familiarity effects.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), "A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix," *Econometrica* 55(3), 703–708. Used in `strategy._hac_tstat` for the
  monthly spread inference.
- **Bonferroni correction.** Classic multiple-comparisons control: reject H_i if
  p_i < α/m where m is the number of hypotheses tested. Here m = 27
  (26 letters + 1 group), α = 0.05, giving p < 0.00185 or |t| ≥ 2.897.
  See Bonferroni (1936), "Teoria statistica delle classi e calcolo delle
  probabilità," *Pubblicazioni del R Istituto Superiore di Scienze Economiche
  e Commerciali di Firenze* 8, 3–62.
- **Survivorship bias disclosure.** Using the current S&P 500 constituent list
  rather than a historical point-in-time universe introduces name-survivorship bias:
  tickers that were included but later deleted (due to bankruptcy, merger or
  renaming) are absent. This is a well-known distortion in backtests based on
  index membership; see Elton, Gruber & Blake (1996), "Survivorship Bias and
  Mutual Fund Performance," *Review of Financial Studies* 9(4), 1,097–1,120,
  for the classic treatment (applied to funds, but the logic is identical).

## Data sources used here

- **Yahoo Finance daily prices** (via `yfinance`), adjusted close, for S&P 500
  names. Resampled to monthly (last-day-of-month close). Date range 2000-01-01
  to 2026-06-15. Per-study cache in `studies/170-alphabetical-bias/_cache/`.
- **EDGAR / quantlab universe.** `quantlab.universe.sp500_symbols` (current-
  constituents list, survivorship-biased); fallback to `_edgar_Revenues.parquet`
  column names if quantlab is unavailable.

## Related desk studies

- **Study 48 — Groundhog**: seasonal/calendar signal with small-n reckoning --
  same family of "did anyone actually check this?" ideas.
- **Study 76 — Rice-Paper**: candlestick patterns with Bonferroni correction --
  same multiple-comparisons discipline as the per-letter scan here.
- **Study 83 — Half-Life**: tiny-n teardown; the n=tiny reckoning that applies
  whenever the honest effective sample is smaller than the advertised one.
- **Study 04 — Social-Oracle**: retail attention (Reddit mentions) and stock
  returns -- the closest cousin in the desk's attention-effects family.
