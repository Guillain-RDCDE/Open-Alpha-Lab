# References & literature map — Study 96 (New-Year-Pop)

## The claim under test

The **January effect** (specifically the *January small-cap effect*): small-capitalisation
stocks systematically outperform large caps in January, the textbook explanation being
tax-loss selling in December (investors dump losers — disproportionately small, beaten-down
names — for the tax write-off, then the buying pressure releases in early January) plus
institutional **window-dressing** at year-end. The strong, sold-at-full-strength version is
that you should **tilt to small caps for the turn of the year** to harvest a reliable
seasonal premium.

## The foundational literature

- **Keim, D. B. (1983).** *Size-related anomalies and stock return seasonality: Further
  empirical evidence.* Journal of Financial Economics 12(1), 13–32. The canonical
  documentation that the size premium is heavily concentrated in January (and within
  January, in the first few trading days).
- **Reinganum, M. R. (1983).** *The anomalous stock market behavior of small firms in
  January: Empirical tests for tax-loss selling effects.* Journal of Financial Economics
  12(1), 89–104. The companion paper tying the January small-firm return to tax-loss
  selling.
- **Roll, R. (1983).** *Vas ist das? The turn-of-the-year effect and the return premia of
  small firms.* Journal of Portfolio Management — the turn-of-the-year framing.
- **Rozeff, M. S. & Kinney, W. R. (1976).** *Capital market seasonality: The case of stock
  returns.* JFE — earlier evidence of January seasonality in returns.

## The post-publication decay

- **Schwert, G. W. (2003).** *Anomalies and market efficiency.* Handbook of the Economics
  of Finance — documents that many seasonal/size anomalies, including the January effect,
  **weakened or disappeared after they were published**, consistent with arbitrage and
  awareness eroding the edge.
- **Haug, M. & Hirschey, M. (2006).** *The January effect.* Financial Analysts Journal —
  re-examines the effect and its persistence/decay across sub-periods.
- The practitioner consensus (and our own tape, 1990→2026) is that the *small-cap* January
  effect is much weaker or absent in recent decades than in the 1960s–early-1980s sample
  Keim and Reinganum studied.

## Why it is likely to fail *as stated* on a modern tape

- **Publication decay.** Once a free calendar edge is widely known, December tax-selling
  and January buying get anticipated and arbitraged, flattening the seasonal.
- **The strong era predates retail data.** Yahoo's ^RUT history begins in 1990; the
  decisive 1970s evidence is outside any free dataset, so a modern reproduction tests the
  *already-decayed* regime — and finds little.
- **Costs and the index-vs-stock gap.** The original effect was strongest in the smallest,
  least-liquid micro-caps; a tradable small-cap *index* (Russell 2000 / IWM) dilutes
  exactly the corner where the seasonal lived.

## Data sources used

- **^RUT (Russell 2000)** and **^GSPC (S&P 500)**, daily closes via `quantlab.data`
  (Yahoo Finance), **price-only** (`mode="split_only"`) — Yahoo carries no dividend-adjusted
  series for these spot indices, so the long-sample spread is honestly labelled price-only.
- **IWM (iShares Russell 2000)** and **SPY (SPDR S&P 500)**, daily closes via
  `quantlab.data`, **total-return** (`mode="total_return"`) — the tradable, dividend-
  inclusive pair, from IWM's 2000 inception.
- Returns are resampled to month-end; in-progress final months are dropped before stats.

## Method lineage

- **Newey–West HAC standard errors** for the January-dummy contrast on an autocorrelated
  monthly spread: Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica.
- **Wilson score interval** for the January hit-rate (small>large): Wilson (1927), *Probable
  Inference, the Law of Succession, and Statistical Inference*, JASA — better small-sample
  coverage than the normal approximation.
- **Welch two-sample *t*** for the pre/post-2000 difference in January spread (the decay
  test): Welch (1947), *The generalization of "Student's" problem…*, Biometrika.

## Related desk studies

- [Study 44 — Growth-Spurt](../../44-growth-spurt/) — the *unconditional* size/growth
  premium on tradable large caps. This study is the **January seasonal in the size spread**,
  not the size premium itself.
- [Study 80 — Cold-Open](../../80-cold-open/) — the "as January goes, so goes the year"
  barometer. This study is about small-vs-large *within* January, not January as a predictor
  of the rest of the year.
- [Study 91 — Death-Cross](../../91-death-cross/) — the HAC-*t* + matched-control inference
  pattern this study reuses.
