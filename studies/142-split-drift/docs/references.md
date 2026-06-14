# References & literature map — Study 142 (Split-Drift)

## The claim under test

- **Ikenberry, Rankine & Stice (1996).** *What Do Stock Splits Really Signal?* — Journal
  of Financial and Quantitative Analysis 31(3), 357–375.  The canonical study: firms
  splitting their stock earn +7.9% abnormal return in the year following the **announcement**
  (measured vs a size-and-book-to-market matched control) and +12.1% over three years.  The
  interpretation: a stock split signals management confidence in future earnings; the market
  under-reacts to this signal, producing a drift.  This is the hypothesis our desk tests —
  in its *effective-date* form, which is strictly weaker (the market has 3–6 extra weeks to
  incorporate news between announcement and the ex-date we can observe in yfinance).

## Why the post-announcement anomaly is *almost* coherent

- **Signalling model.** Brennan & Copeland (1988), *Stock Splits, Stock Prices, and
  Transaction Costs* (Journal of Financial Economics), argue that splits are costly
  signals: by moving to a lower price range the firm invites higher transaction costs and
  microstructure noise, so only managers confident in future performance split.  This
  gives a credible route to a post-split positive surprise if the signal is underweighted.
- **Earnings drift pattern.** McNichols & Dravid (1990), *Stock Dividends, Stock Splits,
  and Signalling* (Journal of Finance), document above-normal earnings growth in the years
  following splits — consistent with the signalling hypothesis.
- **Under-reaction literature.** Bernard & Thomas (1989), *Post-Earnings-Announcement
  Drift* (Journal of Accounting Research), and Jegadeesh & Titman (1993), *Returns to
  Buying Winners and Selling Losers* (Journal of Finance), establish the broader class of
  market under-reaction to fundamental signals; the split drift fits the same mould.

## Evidence against the anomaly (and why our result is consistent)

- **Effective vs announcement date.** By the time the ex-date arrives, the market has
  typically known about the split for weeks.  Fama's (1991) efficient-market review
  (*Efficient Capital Markets: II*, Journal of Finance) would predict full incorporation
  by announcement; any remaining post-effective drift would be a weaker version already
  partially arbitraged.  Our result (no post-effective drift) is consistent with this.
- **Out-of-sample decay.** Byun & Rozeff (2003), *Long-Run Performance after Stock
  Splits: 1927 to 1996* (Journal of Finance), use a broader sample and find no significant
  long-run outperformance; the Ikenberry et al. finding may be in-sample.  Desai & Jain
  (1997), *Long-Run Common Stock Returns following Stock Splits and Reverse Splits*
  (Journal of Business), find post-split outperformance in the 1970s–1980s but not
  uniformly in later periods.
- **Survivorship.** Our basket uses current large-caps projected backwards — the stocks
  that survived and became index giants are systematically the ones that performed well
  *and* split repeatedly.  This introduces a selection bias that inflates the baseline
  returns but does not explain the *underperformance* of split dates vs non-split dates
  from the *same* basket.
- **Post-2000 split rarity.** Split frequency among large-caps collapsed after 2000.
  The 41 events our basket yields span 25 years at ~2 events/year.  Small-n makes it
  easy to miss a real effect; it also makes it easy to be fooled by noise.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  implemented in [`strategy._hac_tstat`](../split_drift/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Matched-period control.** Barber & Lyon (1997), *Detecting Long-Run Abnormal Stock
  Returns* (Journal of Finance), discuss the importance of matching on firm
  characteristics (or at minimum the same stock over non-event periods) to avoid
  spurious benchmark-driven results.  Our "same ticker, non-split window" control is a
  clean within-firm comparison.
- **Event-study inference.** Fama, Fisher, Jensen & Roll (1969), *The Adjustment of
  Stock Prices to New Information* (International Economic Review) — the foundational
  event-study framework we adapt for a post-effective-date window.
- **Survivorship bias.** Brown, Goetzmann, Ibbotson & Ross (1992), *Survivorship Bias
  in Performance Studies* (Review of Financial Studies) — named explicitly in our
  data stamp and results commentary.

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`) — split-adjusted auto-adjusted closes
  for a 30-stock large-cap basket, 2000-present.  Split dates sourced from the same
  `yfinance` `.splits` field, which returns the *effective* (ex-split) date, not the
  announcement date.  Window: 2000-01-03 → 2026-06-12; 41 events across 22 tickers.

## Related desk studies

- **[Study 34 — Aftershock](../../34-aftershock/)**: event-driven post-announcement drift
  in a different event class — the same "under-reaction" hypothesis, different catalyst.
- **[Study 82 — Witching-Hour](../../82-witching-hour/)**: another calendar-driven event
  (quarterly expiry) tested with matched-week controls — the same event-study machinery.
- **[Study 65 — Scorecard](../../65-scorecard/)** and
  **[Study 121 — Magic-Formula](../../121-magic-formula/)**: EDGAR-sourced fundamental
  signals where survivorship bias is named and bounded — the same survivorship discipline
  applied here to the split basket.
