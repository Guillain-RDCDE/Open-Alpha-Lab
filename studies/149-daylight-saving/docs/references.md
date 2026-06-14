# References & literature map — Study 149 (Daylight-Saving)

## The claim under test

- **Kamstra, M. J., Kramer, L. A. & Levi, M. D. (2000).** *Losing Sleep at the
  Market: The Daylight-Saving Anomaly.* American Economic Review 90(4), 1005-1011.
  The founding paper: uses S&P 500 and international indices (1967-1998) to argue
  that the Monday after a DST clock change earns significantly negative returns —
  averaging roughly −50 to −200 bps depending on the market — attributed to the
  sleep disruption that impairs investor decision-making and risk appetite.

## The contested replication

- **Pinegar, J. M. (2002).** *Losing Sleep at the Market: Comment.* American
  Economic Review 92(4), 1251-1256.  Shows that the KKL result is sensitive to
  the exact sample period and time-zone conventions; excluding a few influential
  outlier days eliminates statistical significance.  This is the canonical
  critique — and what makes this anomaly a textbook case of a fragile small-sample
  result.

- **Kamstra, M. J., Kramer, L. A. & Levi, M. D. (2002).** *Losing Sleep at the
  Market: Reply.* American Economic Review 92(4), 1257-1263.  Authors defend
  their methodology but acknowledge the robustness concerns.

## Related calendar-anomaly literature

- **Birru, J. (2018).** *Day of the Week and the Cross-Section of Returns.*
  Journal of Financial Economics 130(1), 182-214.  Documents Monday underperformance
  in the broad cross-section (the "Monday effect"); the DST anomaly is layered on
  top of this known pattern, which makes separating the two effects difficult.

- **French, K. R. (1980).** *Stock Returns and the Weekend Effect.*
  Journal of Financial Economics 8(1), 55-69.  The original Monday-effect paper:
  average Monday returns are significantly negative.  Kamstra et al. argue DST
  Mondays are even more negative, conditional on the day-of-week pattern.

- **Sullivan, R., Timmermann, A. & White, H. (1999).** *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap.* Journal of Finance 54(5), 1647-1691.
  The data-snooping hazard in calendar anomalies: with enough candidate rules the
  probability of finding a false positive is high.  A test based on 2 events/year
  over a short history is especially susceptible.

## The behavioral mechanism

- **Harrison, Y. & Horne, J. A. (1999).** *One night of sleep loss impairs
  innovative thinking and flexible decision making.* Organizational Behavior and
  Human Decision Processes 78(2), 128-145.  Laboratory evidence that even one hour
  of sleep loss measurably impairs decision quality and risk tolerance — the
  mechanism Kamstra et al. invoke.

- **Thaler, R. H. & Sunstein, C. R. (2008).** *Nudge: Improving Decisions About
  Health, Wealth, and Happiness.*  Princeton University Press.  Broader context
  for behavioral biases affecting financial decisions.

## Methodology lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica 55(3), 703-708.  The HAC t-stat used in
  [`strategy._hac_tstat`](../daylight_saving/strategy.py).

- **Welch, B. L. (1947).** *The generalization of 'Student's' problem when several
  different population variances are involved.*  Biometrika 34(1-2), 28-35.
  The Welch t-test for unequal variances between the DST and other-Monday arms.

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock
  Return Predictability?*  Journal of Finance 71(1), 5-32.  Post-publication decay
  of calendar anomalies; we test pre-2000 vs post-2000 to match the KKL sample cut.

## Data sources used here

- **Yahoo Finance (^GSPC)** via `yfinance`, daily close-to-close returns from
  1980-01-03 to 2026-06-12, cached as parquet in `_cache/gspc_daily.parquet`.
  The DST change dates are computed analytically from the statutory US DST rules;
  no external download is required for the calendar.
- **DST rule source:** US Public Law 109-58 (Energy Policy Act 2005, effective 2007)
  for the post-2007 schedule; 15 U.S.C. § 260a for the pre-2007 schedule.

## Related desk studies

- **[Study 55 — Summer-Lull](../../55-summer-lull/)**: the sell-in-May seasonal —
  another calendar anomaly with similar fragility concerns.
- **[Study 48 — Groundhog](../../48-groundhog/)**: a whimsical calendar event test
  that illustrates how isolated date-specific claims look in back-tests.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)** and
  **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)**: event-day calendar studies
  where the mechanism and the data are both stronger than the DST claim.
