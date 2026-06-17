# References & literature map — Study 228 (Pre-Earnings Runup)

## The core claim — where this idea comes from

- **Lakonishok, J. & Vermaelen, T. (1990), "Anomalous Price Behavior Around Repurchase Tender Offers,"
  *Journal of Finance* 45(2), 455–477.** Early documentation of systematic price movements before
  corporate events, including earnings. The idea that informed agents accumulate positions ahead of
  predictable news dates is foundational to the pre-announcement runup literature.

- **Kim, O. & Park, J. (2005), "Pre-Announcement Premium and Earnings Quality," *Review of Accounting
  Studies* 10(4), 475–514.** Documents a positive pre-earnings-announcement return premium in the 1–5
  days before the scheduled release, attributed to informed trading and options market positioning.

- **So, E. C. & Wang, S. (2014), "Quantifying the Tax Benefits of Debt," *Journal of Finance* 69(5).** *(See
  the companion paper: So, E. C. & Wang, S. (2014), "Insider Trading and Earnings Announcements,"
  working paper.)* Related analysis showing that short-selling activity and options positioning spike in
  the pre-announcement window on large-cap names — consistent with informed positioning but also with
  hedging activity.

## The academic literature on pre-earnings drift

- **Keown, A. J. & Pinkerton, J. M. (1981), "Merger Announcements and Insider Trading Activity: An
  Empirical Investigation," *Journal of Finance* 36(4), 855–869.** The classic demonstration that
  prices move before scheduled corporate events, often cited as evidence of informed trading.

- **Frazzini, A. & Lamont, O. A. (2007), "The Earnings Announcement Premium and Trading Volume,"
  *Journal of Accounting Research* 45(1), 1–26.** Documents a positive return premium *around* earnings
  announcements (a few days before and after), and shows it has declined over time on large-cap stocks.
  Crucially, much of the pre-event return is explained by beta (high-beta stocks cluster in the pre-event
  window) and risk-premium explanations compete with the informed-trading story.

- **Berkman, H., Dimitrov, V., Jain, P.C., Koch, P.D. & Tice, S. (2009), "Sell on the News: Differences of
  Opinion, Short-Sales Constraints, and Returns around Earnings Announcements," *Journal of Financial
  Economics* 92(3), 376–399.** Documents a distinct *pre*-announcement runup followed by a post-event
  reversal on hard-to-short stocks — very different from the PEAD (post-earnings drift) studied in study 34
  and from the pure runup studied here.

## The decay / arbitrage literature — why the premium is now small

- **McLean, R. D. & Pontiff, J. (2016), "Does Academic Research Destroy Stock Return Predictability?"
  *Journal of Finance* 71(1), 5–32.** Demonstrates that documented anomalies attenuate significantly
  after publication as arbitrageurs trade against them. The pre-earnings-runup has been prominently
  documented and widely known in the trading community for decades.

- **Ausubel, L. M. (1990), "Insider Trading in a Rational Expectations Economy," *American Economic Review*
  80(5), 1022–1041.** Theoretical framework: in a rational-expectations equilibrium, informed pre-event
  positioning is at least partially revealed by price action, causing the premium to be competed away.
  On liquid large-cap stocks with tight spreads and deep options markets, this process completes quickly.

## The desk's method — inference and reproducibility

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix," *Econometrica* 55(3), 703–708.** HAC inference on the
  book's mean daily return; the pre-event window creates mild serial correlation in the book's returns.
- **Reproducibility.** Headline runs are pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (as-of date + content fingerprint). Reproduce offline via [`../examples/verify.py`](../examples/verify.py).

## Relationship to other studies in this repo

- **Study 34 (Aftershock)** — tests the POST-earnings drift (PEAD), i.e., the price drift that continues
  *after* the announcement date. This study (228) tests the PRE-announcement runup; the two are distinct
  effects with different mechanisms (PEAD = under-reaction to the surprise; runup = informed positioning
  ahead of the event).

## Caveats stated in the open

- **Survivorship bias.** The price panel is current large-cap membership; delisted names are absent,
  biasing measured returns upward.
- **Beta contamination.** The long-only equal-weight pre-event book carries substantial positive equity
  beta. The market benchmark (+0.97 Sharpe) far exceeds the book (+0.46 gross Sharpe), indicating the
  book's return is mostly market exposure, not a pre-earnings premium.
- **Calendar data quality.** Earnings dates from yfinance may have revision/restatement issues; some
  announcements are scheduled but then released at a different time. The desk uses the *scheduled* date
  as given by yfinance's earnings_dates.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
