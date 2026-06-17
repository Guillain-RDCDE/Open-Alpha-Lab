# References & literature map — Study 289 (Diwali-Muhurat)

## The claim under test

Indian market folklore holds that the **Muhurat trading session** — a special ~1-hour
evening session conducted on the NSE and BSE on the day of Diwali (Laxmi Pujan) — is an
auspicious window in which buying brings prosperity for the year ahead. Brokerages and
the financial press recycle the claim each autumn, and many investors place a token
"shagun" (good-luck) trade during the session. The testable version: do Indian equities
earn an abnormal return *around* the Muhurat session relative to an ordinary window?

## Why this is folklore, not a factor

- **No mechanism.** A culturally-chosen evening session cannot move corporate cash flows.
  Any return pattern must come from sentiment/flow, which is exactly the kind of effect
  that is tiny, regime-dependent, and arbitraged away once it is widely advertised.
- **Tiny n.** The NSE/BSE have held Muhurat sessions for decades, but a tradeable foreign
  proxy (INDA, the iShares MSCI India ETF) only exists from 2012 — roughly 14 events. With
  ~1.3% daily volatility, a sub-percent seasonal cannot be resolved at |t| ≥ 2.
- **Proxy mismatch.** The Muhurat session trades in INR on Indian exchanges in an evening
  window; no US-listed instrument trades then. A foreign investor's closest tradeable
  approximation is the US session *following* Diwali — which is what we measure, and which
  strips out the very ritual the folklore is about.

## The honest-baseline problem (the methodological spine)

A calendar window that earns a positive return is unremarkable if equities drift up on
*every* window. The correct null is the **unconditional same-length forward window mean**,
estimated as a block permutation over all possible window start dates — not zero, and not
a coin. This mirrors the base-rate trap that inflated the Super Bowl Indicator's apparent
accuracy ([Study 158 — Super-Bowl](../../158-super-bowl/)).

## Academic literature on calendar / holiday anomalies and small-n mirages

- **Lakonishok, J. & Smidt, S. (1988).** "Are Seasonal Anomalies Real? A Ninety-Year
  Perspective." *Review of Financial Studies*, 1(4), 403–425. The canonical audit of
  day-of-week, turn-of-month, and holiday effects: many shrink or vanish out-of-sample.
- **Ariel, R. A. (1990).** "High Stock Returns Before Holidays: Existence and Evidence on
  Possible Causes." *Journal of Finance*, 45(5), 1611–1626. The pre-holiday effect — the
  closest documented cousin of a "Diwali" effect — is small and fragile to costs.
- **Sullivan, R., Timmermann, A. & White, H. (2001).** "Dangers of Data Mining: The Case
  of Calendar Effects in Stock Returns." *Journal of Econometrics*, 105(1), 249–286.
  Shows that once you account for the universe of calendar rules implicitly searched, most
  "significant" seasonals are not.
- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "… and the Cross-Section of Expected
  Returns." *Review of Financial Studies*, 29(1), 5–68. The modern t ≥ 3 hurdle for a new
  anomaly given the multiple-testing problem; a sub-2 t on 14 events is nowhere near it.

## Diwali-specific commentary

- **NSE / BSE Muhurat trading circulars (annual).** The exchanges publish the Muhurat
  session timing each year; these notices fix the Laxmi-Pujan date used in `data.py`.
- **Popular-press "Muhurat returns" tallies.** Brokerage notes routinely report that the
  Sensex/Nifty "rose in N of the last M Muhurat sessions." These tallies (a) test against
  a 50% coin rather than the up-drift base rate, and (b) cherry-pick the one-hour session
  in INR — neither survives an honest baseline or a tradeable foreign proxy.

## Method lineage

- **Event study with execution lag.** Enter at the first proxy close strictly after the
  event (one-session lag); hold a fixed horizon. No look-ahead — the date is a known
  calendar event but the return is realized strictly afterwards.
- **Block-permutation test.** Place the same number of equal-length windows at random
  start dates 10,000 times; the one-sided p-value is the fraction of random placements
  whose mean meets or exceeds the observed Diwali mean.
- **Newey-West (HAC) t-stat.** On the per-event excess returns with lag = holding period,
  to be robust to any within-window overlap. With 14–23 well-separated events the plain
  and HAC t-stats are close.
- **Costs.** One-way × NAV on entry and exit (long-only ETF, no borrow); round-trip drag
  reported alongside gross and net excess.

## Data sources

- **iShares MSCI India ETF (INDA).** Daily auto-adjusted (total-return) closes via
  `yfinance`, cache-only by default under `_cache/inda_daily.parquet`. USD, US-listed,
  history from 2012. Survivorship is not a concern (single liquid ETF) but the **USD/proxy
  mismatch and tiny n are named on the Signal axis**.
- **Diwali / Muhurat-session dates.** Hardcoded in `data.py` (2003–2025). Sources: NSE/BSE
  Muhurat trading notices; drikpanchang.com Diwali/Lakshmi-Puja calendar; Wikipedia
  "Diwali."

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the base-rate-trap teardown of a
  famous folklore "indicator" — the methodological template for this study.
- **[Study 223 — Same-Month Seasonality](../../223-same-month-seasonality/)**: a
  calendar-seasonality study with a real (but fragile) effect, for contrast.
- Other seasonal-folklore teardowns in the desk's calendar/quirk family (turn-of-month,
  pre-holiday, day-of-week).
