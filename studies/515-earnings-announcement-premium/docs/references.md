# References & literature map — Study 515 (Earnings-Announcement Premium)

## The claim under test

- **The folklore / the factor.** A *disproportionate* share of a stock's total return is earned
  in the handful of days *around* its scheduled earnings announcement — so a calendar strategy
  that simply *holds names while they report* should beat buy & hold. This is a **level premium
  on the announcement days themselves**, paid for bearing announcement risk, and is independent
  of the *sign* of the surprise.
- **The seminal evidence.** William Beaver, *The Information Content of Annual Earnings
  Announcements* (1968, Journal of Accounting Research) first documented that **return variance**
  and volume spike on announcement days. Andrea Frazzini & Owen Lamont, *The Earnings
  Announcement Premium and Trading Volume* (2007, NBER w13090) showed that average **returns** —
  not just variance — are systematically *higher* in the announcement month/window, and built the
  long-the-announcers strategy we replicate here. Lamont & Frazzini estimate a premium on the
  order of a few percent annualised on the broad cross-section.
- **Why a premium could exist.** Announcement-day returns compensate for a spike in
  **idiosyncratic risk** that cannot be diversified away over the event (Savor & Wilson, *Earnings
  Announcements and Systematic Risk*, 2016, Journal of Finance; and *How Much Do Investors Care
  About Macroeconomic Risk?*, 2013, JFQA — the same logic for FOMC/CPI days). Barber, De George,
  Lehavy & Trueman (2013) on the cross-country announcement premium.

## Distinct from PEAD (study 363)

- **PEAD = drift conditional on the surprise sign.** [`../363-pead-drift`](../363-pead-drift)
  sorts events by the *signed* EPS surprise and measures the forward drift — a *cross-sectional*,
  *post*-announcement effect (Bernard & Thomas 1989).
- **This study = the announcement-day level premium.** We do **not** condition on the surprise
  sign; we ask whether the announcement *days themselves* carry extra return on average, the
  Frazzini–Lamont calendar effect. A stock can have zero PEAD and still have an announcement
  premium (or vice-versa) — they are orthogonal claims, which is why the desk runs both.

## Why the effect can vanish on a large-cap survivor basket

- **The premium concentrates in small, neglected names.** Frazzini & Lamont (2007) find the
  premium is strongest among small, low-analyst-coverage, high-retail-ownership stocks — the
  opposite of our liquid large-cap basket, where it is expected to be small or absent.
- **Decay post-publication.** McLean & Pontiff (2016, *Does Academic Research Destroy Stock
  Return Predictability?*, Journal of Finance) document ~mid-double-digit-percent decay in
  published anomalies after their discovery; an announcement premium first published in 2007 and
  measured here through 2026 is a prime candidate. Harvey, Liu & Zhu (2016, RFS) on the
  multiple-testing discount for any single published factor.
- **Survivorship cuts the *wrong* way here.** Because we keep only names that survived their
  earnings, our basket is, if anything, biased *toward* a positive announcement premium — and we
  still find none, which strengthens (not weakens) the null.

## Why a small premium still needs a placebo + a powered control

- **Welch / one-sample t** (Welch, 1947, *The generalization of "Student's" problem*) for the
  announce-vs-rest mean. Announcement days **cluster** in earnings seasons, so a naive *t* over
  pooled stock-days overstates independence; we add a **random-calendar placebo** that relocates
  the announcement tags to random days of the *same density* per name (Fisher's randomization
  logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Faithful, *powered* engine.** A null result is only credible if the detector *could* have
  seen a real effect. The deterministic synthetic control plants a known announcement premium and
  confirms the test lights up at *t* ≈ 7 — so the real-tape *t* ≈ 0 is a genuine absence, not a
  blunt instrument.

## Method lineage (the desk's shared engine)

- **Per-day & per-event premium + one-sample / Welch t.**
  [`strategy.summarize_premium`](../earnings_premium/strategy.py),
  [`strategy.per_event_premium`](../earnings_premium/strategy.py) — announce vs rest, with a
  within-name baseline so cross-name level differences cancel.
- **Random-calendar placebo.** [`strategy.placebo_pvalue`](../earnings_premium/strategy.py) —
  20,000 random re-taggings of the same density; the honest calendar-effect null.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../earnings_premium/data.py) plants a known announcement premium; with
  the edge set to zero the inference must NOT manufacture significance — the offline core runs
  with no network.

## Data sources used here

- **yfinance** daily adjusted closes for a fixed 30-name large-cap basket + per-name
  `Ticker.get_earnings_dates` (scheduled earnings dates), 2005-01-03 → 2026-06-26, cached under
  `_cache/ep_prices.csv` and `_cache/ep_dates.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../363-pead-drift`](../363-pead-drift) — the *signed-surprise drift* sibling (the
  announcement premium's cross-sectional cousin); the contrast is the whole point.
- The **research-method demos** (look-ahead, multiple-testing) frame why a powered synthetic
  control is the only honest way to stamp a *null*: you must prove the detector wasn't simply too
  blunt to see the effect.
