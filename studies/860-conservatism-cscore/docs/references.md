# References & literature map — Study 860 (Accounting Conservatism, C-Score)

## The claim under test

- **The thesis.** *Conservative* accounting recognises bad news faster than good news (Basu's
  asymmetric timeliness) and applies a "lower-of-cost" prudence that systematically
  **understates net operating assets**. The gap between conservative book value and economic
  value is a stock of **hidden reserves** — allowances, valuation reserves, LIFO/inventory
  reserves, expensed R&D — that later unwind into earnings. Penman & Zhang (2002) formalise a
  **C-score** = estimated reserves ÷ net operating assets and argue that the *level* (and its
  change) signals the quality of current earnings and, in the strong form, **predicts forward
  stock returns**: firms carrying a large, growing reserve cushion hold un-booked value the
  market may under-price. We test the strong form: rank names on a simplified reserve-intensity
  C-score and measure the forward long-short return spread (long high-conservatism / short low).
- **The academic anchor.** Two pillars. **Basu (1997, "The conservatism principle and the
  asymmetric timeliness of earnings", *Journal of Accounting & Economics* 24)** defines
  conservatism as the greater sensitivity of earnings to bad news (proxied by negative stock
  returns) than to good news — the interaction test we run as the mechanism axis. **Penman &
  Zhang (2002, "Accounting Conservatism, the Quality of Earnings, and Stock Returns", *The
  Accounting Review* 77)** build the C-score from estimated reserves over NOA and link
  conservatism-driven earnings changes to future returns. The idea sits in the broader
  **fundamentals / accruals mispricing** family (Sloan 1996), alongside the balance-sheet-quality
  scores of Piotroski (2000) and Mohanram (2005).
- **The open question we test.** On a small, honestly-thin panel of US non-financial names that
  actually tag reserve/allowance accounts on EDGAR, does a *coarse* reserve-intensity C-score
  (a) **predict a forward return spread** (the market-mispricing claim) once ranked strictly on
  point-in-time filed values, held with one execution lag, and charged realistic long-short costs
  plus borrow, and (b) does the panel even **exhibit Basu conservatism** in the first place?

## What we measure, and the honesty rails

- **Signal, point-in-time.** `cscore` = (allowance for doubtful accounts + inventory valuation
  reserve + deferred-tax valuation allowance, whichever are tagged) ÷ `Assets`, known only at the
  **10-Q/10-K filing date** (`filed`), never the period end. A Penman-Zhang-style variant
  `cscore_noa` = reserves ÷ **net operating assets** (NOA = Assets − Cash − (Liabilities − Debt))
  is carried as a robustness cut where the components are disclosed. **This is a deliberately
  coarse proxy** — XBRL exposes only a subset of the reserves Penman-Zhang estimate (no LIFO
  reserve, no capitalised-R&D or advertising reserve), so the reserve total is a *floor* and the
  score conflates accounting prudence with business mix. We say so throughout.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names
  carrying a fresh signal into terciles (the panel is too thin for quintiles), **long the top
  (most conservative) / short the bottom** equal-weight, earn the **next** month's return (one
  execution lag). The decisive statistic is the **Newey-West (HAC, Bartlett) t** of the monthly
  long-short series — the autocorrelation-robust bar `REAL` is written against (METHODOLOGY →
  *The inference bar*). A one-sample t and a monthly hit-rate accompany it.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signal, measure
  top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a one-sample t and a
  **label-shuffle placebo** (permute signals, re-form random terciles), plus the tercile
  monotonicity picture.
- **Third axis — is the accounting conservative?** A pooled **Basu (1997)** regression of
  quarterly ROA on the contemporaneous return, a bad-news dummy, and their interaction; the
  interaction slope is the asymmetric-timeliness measure of conservatism. This is the *mechanism*
  check: the C-score can only be a proxy for conservatism if the panel is actually conservative.
  The pooled t is read as suggestive — filings cluster by quarter, so it is not a
  calendar-robust HAC statistic.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model.
- **Coverage is a first-class caveat, not a footnote.** Reserve/allowance tags are irregular:
  many names disclose only the allowance for doubtful accounts, and only in some years, so the
  cross-section is modest (≈20 names early, ≈31 late) and the reserve total under-counts true
  conservatism. Terciles on a coarse, floor-biased signal are noisy by construction; every number
  here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** (all still listed): a fixed roster of ~42 non-financial names
that tag the reserve concepts today. It cannot include operating firms that were acquired or
failed. For a long-*high*-conservatism / short-*low*-conservatism book both legs are drawn from
the same survivor pool, so the first-order equity-survivorship tilt partly cancels; the residual
risk is that reserve-signal informativeness itself is survivor-conditioned. We reason about the
bias direction explicitly rather than claiming it away, and never cite the survivor panel to
certify magnitude.

## Data sources

- **Reserve/allowance accounts, Assets, Cash, Liabilities, Debt, NetIncomeLoss** — SEC EDGAR
  XBRL `companyconcept` API (`data.sec.gov`), 10-Q/10-K instant/duration facts, de-duplicated on
  period end (earliest filing wins), keeping the filing date so the signal is strictly
  point-in-time. Cached under `_cache/cs_events.csv`.
- **Daily adjusted closes** — yfinance (no key), cached under `_cache/cs_prices.csv`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [229-beneish-m-score](../../229-beneish-m-score/) — the Beneish earnings-**manipulation**
  detector: an eight-ratio probit built to flag *aggressive* (income-inflating) accounting. This
  study scores the **opposite** posture — accounting *conservatism* (income-deferring prudence) —
  via reserve intensity, not a manipulation probability.
- [232-mohanram-g-score](../../232-mohanram-g-score/) — Mohanram's **G-score** for growth firms
  (profitability, cash-flow and earnings-stability signals). A different fundamental composite;
  it does not measure reserve-driven conservatism or Basu asymmetric timeliness.
- [855-accrual-quality](../../855-accrual-quality/) — Dechow-Dichev **accrual quality** (residual
  volatility of accruals regressed on cash flows). Both live in the "quality of accruals" family,
  but 855 measures how well accruals *map to cash* (estimation-error noise); this study measures
  the *level of hidden reserves* (accounting prudence) and the Basu bad-news asymmetry — a
  distinct construct and a distinct signal.
- [52-smoke-screen](../../52-smoke-screen/) — a research-method demonstration of how easily a
  plausible accounting overlay can manufacture a spurious backtest; the methodological cousin to
  our honest-null finding here (a good story — hidden reserves — that the tape does not reward).

None of the siblings rank on **reserve-intensity conservatism (Penman-Zhang C-score / Basu
asymmetry)** — this study's own axis.
