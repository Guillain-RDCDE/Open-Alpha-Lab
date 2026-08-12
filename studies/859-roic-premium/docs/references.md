# References & literature map — Study 859 (Return-on-Invested-Capital Premium)

## The claim under test

- **The thesis.** **ROIC — return on invested capital — is the "cleanest" profitability/quality
  gauge**, and sorting the cross-section on it should earn a forward long-short spread. The pitch,
  as value investors and quality-factor practitioners state it: ROE (net income ÷ book equity)
  can be inflated by leverage — a firm that buys back stock or loads up on debt lifts ROE without
  improving the business — whereas ROIC divides *unlevered* operating profit (NOPAT) by *all* the
  capital deployed (debt **plus** equity **minus** cash), so it isolates how efficiently the
  operating business itself converts capital into profit. High, stable ROIC is the numerical
  fingerprint of a "wide-moat compounder"; low or falling ROIC flags a capital-destroying
  business. We test the strong form: rank names on **ROIC level** (and on its year-over-year
  **change**), long the top tercile / short the bottom, and measure the forward return spread.

- **The definitions we use.**

      NOPAT           = OperatingIncomeLoss (TTM) × (1 − tax_rate)
      InvestedCapital = LongTermDebt + StockholdersEquity − CashAndCashEquivalents
      ROIC            = NOPAT / InvestedCapital

  A flat 21% ``tax_rate`` (post-2017 US statutory) turns operating income into NOPAT. Because a
  *common* scalar ``(1 − tax_rate)`` multiplies every name's NOPAT identically, **it does not
  change the cross-sectional ranking** — only the reported ROIC magnitude — so the sort (and every
  long-short result) is invariant to the tax assumption. Invested capital uses long-term
  (noncurrent) debt as the debt proxy per the study spec; this understates gross debt for firms
  with large short-term borrowings and is noted as a first-class simplification.

- **The academic anchor.** ROIC-vs-cost-of-capital ("economic profit" / EVA) is the value-creation
  identity of corporate finance (Stewart 1991; Koller, Goedhart & Wessels, *Valuation*,
  McKinsey). Its use as a *return-predicting* cross-sectional factor sits inside the **quality**
  literature: Novy-Marx (2013, *JFE*, "The Other Side of Value: The Gross Profitability Premium")
  argues a *gross*, un-levered profitability measure predicts returns better than ROE; Fama &
  French (2015) fold profitability (RMW) into the five-factor model; Asness, Frazzini & Pedersen
  (2019, *RFS*, "Quality Minus Junk") build a composite quality factor in which profitability and
  *return on capital* are core inputs. The open question we test is whether **ROIC specifically**
  — the unlevered, cash-adjusted return-on-capital number — earns a spread on this panel, and
  whether it **adds anything over plain ROE and gross profitability** measured the same way.

- **The open question we test.** On a small, honestly-thin panel of large US non-financial filers,
  does ROIC (a) **earn a forward return spread** and (b) **beat / add to** plain ROE (Study 200)
  and gross profitability (Study 122), once we rank strictly on point-in-time filed values, hold
  with one execution lag, and charge realistic long-short costs plus borrow?

## What we measure, and the honesty rails

- **Signal, point-in-time.** ``roic`` = TTM NOPAT ÷ invested capital, known only at the
  **10-Q/10-K filing date** (``filed``), never the period end. A year-over-year **change** variant
  (``roic_chg``) is carried. Concepts: ``OperatingIncomeLoss`` (summed to trailing-twelve-month),
  ``StockholdersEquity``, ``LongTermDebtNoncurrent``, ``CashAndCashEquivalentsAtCarryingValue``.
- **Primary test — calendar-time long-short (Newey-West).** Each month-end, rank the names that
  carry a fresh signal into terciles (the panel is too thin for quintiles), long the top / short
  the bottom equal-weight, earn the **next** month's return (one execution lag). The decisive
  statistic is the **Newey-West (HAC, Bartlett) t** of the monthly long-short series — the
  autocorrelation-robust bar `REAL` is written against. A one-sample t and a monthly hit-rate
  accompany it, and an **era split** at 2018 (the TCJA statutory-rate change) checks sub-period
  robustness.
- **Cross-check — pooled event drift.** Bucket all (ticker, filing) events by the signal, measure
  top-minus-bottom forward drift over ≈1m/1q/2q horizons, with a one-sample t, a **label-shuffle
  placebo** (permute signals, re-form random terciles), and the tercile **monotonicity** picture.
- **Third axis — does ROIC add anything?** The *same* calendar long-short run on ``roic``,
  ``roe`` (Study 200's signal) and ``gp`` (Study 122's gross profitability) over the identical
  panel, plus the pooled ROIC↔ROE cross-sectional **rank correlation**. If ROIC's spread is no
  better than ROE's and the two are ~collinear, ROIC "adds nothing" over the cheaper, older
  signals.
- **Costs & borrow.** The tradability timer charges one-way cost × NAV × monthly turnover on
  **both** legs and makes the short leg pay an annualised borrow — the standard desk long-short
  friction model.
- **Coverage is a first-class caveat, not a footnote.** XBRL fundamentals begin ~2009-2010, and
  several names re-registered under new CIKs after mergers/reorganisations (XOM 2024, DIS 2019,
  LIN 2018), truncating their machine-readable history. The monthly cross-section is therefore
  small early and only becomes reasonably wide after ~2012. Terciles on a thin cross-section are
  noisy by construction; every number here should be read in that light.

## Survivorship — named on the Signal axis

The basket is **current survivors** — a fixed roster of ~44 large US non-financial names all still
listed today. It cannot include mega-caps that were acquired, broken up or delisted (the
capital-destroyers a low-ROIC short leg most wants to hold). For a long-*high-ROIC* / short-*low-
ROIC* sort the short leg is exactly where survivorship bites hardest — the worst businesses left
the index — so any short-side alpha is an **upper bound**. We name the bias on the Signal axis and
never cite the survivor panel to certify magnitude.

## Data sources

- **Operating income, net income, gross profit, equity, long-term debt, cash, assets** — SEC
  EDGAR XBRL ``companyconcept`` API (``data.sec.gov``), 10-Q/10-K instant/duration facts,
  de-duplicated on period end (earliest filing wins), keeping the filing date so the signal is
  strictly point-in-time; single-quarter operating/net/gross flows summed to trailing-twelve-
  month. Cached under ``_cache/roic_events.csv``.
- **Daily adjusted closes** — yfinance (no key), cached under ``_cache/roic_prices.csv``.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [200-roe-quality](../../200-roe-quality/) — ranks on **return on equity** = NetIncome ÷ lagged
  book equity, the *levered* quality number. ROIC is the point: an **unlevered, cash-adjusted**
  return-on-capital that strips out the leverage ROE bakes in. Study 859 runs ROE *alongside* ROIC
  on the same panel precisely to ask whether the unlevered version adds anything.
- [122-gross-profitability](../../122-gross-profitability/) — Novy-Marx **GrossProfit ÷ Assets**,
  a top-of-the-income-statement profitability ratio with **no capital-structure adjustment and no
  operating-cost/​tax treatment**. ROIC is an after-operating-cost, after-tax return on *invested*
  (not total) capital. Study 859 carries GP/Assets as a head-to-head baseline.
- [242-quality-minus-junk](../../242-quality-minus-junk/) — the AQR **composite** quality factor
  (profitability + growth + safety + payout across many inputs). ROIC is a **single**, specific
  return-on-capital ratio, not a multi-signal composite.
- [521-cash-based-operating-profitability](../../521-cash-based-operating-profitability/) — Ball et
  al.'s **cash-based** operating profitability (accruals stripped out of the *numerator*,
  scaled by assets). ROIC keeps accrual operating income (NOPAT) and changes the **denominator**
  to invested capital (debt + equity − cash); a different axis of the profitability cube.

None of the siblings rank on **NOPAT ÷ invested capital** — this study's own axis.
