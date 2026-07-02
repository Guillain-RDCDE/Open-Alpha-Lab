# References & literature map — Study 563 (Secondary-Offering-Drift)

## The claim, at full strength

- **Loughran & Ritter (1995)**, *"The New Issues Puzzle."* *Journal of Finance* 50(1). The
  canonical long-run *underperformance* of equity issuers: firms that conduct IPOs and **seasoned
  equity offerings (SEOs)** go on to earn substantially *lower* returns than matched non-issuers
  over the following years. The empirical seed of the "issue shares → the stock slides" claim this
  study tests as a short-horizon drift.
- **Spiess & Affleck-Graves (1995)**, *"Underperformance in Long-run Stock Returns Following
  Seasoned Equity Offerings."* *Journal of Financial Economics* 38(3). The SEO-specific long-run
  underperformance result — the direct academic statement that seasoned-offering firms drift down.
- **Myers & Majluf (1984)**, *"Corporate Financing and Investment Decisions When Firms Have
  Information That Investors Do Not Have."* *JFE* 13(2). The **adverse-selection / negative-signal**
  mechanism: managers issue equity when they believe the stock is *over*valued, so an offering is
  bearish news — the theory behind a negative announcement/drift effect.
- **Asquith & Mullins (1986)**, *"Equity Issues and Offering Dilution."* *JFE* 15(1). The
  **announcement-effect** result: a seasoned equity offering announcement is met with a negative
  ~2–3% price reaction — the documented *jump* (which our 1-day lag deliberately steps past to
  isolate the subsequent *drift*).
- **Pontiff & Woodgate (2008)**, *"Share Issuance and Cross-Sectional Returns."* *Journal of
  Finance* 63(2). Net share issuance is a robust cross-sectional return predictor — issuers
  underperform, repurchasers outperform. The factor cousin of this event study (see Study 519).

## The measure we build

- We measure the **abnormal** (stock − SPY) forward return over 1 / 3 / 6 / 12 months after each
  offering's public pricing, entering **one day after** (no look-ahead, and past the documented
  announcement-day jump), and test the mean against zero with a **Welch t** plus a **same-names
  left-tail placebo null**. The tradable expression is a **short** of the issuer, charged a
  one-way cost and a punitive short borrow (these names skew hard-to-borrow).

## Neighbours on this bench (the dedup map)

- **[Study 519 — Net-Share-Issuance](../../519-net-share-issuance/)** — the *cross-sectional
  factor*: rank firms by their annual net change in shares outstanding and sort. Study 563 is the
  **event study** of individual seasoned/secondary offerings — the discrete dilution *event*, not
  the continuous issuance *factor*. Complementary framings of the same underlying claim.
- **[Study 368 — Buyback-Drift](../../368-buyback-drift/)** — the **mirror image**: does a stock
  drift *up* after a buyback authorization? Same event-study machinery (abnormal returns vs SPY,
  1-day lag, same-names placebo), opposite corporate action and opposite predicted sign.
- **[Study 363 — PEAD-Drift](../../363-pead-drift/)** / **[Study 534 —
  Revenue-Surprise-Drift](../../534-revenue-surprise-drift/)** — other post-event drifts; the
  shared lesson is that a few-dozen single-name events are dominated by single-stock variance.

## Shared method

- **Welch (1947)** — the unequal-variance / one-sample *t* used on the abnormal-return sample.
- **Event-study methodology** (Fama, Fisher, Jensen & Roll 1969; MacKinlay 1997) — abnormal
  returns around a corporate event; here a market-model-lite (stock − SPY) abnormal return.
- **Label-shuffle / permutation and matched-placebo testing** (Fisher 1935; Good 2005) — the
  same-names random-date null: re-enter each name on random valid dates and read the tail
  probability of the real offering set's mean drift.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a *t* ≥ 2 on
  the real tape plus a placebo null and seed-robustness), the explicit survivorship/selection
  caveat, one execution lag, and costs one-way × NAV with shorts paying borrow.
