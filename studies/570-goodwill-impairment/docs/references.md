# References & literature map — Study 570 (Goodwill-Impairment)

## The claim, at full strength

- **Li, Shroff, Venkataraman & Zhang (2011)**, *"Causes and Consequences of Goodwill Impairment
  Losses."* *Review of Accounting Studies* 16(4). Goodwill write-downs are associated with prior
  **overpayment** in acquisitions and are followed by **negative** market reactions and downward
  analyst revisions — the empirical backbone of the overpaid-acquisition story this study proxies.
- **Hayn & Hughes (2006)**, *"Leading Indicators of Goodwill Impairment."* *Journal of Accounting,
  Auditing & Finance* 21(3). A large goodwill balance relative to the acquisition premium is a
  **leading indicator** of a future impairment — the "bloated goodwill predicts the write-down"
  leg — and managers often *delay* the write-down, which is why the market is slow to price it.
- **Gu & Lev (2011)**, *"Overpriced Shares, Ill-Advised Acquisitions, and Goodwill Impairment."*
  *The Accounting Review* 86(6). Firms that use over-valued stock to overpay for acquisitions book
  goodwill that later impairs — linking the goodwill balance to *value destruction* and subsequent
  underperformance.
- **Ramanna & Watts (2012)**, *"Evidence on the Use of Unverifiable Estimates in Required Goodwill
  Impairment."* *Review of Accounting Studies* 17(4). Under SFAS 142, impairment is a
  managerial-discretion decision — write-downs are timed, not mechanical — which is the friction
  that lets the balance predict the (delayed) event and its return drag.

## The effect we model

- The synthetic panel plants the two legs of the story with one knob. `imp_beta` (log-odds of a
  forward impairment per unit centred **overpayment**) reproduces the **leading-indicator** result
  (Hayn–Hughes): high goodwill/assets impairs more. `ret_alpha ≤ 0` reproduces the **return drag**
  (Li et al.; Gu–Lev): high goodwill/assets underperforms, with an extra announcement jolt on firms
  that actually impair. `ret_alpha = 0` is the null — the impairment link survives (bloated goodwill
  still impairs more) while the *return* puzzle switches off, isolating the tradable claim.

## Neighbours on this bench (the dedup map)

- **[Study 231 — Sloan accruals](../../231-sloan-accruals/)** — the accruals anomaly (high accruals
  → low returns via earnings-quality reversal). A *different* balance-sheet red flag; Study 570 is
  the **goodwill / overpaid-M&A** channel, not accrual reversal.
- **[Study 522 — percent operating accruals](../../522-percent-operating-accruals/)** — the scaled
  accruals variant. Same family (earnings quality), distinct signal.
- **[Study 540 — distress-risk anomaly](../../540-distress-risk-anomaly/)** — a *return* puzzle from
  a balance-sheet condition (distress). Study 570 shares the sort-and-test machinery but tests the
  **goodwill/asset ratio and its impairment events**, not failure probability.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the low-minus-high bucket spread.
- **Two-proportion z-test** — for the high-minus-low impairment-rate gap.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  goodwill labels against forward returns and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../METHODOLOGY.md) — the inference bar (a synthetic-only
  study is capped at `WEAK`; `REAL` needs a robust *t* ≥ 2 on a real tape), the explicit
  data-availability caveat on the SIGNAL axis, one execution lag, costs one-way × NAV with shorts
  paying borrow, and the seed-robust synthetic positive control (≥ 20 seeds).
