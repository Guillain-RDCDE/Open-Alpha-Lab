# References & literature map — Study 915 (K-1 vs 1099)

## The claim under test

- **The "No K-1" pitch.** Commodity ETPs that hold futures directly are partnerships for
  tax purposes: they send a **Schedule K-1** every spring, often late enough to force a
  filing extension, and they can generate unrelated business taxable income (UBTI) inside
  a retirement account. Invesco's PDBC — literally branded *"No K-1"* — wraps the same
  strategy family inside a 1940-Act fund by pushing the futures into a Cayman controlled
  foreign corporation, and so issues an ordinary **Form 1099**. The industry pitch is
  identical exposure, less paperwork. The counter-claim, repeated across tax-aware
  commentary, is that the convenience is bought: the 1940-Act structure loses the
  Section 1256 60/40 rate, adds a subsidiary layer, and forces ordinary-income
  distributions. Study 915 asks whether that give-back is visible on the tape.
- **The steelman for the K-1 wrapper.** Under **IRC §1256**, regulated futures contracts
  are marked to market at year end and taxed **60% long-term / 40% short-term regardless
  of holding period** — a materially better statutory rate than ordinary income, available
  to the partnership wrapper and not to the fund wrapper. The price is the loss of
  deferral: the tax is due whether or not you sell.
- **The steelman for the 1099 wrapper.** Deferral has real value. Price appreciation inside
  a RIC compounds untaxed until sale and is then taxed at long-term rates; only the
  distributed income is taxed annually. Whether 60/40-without-deferral beats
  ordinary-distributions-with-deferral is an arithmetic question whose answer depends on
  the distribution policy and the investor's bracket — which is exactly why this study
  sweeps both instead of quoting one number.

## The tax mechanics (the source of the modelled layer)

- **IRC §1256**, *Section 1256 contracts marked to market* — the 60/40 rule and the annual
  mark-to-market that removes deferral and steps up basis. The blend implemented in
  [`strategy.blended_1256_rate`](../wrapper_tax/strategy.py).
- **IRC §851 / §7704 and Subchapter M** — the regulated-investment-company qualifying-income
  rules that make direct commodity-futures income problematic for a 1940-Act fund, and the
  controlled-foreign-corporation subsidiary route that PDBC-style funds use to satisfy them.
  This is the structural reason the two wrappers exist at all.
- **IRS Revenue Ruling 2006-1 and Rev. Rul. 2006-31**, and the subsequent private-letter
  ruling practice on commodity-linked notes and CFC subsidiaries — the administrative
  history that produced the "No K-1" fund format.
- **IRC §1411** — the 3.8% net investment income tax, folded into the upper two bracket
  rows of the sweep.

## Why a wrapper difference is hard to see on a tape

- **Elton, Gruber & Busse (2004), *Are Investors Rational? Choices among Index Funds*,
  Journal of Finance** — index products differentiated only by fee and structure show
  performance differences dominated by cost, and cost differences of tens of basis points
  need long samples to establish. The same power problem governs this comparison, which is
  why the study reports an explicit **minimum detectable difference**.
- **Petajisto (2017), *Inefficiencies in the Pricing of Exchange-Traded Funds*, Financial
  Analysts Journal** — ETF closing prices carry premium/discount noise that mean-reverts
  within days. That is precisely the −0.43 lag-1 autocorrelation the daily wrapper
  difference shows here, and the reason the naive √252-scaled tracking error (5.30%/yr)
  overstates the divergence a holder lives through (monthly: 1.46%/yr).
- **Poterba & Shoven (2002), *Exchange-Traded Funds: A New Investment Option for Taxable
  Investors*, American Economic Review P&P** — the canonical framework for comparing
  wrappers on an **after-tax** rather than pre-tax basis, and the source of the convention
  used here (taxes paid annually out of the account, terminal liquidation at long-term
  rates, basis tracked through reinvested distributions).
- **Bergstresser & Poterba (2002), *Do After-Tax Returns Affect Mutual Fund Inflows?*,
  Journal of Financial Economics** — after-tax return differences of this size (tens of bp)
  are economically real to investors even when they are statistically invisible in a
  decade of returns. Hence the study reports the modelled gap *and* refuses to promote it
  past an assumption-dependent range.
- **Dickson & Shoven (1995), *Taxation and Mutual Funds: An Investor Perspective*, Tax
  Policy and the Economy / NBER WP 4393** — the deferral arithmetic itself: the value of
  postponing capital-gains tax rises with the horizon and falls with the payout ratio, the
  two axes swept here (`payout_share` and the eleven-year hold).

## Related desk studies (dedup)

- **[Study 908 — Optimized-Roll Commodities](../../908-optimized-roll-commodities/)**: races
  **different indices** (USCI's optimized roll against DBC / GSG / DJP) to ask whether the
  roll methodology pays. Study 915 holds the index family *constant* and varies only the
  **legal wrapper** — DBC and PDBC are the same manager on the same strategy family, so the
  difference isolates structure rather than methodology. 908's question is "which index?";
  915's is "which envelope around it?".
- **[Study 35 — Contango](../../35-contango/)** and
  **[Study 794 — Commodity-Carry](../../794-commodity-carry/)**: the roll yield itself, as a
  timing rule and as a cross-sectional long-short. Neither compares wrappers.
- **[Study 661 — USO-Roll-Decay](../../661-uso-roll-decay/)** and
  **[Study 619 — BITO-Roll-Drag](../../619-bito-roll-drag/)**: the *futures-roll* cost of a
  single front-month vehicle. Both wrappers here roll the same way; the roll cancels in the
  difference.
- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  whether a fund's past tracking difference predicts its future one, across many pairs.
  Study 915 is a single, structurally motivated pair and adds the **tax regime** as the
  object of interest — the tracking difference is the nuisance term here, not the subject.
- **[Study 378 — ETF-NAV-Premium](../../378-etf-nav-premium/)**: the premium/discount itself
  as a tradable signal. Here the premium noise is only diagnosed (the −0.43 autocorrelation)
  to justify measuring the wrapper difference at monthly frequency.
- **[Study 599 — Tax-Loss Harvesting](../../599-tax-loss-harvesting/)** and
  **[Study 616 — Muni-CEF Tax-Loss](../../616-muni-cef-tax-loss/)**: taxes as an *action*
  (harvesting). Here tax is a fixed property of the wrapper you chose, not a decision you
  revisit.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../wrapper_tax/strategy.py). It matters unusually much here:
  the daily difference is strongly negatively autocorrelated, so the HAC standard error is
  a third of the i.i.d. one.
- **Circular block bootstrap.** Politis & Romano (1992, 1994) — the paired resampling in
  [`strategy.bootstrap_diff_ci`](../wrapper_tax/strategy.py) and
  [`strategy.bootstrap_sharpe_adv_ci`](../wrapper_tax/strategy.py), where both legs are
  drawn on the same dates so the shared commodity beta cancels.
- **Sharpe-difference inference.** Jobson & Korkie (1981), *Performance Hypothesis Testing
  with the Sharpe and Treynor Measures*, Journal of Finance — the return-difference framing
  used for the Sharpe advantage.
- **Binomial / Wilson interval on the annual win count.** Wilson (1927) —
  [`strategy.wilson_interval`](../wrapper_tax/strategy.py).

## Data sources

- **DBC** (K-1 commodity pool), **PDBC** (1099 fund), **USCI / USO / BNO** (K-1 context
  vehicles) and **BIL** (1-3 month T-bills, the cash leg *and* the collateral-interest
  proxy) — daily **total-return** closes via `yfinance` (`auto_adjust=True`),
  2006 → 2026-06-30, sliced to the as-of. Total return rather than price is essential:
  PDBC's ordinary distributions were large in 2023-2025, and a price-only comparison would
  hand the K-1 wrapper a spurious win.
- **As-of 2026-06-30.** The partial current month is dropped; the after-tax model uses only
  the eleven complete calendar years 2015-2025.
- **Non-tape inputs, all labelled PROXY/ASSUMPTION and swept where they matter:** the two
  **current** prospectus expense ratios (DBC 0.87%, PDBC 0.59%) applied to the whole window
  — a labelled **hindsight** input, used only to compare the fee gap's prediction against
  the tape and never subtracted from a return series; the marginal-rate pairs; the
  distribution `payout_share`; and the use of BIL's total return as the collateral-interest
  component of each wrapper's annual result. No K-1, 1099, holdings, or fund distribution
  filing was parsed — the tax layer is a transparent model, not a measurement, and the
  study's verdict does not rest on it.
- **What the design cannot separate.** PDBC is *actively* managed and does not mechanically
  replicate DBC's index, so "wrapper" and "basket" are confounded in this pair by
  construction. The study's null is therefore a null on the *joint* difference; the two
  outlier years (2018, 2025) are interpreted as basket drift but are **not decomposed**.
  Every other honest reading — that a wrapper cost of up to ~1%/yr could hide inside the
  noise — is what the minimum-detectable-difference exhibit exists to bound.
