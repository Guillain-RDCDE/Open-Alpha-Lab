# References & literature map — Study 914 (Securities-Lending Offset)

## The claim under test

- **The lending-offset thesis.** An index fund may lend the shares it holds to short
  sellers and keep a share of the fee. In hard-to-borrow classes — US small caps, emerging
  markets — that revenue is supposed to be large enough to offset a meaningful part of the
  expense ratio, so a lending-heavy fund's realised **tracking difference** should be
  *better* than `−ER`. Sponsors advertise exactly this: iShares publishes a securities
  lending report per fund, and Vanguard has long marketed returning "substantially all"
  lending revenue to the fund. The marketing version — "our fee is effectively free in EM"
  — is the version this study tests on returns.
- **The steelman.** The mechanism is real and audited. It is not a forecast, a factor or a
  timing rule; it is an accrual that either shows up in the fund's NAV or does not. If it
  is worth what the brochures imply, two same-class funds with different lending programmes
  should drift apart by more than their fee gap.

## Why it can be invisible even when it is real

- **Size.** Realised net lending yields for broad, liquid index funds are small. Blocher &
  Whaley (2016), *Two-Sided Markets in Asset Management: Exchange-Traded Funds and
  Securities Lending* (SSRN 2474904), find lending income for most large equity ETFs
  measured in a very few basis points a year, concentrated in the small-cap and
  hard-to-borrow tail. The number to be detected is 1–15 bp/yr for the funds on this tape.
- **The measurement floor.** Two funds in the same asset class still track different
  baskets. Their relative tracking error here runs 0.5–3.8% a year, which sets a
  |*t*| = 2 detection floor of 20–147 bp/yr over the available sample — an order of
  magnitude above the effect. This is the study's central methodological point, and it is
  why the honest output is a **bound**, not an estimate.
- **The agency wedge.** Even when the gross fee is large, the fund keeps only a split, and
  the lending agent (often an affiliate of the sponsor) keeps the rest. Evans, Ferreira &
  Prado (2017), *Fund Performance and Equity Lending: Why Lend What You Can Sell?*, Review
  of Finance 21(3), document that lending revenue is frequently *not* fully passed through
  and that the collateral-reinvestment risk is borne by the fund. Duffie, Gârleanu &
  Pedersen (2002), *Securities Lending, Shorting, and Pricing*, Journal of Financial
  Economics 66 — the price-of-borrow theory underneath all of it.
- **Where the fee actually is largest.** D'Avolio (2002), *The Market for Borrowing Stock*,
  Journal of Financial Economics 66 — the specials tail. Broad index funds hold the market,
  not the specials tail, so their weighted-average lending yield is dominated by general
  collateral, which is nearly free.

## Tracking difference as a measurement

- **Elton, Gruber, Comer & Li (2002),** *Spiders: Where Are the Bugs?*, Journal of Business
  75(3) — the original SPY tracking-difference autopsy, and the reason SPY is the natural
  control here: as a **unit investment trust** it may neither lend its securities nor
  reinvest dividends between distributions.
- **Petajisto (2017),** *Inefficiencies in the Pricing of Exchange-Traded Funds*, Financial
  Analysts Journal 73(1) — the ETF price/NAV wedge and how much of a "tracking" number is
  actually a pricing artefact.
- **Ben-David, Franzoni & Moussawi (2017),** *Exchange-Traded Funds*, Annual Review of
  Financial Economics 9 — the survey of ETF frictions: fees, lending, creation/redemption,
  and what each is worth.

## Related desk studies (dedup)

- **[Study 557 — Borrow-Fee-Signal](../../557-borrow-fee-signal/)**: uses the borrow fee as
  a *cross-sectional forecasting signal for stock returns*. Study 914 does the opposite —
  it asks who receives the money, and never uses borrow as a predictor.
- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  asks whether last year's *best* S&P 500 tracker stays best next year (a persistence
  question on a single asset class). Study 914 asks a *level* question across five asset
  classes: is the residual systematically better than the fee gap, and can lending revenue
  be seen in it at all.
- **[Study 920 — Total Cost of Ownership](../../920-total-cost-of-ownership/)**: trades off
  expense ratio against spread to find the break-even holding period. Study 914 holds the
  holding period fixed at "forever" and asks only whether the fee gap is the *whole* story.
- **[Study 378 — ETF-NAV-Premium](../../378-etf-nav-premium/)**: the price-versus-NAV
  discount and whether it mean-reverts — a *pricing* wedge, not an *accrual* wedge.
- **[Study 624 — Buffer-ETF-Cost](../../624-buffer-etf-cost/)** and
  **[Study 619 — BITO-Roll-Drag](../../619-bito-roll-drag/)**: the desk's other
  "what does the wrapper really cost you" structural races. Same family, different
  wrappers, and neither touches securities lending.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica 55 —
  [`strategy.newey_west_t`](../lending_offset/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA 89
  — [`strategy.block_bootstrap_mean_ci`](../lending_offset/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Bounding rather than estimating.** When the noise floor exceeds the effect size, the
  reportable quantity is the confidence-interval edge, not the point estimate — the desk's
  standing rule for underpowered tests (see METHODOLOGY → *The inference bar*).

## Data sources

- **SPY / IVV** (S&P 500), **IWM / IJR** (US small cap), **EEM / IEMG** (emerging markets),
  **EFA / IEFA / VEA** (developed ex-US), **BIL** (cash proxy) — daily **total-return**
  closes via `yfinance` (`auto_adjust=True`), through **2026-06-30**. Nothing in this study
  is price-only; a price-only series would be swamped by dividend timing at the basis-point
  scale that matters here.
- **Expense ratios are a PROXY/ASSUMPTION**, not tape: the sponsors' current published net
  ERs held constant through history (`data.EXPENSE_RATIOS`). **This is the study's binding
  assumption, and it is one-sided.** Almost every fund here was cut during the sample, and
  the *cheap* legs were cut hardest — IVV 0.0945 → 0.03, IEMG 0.18 → 0.09, IEFA 0.14 →
  0.07, IJR 0.20 → 0.06 — while the dear legs barely moved (EEM 0.75 → 0.70, IWM 0.20 →
  0.19, SPY unchanged). Holding today's values constant therefore overstates every fee gap
  and biases every residual upward. Both legs are swept (`strategy.er_sweep`,
  `strategy.er_sweep_b`) and the whole table is re-run under early-era and approximately
  time-weighted schedules (`data.ER_HISTORY_BAND`). The earliest in-sample and current values are
  the sponsors' published headline ERs; the intermediate values are an **approximation** of
  the published cut schedule, used only as a bracketing scenario — no conclusion rests on
  them being exact, because the two endpoints bound them.
- **A note on precision.** Published *net* ERs also differ from realised fee accrual through
  fee waivers and the gross/net distinction; this study does not model that, which is a
  further reason its output is a magnitude bound rather than a point estimate.
- **Borrow fees are an ASSUMPTION**, swept 0 → 100 bp/yr in `strategy.borrow_sweep`.
- **Fund-level securities-lending revenue is not in the tape at all.** It appears only in
  annual and semi-annual reports and in sponsor lending disclosures, at a frequency and lag
  this study does not use. Every statement here is an inference from realised relative
  returns, and is written that way.
