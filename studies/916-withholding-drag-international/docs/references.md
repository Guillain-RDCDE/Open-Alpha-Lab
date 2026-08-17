# References & literature map — Study 916 (Withholding Drag)

## The claim under test

- **The withholding-drag thesis.** A US-domiciled fund that holds foreign equities has
  foreign dividend withholding tax deducted at source before the cash ever reaches the
  fund's NAV. Treaty rates for a US fund are typically 15% (Japan, Germany, France,
  Switzerland after reclaim) with some markets at 0% (the United Kingdom) and some
  higher. On a developed-ex-US dividend yield of roughly 3%, an effective blended rate
  in the 10–15% range implies a drag of the order of 30–50 bp/yr — comparable to, or
  larger than, the fund's own expense ratio. The popular claim is that this drag is a
  hidden cost that "eats" a slice of an international fund's dividend before you see it.
- **The steelman.** The arithmetic is not in dispute; the question is whether the
  *magnitude* can be measured from public price and distribution data rather than
  assumed. This study attempts exactly that and reports where it fails.
- **Why identification is the whole problem.** The tape reports the fund's *net*
  distribution precisely. It never reports the *gross* dividend the underlying companies
  declared. Any candidate benchmark that is itself a US-domiciled fund suffers the same
  withholding, so differencing two US wrappers cancels the tax and leaves fees and index
  composition. The gross number lives in the fund's Form 1099-DIV box 7 / annual report,
  not in any price series.

## Where the withholding numbers come from

- **OECD Model Tax Convention, Article 10 (Dividends)** — the 15% portfolio-dividend
  ceiling that most US bilateral treaties adopt, and the legal basis for the rates
  assumed (and swept) here.
- **US-Japan Income Tax Convention (2003, protocol signed 2013)** and **US-Germany
  Income Tax Convention (1989, protocol 2006)** — 15% on portfolio dividends. Germany's
  26.375% statutory rate is reduced to the treaty 15% by *reclaim*, which funds do not
  always recover in full, so 15% is a floor rather than a certainty.
  **The UK's 0% is not a treaty rate.** The United Kingdom levies no withholding tax on
  dividends at all under its own domestic law, so a US fund holding UK shares suffers
  nothing regardless of treaty — which is why EWU's realised yield (374 bp) needs no
  withholding adjustment while EWJ's (148 bp) does, and why a Japan-weighted blend moves
  the estimate so much. Attributing the UK zero to the treaty would be wrong.
- **IRC §901/§904 — the foreign tax credit.** A US taxable holder can generally credit
  the withheld amount against US tax, so the economic drag falls mostly on holders in
  IRAs and 401(k)s who cannot claim it. The asset-location consequence is the subject of
  Dammon, Spatt & Zhang (2004), *Optimal Asset Location and Allocation with Taxable and
  Tax-Deferred Investing*, Journal of Finance, and of Sialm & Starks (2012), *Mutual
  Fund Tax Clienteles*, Journal of Finance. This is a tax-return mechanic, off-tape, and
  explicitly not modelled here.

## Measurement method

- **Total return minus price return = distribution.** The identity behind the estimator.
  See Ibbotson & Sinquefield (1976), *Stocks, Bonds, Bills and Inflation*, Journal of
  Business, for the canonical decomposition of a total-return index into a price index
  and a reinvested-income index — [`strategy.income_yield_daily`](../withholding/strategy.py).
- **Fund distributions are paid net of expenses**, so a fee difference is
  observationally identical to an income difference. Sharpe (1966) and, for funds
  specifically, Carhart (1997), *On Persistence in Mutual Fund Performance*, Journal of
  Finance — expenses come out of the same pot. This is why the fee-adjusted gap is
  reported beside the raw gap.
- **Fee compression is an anachronism risk.** `EXPENSE_RATIO` holds *2026* fact-sheet
  fees applied to a window opening in 2007 (2001 for EFA), and broad-index ETF fees fell
  sharply over it. The time-averaged fee gap is therefore smaller than the 47 bp the
  headline adds back, making the fee-adjusted gap an **upper bound** — generous to the
  withholding hypothesis, which needs a positive gap. Swept 35–55 bp in
  [`strategy.fee_gap_sweep`](../withholding/strategy.py) and reported in `docs/results.md`.
- **Tracking difference vs tracking error.** Elton, Gruber, Comer & Li (2002),
  *Spiders: Where Are the Bugs?*, Journal of Business — an ETF's shortfall against its
  index decomposes into fees, cash drag, securities lending and tax, and the components
  are not separately identified from price data alone. That decomposition failure is
  precisely this study's finding.

## Related desk studies (dedup)

- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  whether last year's best index tracker stays best. Same family (wrapper mechanics),
  but it races *total returns* between trackers of one index; 916 opens the total return
  up and measures only the **income** leg, on funds tracking a *foreign* market.
- **[Study 914 — Securities-Lending Offset](../../914-sec-lending-offset/)**: another
  hidden fund-level cash flow (borrow revenue) that shows up inside the same tracking
  difference. 914 asks whether the fund keeps it; 916 asks whether a *tax* deducted
  before the fund's NAV can be seen at all. Complementary halves of the same wedge.
- **[Study 915 — K-1 vs 1099](../../915-k1-vs-1099-structure/)**: a wrapper-choice tax
  question on commodity funds — a reporting/structure difference between two live funds,
  which *is* identified by racing them. 916 is the case where the analogous race fails,
  because both wrappers sit on the same side of the tax.
- **[Study 516 — Dividend-Month Premium](../../516-dividend-month-premium/)** and
  **[Study 143 — Dividend Capture](../../143-dividend-capture/)**: trading *around*
  ex-dates. 916 does not trade ex-dates at all — it uses them only as the sessions on
  which the income measurement is non-zero.
- **[Study 568 — Effective Tax Rate](../../568-effective-tax-rate/)** and
  **[Study 599 — Tax-Loss Harvesting](../../599-tax-loss-harvesting/)**: tax as a
  *corporate* characteristic and as an *investor* action respectively; 916 is tax as a
  **fund-level leak**, a third and distinct place tax enters a return series.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../withholding/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1992), *A Circular Block-Resampling
  Procedure for Stationary Data* (in *Exploring the Limits of Bootstrap*) — the
  fixed-length circular variant actually implemented here; see also Politis & Romano
  (1994), *The Stationary Bootstrap*, JASA, for the random-length relative. Blocks here
  are one quarter (63 sessions) so each resampled block carries whole ex-dividend
  events — [`strategy.block_bootstrap_mean_ci`](../withholding/strategy.py).
- **Return-difference (Sharpe comparison) t-stat.** Jobson & Korkie (1981),
  *Performance Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of
  Finance — the excess-of-cash race in
  [`strategy.total_return_race`](../withholding/strategy.py).
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — as-of slice
  plus content fingerprint on every headline run.

## Data sources

- **VEA, IEFA, EFA, VXUS** (broad developed / total-ex-US), **EWJ, EWG, EWU**
  (single-country iShares MSCI funds for Japan, Germany and the United Kingdom), **BIL**
  (1-3 month T-bills, the cash leg) — two legs each from `yfinance`: daily
  **total-return** closes (`auto_adjust=True`, cached as `prices_<TICKER>_1d.parquet`)
  and daily **price-only** closes with declared per-share cash distributions
  (`auto_adjust=False, actions=True`, cached as `divs_<TICKER>_1d.parquet`). Both live
  in the shared desk cache `studies/_cache`.
- **Survivorship.** None of the seven funds has ever been liquidated or merged, and the
  study makes no cross-sectional selection — but the universe *is* chosen ex-post from
  today's largest surviving international ETFs, which is a mild survivorship tilt on the
  Signal axis and is named as such.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  Each fund is measured over its own inception-to-as-of window; the headline VEA
  comparison uses the VEA∩EWJ∩EWU∩EWG common window from 2007-07-30.
- **Non-tape inputs** — expense ratios (2026 fund fact sheets, a named anachronism),
  blend country weights (MSCI EAFE country shares) and the effective withholding rate
  (treaty schedules plus UK domestic law) are **assumptions**, labelled in
  `docs/results.md` and each swept there.
