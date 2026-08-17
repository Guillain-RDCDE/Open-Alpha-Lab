# References & literature map — Study 931 (the CEF IPO hole)

## The claim under test

- **The mechanics of a closed-end-fund IPO.** A US closed-end fund is sold at a round
  offering price — $20.00 for equity and multi-sector credit funds, $25.00 for
  preferred-share funds — and the underwriting syndicate's compensation (typically 3-6% of
  the offering price, disclosed in the prospectus as a sales load) is deducted from the
  proceeds. The fund therefore begins life with a net asset value several percent *below*
  what the subscriber paid. Unlike an operating-company IPO, there is no growth story to
  reprice: the fund is a basket of the same securities anyone can buy directly. The
  syndicate stabilises the price for the first weeks; when that support ends, the fund
  drifts toward the discount at which seasoned closed-end funds trade. The claim we test:
  the subscriber's money goes into a measurable hole, and the tradable read is the mirror —
  buy closed-end funds seasoned, never at issue.
- **The steelman.** The load buys distribution, and a fund launched into a dislocated asset
  class could in principle earn it back. If the drift were merely the asset class doing
  badly, an asset-class benchmark would absorb it; if it were merely that these are poor
  funds, a placebo drawn from their seasoned life would show the same damage. This study
  runs both controls.

## Why the hole should exist — the mechanism

- **Weiss (1989), *Closed-End Fund IPOs: Sold, Not Bought*, and Peavy (1990), *Returns on
  Initial Public Offerings of Closed-End Funds*, Review of Financial Studies 3(4).** The
  founding results: CEF IPOs trade near the offering price for roughly the first 20-30 days
  and then fall by around 10% relative to the market over the following months. Our
  1-month/3-month split (*t* = −1.34 then −3.94) reproduces the stabilisation window and the
  drop with 2012-2022 vintages the original papers could not see.
- **Weiss Hanley, Lee & Seguin (1996), *The Marketing of Closed-End Fund IPOs*, Journal of
  Financial Intermediation.** Documents the underwriters' price-stabilising activity in the
  aftermarket and its expiry — the direct explanation for why month one is quiet and month
  three is not.
- **Lee, Shleifer & Thaler (1991), *Investor Sentiment and the Closed-End Fund Puzzle*,
  Journal of Finance 46(1).** The classic sentiment framing: CEFs are issued when investor
  demand for the sleeve is hot and then converge to the standard discount. The IPO wave is
  the sentiment peak; the decay is the convergence.
- **Cherkes, Sagi & Stanton (2009), *A Liquidity-Based Theory of Closed-End Funds*, Review
  of Financial Studies.** The rational counterweight: a CEF can be worth a premium at issue
  if it delivers liquidity transformation the investor cannot get otherwise — which is why
  we measure the fund against the *asset class* rather than against cash, and why the size
  of the hole (not merely its sign) is the object of interest.
- **Ritter (1991), *The Long-Run Performance of Initial Public Offerings*, Journal of
  Finance.** The generic long-run IPO underperformance result. The CEF case is the clean
  laboratory version of it: no operating business, no earnings surprises, a benchmark that
  holds the same securities.

## Why the measurement could mislead — and what we did about it

- **Overlapping event windows.** Twenty-eight funds with staggered but partly clustered
  vintages are not 28 independent draws. We report the one-sample cross-sectional *t*, a
  **vintage-cluster bootstrap** (whole IPO years resampled), and a **calendar-time
  portfolio** with a Newey-West *t* — the last following Fama (1998), *Market Efficiency,
  Long-Term Returns, and Behavioral Finance*, Journal of Financial Economics, which argues
  precisely that calendar-time portfolios are the dependence-robust way to aggregate
  overlapping long-horizon event windows.
- **Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica** —
  [`strategy.newey_west_t`](../cef_ipo/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Politis & Romano (1994), *The Stationary Bootstrap*, JASA** — the circular block
  bootstrap in [`strategy.block_bootstrap_mean_ci`](../cef_ipo/strategy.py) and
  [`quantlab.stats`](../../../quantlab/stats.py).
- **Wilson (1927), *Probable Inference, the Law of Succession, and Statistical Inference*,
  JASA** — the interval on the share-of-funds-negative statistic.
- **Beta, not just the benchmark.** Subtracting *one* unit of benchmark assumes β = 1 from a
  vehicle that is typically 25-40% leveraged. **Brown & Warner (1985), *Using Daily Stock
  Returns: The Case of Event Studies*, Journal of Financial Economics** is the standard
  answer — a market-model CAR — and [`strategy.beta_adjusted_table`](../cef_ipo/strategy.py)
  runs it with each fund's own β fitted on its *seasoned* window. Mean fitted β is 0.90 and
  the hole survives (−8.40% → −7.34%). Note what that estimator is and is not: the β comes
  from data *after* the window it corrects, so it is an ex-post decomposition, never a
  tradable weight, and it never touches the mirror trade.
- **Universe construction.** There is no free, complete listing-date screen of US closed-end
  fund IPOs. The 28 names here are a **hand-built convenience sample** of large, well-known
  launches that Yahoo still carries — several hundred CEFs came to market in the window, and
  the 2017 and 2018 vintages contribute nothing at all. Whether the sample is representative
  cannot be established from inside the study; it is named on the Signal axis, and it is the
  single largest reason to treat the *magnitude* here as indicative rather than settled.
- **Survivorship.** The list contains funds still trading under their IPO ticker; BIGZ
  (2021 vintage) left the tape when it was converted to an ETF in 2025, and XFLT (2017) was
  dropped because Yahoo's split factors for it are visibly corrupt. Because the funds
  that fare worst are the ones most likely to be merged, converted or wound up, this biases
  the measured hole *toward zero* — named on the Signal axis rather than buried here.

## Related desk studies (dedup)

- **[Study 367 — CEF Discount](../../367-closed-end-fund-discount/)**: the *cross-sectional*
  discount sort on **seasoned** funds (buy the widest discount, short the narrowest). Study
  931 is the other end of a fund's life — the *event* that creates the discount in the first
  place — and is a time-since-issue effect, not a cross-sectional one.
- **[Study 910 — Managed-Distribution CEF](../../910-managed-distribution-cef/)**: whether a
  seasoned CEF's payout beats its asset class. Same instruments, opposite question: 910 asks
  about the steady state, 931 about the first twelve months.
- **[Study 616 — Muni CEF Tax-Loss](../../616-muni-cef-tax-loss/)**: a *calendar* (December
  tax-loss) effect in municipal CEFs — a seasonal, not an issuance, effect.
- **[Study 929 — Rights-Offering Discount](../../929-rights-offering-discount/)**: the
  *other* moment a closed-end fund sells shares to its holders. 931 is the first sale, 929
  a later one.
- **[Study 378 — ETF NAV Premium](../../378-etf-nav-premium/)**: premium/discount dynamics in
  *open-end* ETFs, where creation/redemption arbitrage keeps the gap near zero — the
  structural contrast that explains why a closed-end fund's discount can persist at all.
- **IPO studies on operating companies — [219 — IPO Pop](../../219-ipo-pop/),
  [623 — IPO Long-Run Underperformance](../../623-ipo-long-run-underperformance/),
  [545 — IPO Birthday](../../545-ipo-birthday/), [874 — IPO Anchoring](../../874-ipo-anchoring/),
  [783 — IPO Deal of the Year](../../783-ipo-deal-of-year/)** — all about companies, where
  underperformance is confounded with business fundamentals. A closed-end fund has no
  business: the benchmark holds the same securities the fund does, which is what makes this
  the clean version of the experiment.

## Data sources

- **28 closed-end funds** (2012-2022 vintages: PDI, LDP, BGB, ARDC, DFP, KIO, THQ, ECC, BST,
  THW, RIV, OPP, FINS, BSTZ, EIC, AIO, BMEZ, FTHY, ASGI, BCAT, PTA, PDO, NBXG, WDI, ECAT,
  MEGI, NPFD, PAXS) and **eight benchmark ETFs** (XLK, XLV, IGF, AOR, BKLN, HYG, LQD, PFF)
  plus **BIL** — daily **total-return** closes via `yfinance` (`auto_adjust=True`).
  Distributions are reinvested on both legs; a price-only comparison would manufacture a
  fake decay out of a CEF's 8-12% distribution rate.
- **Unadjusted closes** for the funds only, cached under the `rawclose_` prefix, used solely
  to measure the day-one gap between the assumed offering price and the first traded close.
- **As-of 2026-06-30**, the last complete calendar month; the partial current month is
  dropped so the sample never creeps between reruns.
