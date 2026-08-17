# References & literature map — Study 933 (Same Issuer, Two Ladders)

## The claim under test

- **The seniority-premium claim.** A retail income investor shopping one issuer's ticker page
  often finds two $25-par instruments side by side: a **preferred** and an exchange-traded
  **baby bond** (retail-denominated notes listed on the NYSE/Nasdaq). The preferred sits
  **below** the note in the capital stack: its coupon is discretionary and, for most
  corporate issuers, non-cumulative; it is usually perpetual, so it has no stated maturity to
  pull it back to par; and in a restructuring it ranks behind every debt claim. Standard risk
  pricing therefore says the preferred must pay a **higher expected return**, and the honest
  question is whether the market actually charges that premium — and how large it is once you
  compare two rungs of the **same** balance sheet rather than two different credits.
- **How big is the step, really?** In practice the listed universe does not offer a clean
  two-rung gap. Across our eight pairs (`data.RUNG_STRUCTURE`, tabulated in `docs/results.md`)
  only **two** put a *senior* note against a perpetual preferred; in four the note is itself a
  **junior subordinated** or **perpetual subordinated** note ranking immediately above the
  preferred, and in two the "preferred" is a **dated term preferred** carrying a mandatory
  redemption date, which makes it behave like the note it is being compared with. The
  question this panel can actually answer is therefore narrower than the claim: *what is one
  rung, sometimes less, worth?* — and it is a constraint of the listed universe, not a
  correctable design choice.
- **Why the same-issuer design matters.** Any preferred-vs-bond comparison across *different*
  issuers confounds seniority with credit quality, sector, and duration. Holding the obligor
  fixed cancels the issuer's spread factor in the difference, so what remains is the price of
  seniority alone. The cost is sample size: only a few dozen US issuers have ever had both
  rungs listed simultaneously, and the overlapping window is short.

## Where the theory comes from

- **Modigliani & Miller (1958), *The Cost of Capital, Corporation Finance and the Theory of
  Investment*, American Economic Review** — the origin of the claim that expected returns rise
  monotonically down the capital structure as claims absorb residual risk first.
- **Merton (1974), *On the Pricing of Corporate Debt: The Risk Structure of Interest Rates*,
  Journal of Finance** — the structural model in which a junior claim is a levered position on
  the same firm value as the senior claim; the two rungs of one balance sheet are, formally,
  different strikes on the same underlying.
- **Black & Cox (1976), *Valuing Corporate Securities: Some Effects of Bond Indenture
  Provisions*, Journal of Finance** — subordination, safety covenants and the mechanics that
  make a junior instrument's payoff genuinely different from a senior one's.
- **Elton, Gruber, Agrawal & Mann (2001), *Explaining the Rate Spread on Corporate Bonds*,
  Journal of Finance** — the classic decomposition showing how little of a credit spread is
  actually default compensation; a useful prior for how small the *pure* seniority component
  of a spread can be.

## Why the premium can fail to show up

- **Duration, not seniority, dominates.** A perpetual preferred and a 30- or 40-year junior
  subordinated note are both long-duration instruments; a $25-par senior note with a 2028
  maturity is not. Between two rungs of one issuer, the difference in **rate** exposure often
  swamps the difference in **credit** seniority — exactly what the 2022 rate shock and the
  2020 COVID crash show on our tape (the junior leg fell *less* in COVID).
- **Retail clientele and segmented pricing.** Nelson (1999) and the wider $25-par literature
  document that exchange-traded fixed income is priced by a retail clientele buying on
  headline coupon, in small size, on wide spreads — a market where relative-value between two
  rungs of one issuer is not obviously arbitraged. That cuts both ways: it is a reason a
  premium *could* persist, and a reason the observed spread is noisy.
- **Call and redemption asymmetry.** Both rungs are callable at par, usually five years after
  issue. Whichever rung is *in the money* to the issuer gets refinanced away — which is a
  selection mechanism, not a return, and it is precisely what a screen on "both rungs still
  listed today" quietly deletes.
- **Rare-event compensation.** The seniority premium is compensation for a default state that
  a four-year sample containing no default cycle is unlikely to observe. A short, quiet sample
  should be expected to under-measure it; our synthetic control is calibrated to show the
  estimator *would* see a +6%/yr premium if one were paid at this sample length.

## Related desk studies (dedup)

- **[Study 338 — Preferred Stocks](../../338-preferred-stocks/)**: PFF as an *asset class*,
  against **equities and Treasuries** — an identity test ("is it a bond or a stock?"). Study
  933 never leaves the issuer: it compares a preferred to **that same issuer's own listed
  debt**, so the equity/bond identity question is held fixed by construction.
- **[Study 909 — Preferred Reset Premium](../../909-preferred-reset-premium/)**: *variable*
  vs *fixed* coupon preferreds — a **coupon-structure** (rate-reset) question **within** the
  preferred sleeve. Study 933 changes the **rung**, not the coupon type.
- **[Study 907 — Senior Loans vs High-Yield](../../907-senior-loans-vs-hy/)**: seniority
  again, but at the **index** level and across **different borrowers** (BKLN vs HYG) — so
  seniority is confounded with the loan and HY universes' different issuer mixes. Study 933
  is the clean version of that experiment: **same obligor, two rungs**.
- **[Study 885 — Ultra-Short Credit Pickup](../../885-ultra-short-credit-pickup/)** and
  **[Study 115 — Credit Spreads](../../115-credit-spreads/)**: the spread as a *level* and as
  a *timing signal*. Study 933 has no timing element at all — it is a static ladder race.
- **[Study 832 — High-Yield Credit Momentum](../../832-high-yield-credit-momentum/)** and
  **[Study 865 — Credit-Equity Lead-Lag](../../865-credit-equity-lead-lag/)**: cross-asset
  *signals*. Orthogonal to a capital-structure level comparison.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../two_ladders/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap** (arms resampled jointly to preserve the near-unit correlation
  between two rungs of one balance sheet). Politis & Romano (1994), *The Stationary
  Bootstrap*, JASA — [`strategy.bootstrap_ci`](../two_ladders/strategy.py),
  [`strategy.bootstrap_sharpe_adv_ci`](../two_ladders/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Return-difference (Sharpe comparison) test.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance — the HAC form
  used on the daily ladder difference.
- **Wilson (1927) score interval** for the pair win share —
  [`strategy.wilson_interval`](../two_ladders/strategy.py).
- **Bid-ask bounce and the microstructure audit.** Roll (1984), *A Simple Implicit Measure of
  the Effective Bid-Ask Spread in an Efficient Market*, Journal of Finance — the negative
  lag-1 autocorrelation induced by transactions alternating between bid and offer, which
  **inflates** measured variance at the daily horizon. Combined with Scholes & Williams
  (1977) and Lo & MacKinlay (1990) on **non-synchronous / stale** trading, this is why the
  study re-states every risk claim at daily, weekly and monthly frequency —
  [`strategy.staleness_table`](../two_ladders/strategy.py),
  [`strategy.vol_ratio_by_frequency`](../two_ladders/strategy.py). On this panel the "junior
  rung is riskier in 6/8 pairs" fact is a daily-horizon artefact: it is 2/8 monthly.
- **Concentration / leave-two-out.** The desk's standard jackknife discipline: a panel result
  must survive the removal of its largest contributors, or it is a statement about those names
  and not about the effect — [`strategy.drop_issuers`](../two_ladders/strategy.py). Here it is
  decisive: without the two distressed obligors both the return spread and the risk step
  invert.
- **The Jensen trap.** The arithmetic mean of daily log-to-simple percentage differences
  exceeds the compounded outcome by roughly sigma^2/2, and the gap explodes on a leg that
  halves and doubles — which is exactly what B. Riley's +19.15%/yr arithmetic spread against a
  −7.71%/yr realised CAGR shows. This is why the headline pairwise number is labelled a
  **statistic, not a portfolio**.

## Data sources, proxies and assumptions

- **Eight issuer pairs**, daily **total-return** closes via `yfinance` (`auto_adjust=True`):
  CMS Energy (CMS-PB / CMSC), Duke Energy (DUK-PA / DUKB), Brookfield Renewable (BEP-PA /
  BEPH), Brookfield Infrastructure (BIP-PA / BIPH), B. Riley Financial (RILYP / RILYZ),
  Oxford Lane Capital (OXLCP / OXLCZ), Eagle Point Credit (ECC-PD / ECCC), Babcock & Wilcox
  (BW-PA / BWNB). Cash leg **BIL**; sector benchmark **PFF**. Total return throughout —
  price-only would be meaningless between two coupon instruments, and it is labelled as such
  wherever a number appears.
- **PROXY — transaction cost.** 25 bps one-way is an assumed half-spread for a $25-par
  exchange-traded instrument, not a measured one; it is swept 5-100 bps in `docs/results.md`
  and barely moves the answer (it hits both ladders alike and cancels in the spread).
- **ASSUMPTION — borrow.** The dollar-neutral leg charges a borrow rate on the short side.
  Neither $25-par preferreds nor baby bonds are general-collateral easy-to-borrow, so the rate
  is swept 0-300 bp/yr rather than asserted.
- **Survivorship, named.** The panel requires *both* rungs to be listed on 2026-06-30, which
  deletes every redeemed note and called preferred. The one dead pair still measurable
  (Sachem Capital, SACH-PA vs SACC, matured 2024-12) has a **−10.63%/yr** spread — the
  opposite sign — so the live screen flatters the junior rung. Named on the Signal axis.
- **Short sample, named.** The common window is **2022-02-02 → 2026-06-30** (4.4 years),
  gated by OXLCZ and BIPI listing in January 2022, and Yahoo's history for most $25-par
  tickers does not reach back further in any case. Two pairs (CMS, Duke) extend to 2019-04 and
  are reported separately. One rate cycle, no default cycle.
- **PROXY — total return on a thin tape.** Yahoo's `auto_adjust` close is a proxy for a clean
  total-return series on $25-par tickers: the distribution adjustment is applied to closes that
  are frequently **stale prints**. Measured stale-day shares run 1.5–15.5% per leg (mean 5.4%
  preferreds / 5.0% baby bonds) with lag-1 autocorrelations to −0.28 — see the microstructure
  table in `docs/results.md`. Named on both axes, and the reason no risk claim rests on daily
  data alone.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
