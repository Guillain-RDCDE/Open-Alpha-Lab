# References & literature map — Study 929 (Rights Offering)

## The claim under test

- **The folk claim.** A rights offering lets existing holders buy new shares at a price
  well below the market — often 20% or more below for the small closed-end funds and
  BDCs that dominate US rights activity. Retail commentary reads that spread as a *gift*:
  buy the stock before the record date, subscribe cheaply, bank the discount. The
  competing reading, straight from corporate finance, is that the discount is a *warning*:
  the size of the discount is chosen to guarantee take-up, it transfers nothing to
  subscribers as a group, and an issuer selling equity at all is telling you something
  about its own view of the price.
- **Why the discount is (mostly) an illusion.** Because rights are distributed *pro rata*
  to existing holders, a shareholder who subscribes in full is unaffected by the size of
  the discount: the theoretical ex-rights price falls by exactly the value of the right
  received. The discount is a bookkeeping choice, not a wealth transfer — it only bites
  the holder who neither subscribes nor sells the right. See Smith (1977) and any
  standard treatment (Brealey, Myers & Allen, *Principles of Corporate Finance*, ch. 15).
- **The steelman.** If the discount conveys information (a deeper discount signals a
  weaker issuer or a more desperate raise), one should see a cross-sectional gradient:
  deeper-discount deals should drift differently after the deal is done. That is a
  testable claim, and this study tests it directly.

## The literature

- **Smith, C. W. (1977), *Alternative Methods for Raising Capital: Rights versus
  Underwritten Offerings*, Journal of Financial Economics 5.** The classic statement of
  why the subscription discount is irrelevant to a subscribing shareholder, and of the
  "rights offering paradox" — US issuers overwhelmingly abandoned rights for
  underwritten deals despite rights being cheaper.
- **Asquith, P. & Mullins, D. (1986), *Equity Issues and Offering Dilution*, JFE 15;
  Masulis, R. & Korwar, A. (1986), *Seasoned Equity Offerings*, JFE 15.** The canonical
  negative announcement effect of a seasoned equity issue (about −3% for US industrials).
  Our announcement window is +0.27% (*t* = +0.34) — the negative announcement effect does
  **not** appear on this closed-end-fund-heavy list, which is consistent with the
  adverse-selection channel being weak when the issuer's assets are marked-to-market
  securities rather than opaque real assets.
- **Myers, S. & Majluf, N. (1984), *Corporate Financing and Investment Decisions When
  Firms Have Information That Investors Do Not Have*, JFE 13.** The information-asymmetry
  reason a seasoned issue is bad news — and, by the same token, the reason the effect
  should be muted for a transparent portfolio vehicle.
- **Loughran, T. & Ritter, J. (1995), *The New Issues Puzzle*, Journal of Finance 50;
  Spiess, D. & Affleck-Graves, J. (1995), JFE 38.** Long-run under-performance after
  seasoned equity issues. The desk's Study 519 and Study 790 test the modern
  net-share-issuance form of this on a broad cross-section; Study 929 asks the much
  narrower question of what happens in the *weeks* around a rights deal.
- **Eckbo, B. E. & Masulis, R. (1992), *Adverse Selection and the Rights Offer Paradox*,
  JFE 32.** Why the observed announcement reaction to a rights offer differs from an
  underwritten offer, and why standby arrangements matter — a reminder that "rights
  offering" is not one homogeneous event type, which is a real limitation of a
  39-deal hand-compiled list.
- **Cherkes, M., Sagi, J. & Stanton, R. (2009), *A Liquidity-Based Theory of Closed-End
  Funds*, Review of Financial Studies 22.** Why closed-end funds behave differently from
  operating companies around issuance, and why premium-to-NAV funds (Cornerstone being
  the archetype) can rationally keep issuing.
- **Brown, S. & Warner, J. (1985), *Using Daily Stock Returns: The Case of Event
  Studies*, JFE 14.** The market-model event-study specification used here, and the
  warning this study takes seriously: **cross-sectional dependence** (event-date and
  issuer clustering) invalidates the naive cross-sectional *t*. Our placebos are the
  empirical version of that correction — note that only the *clustered* one (sliding the
  whole list by a common offset) actually preserves the dependence being corrected for;
  independent random anchors correct for something else, namely the fat tails of a
  28-day window in this universe.
- **Kolari, J. & Pynnönen, S. (2010), *Event Study Testing with Cross-Sectional
  Correlation of Abnormal Returns*, RFS 23.** Quantifies how badly clustered event dates
  inflate the standard test. Our −2.03 subscription *t* becomes *z* = −1.60 (*p* = 0.09)
  under the **era-matched** placebo — the fair one — and *z* ≈ −0.9 under the whole-tape
  and clustered versions, whose draw range includes crash regimes these deals never saw.
  The honest reading is that the resampled *t* is materially weaker than the parametric
  one, not that the drift is annihilated; what annihilates it is the issuer jackknife,
  the era cut, the timetable and the anchor jitter.

## Related desk studies (dedup)

- **[Study 563 — Secondary-Offering Drift](../../563-secondary-offering-drift/)**: drift
  after *underwritten follow-on* offerings — a different security-issuance mechanic
  (marketed deal, no pro-rata rights, no subscription discount to test).
- **[Study 519 — Net Share Issuance](../../519-net-share-issuance/)** and
  **[Study 790 — Composite Equity Issuance](../../790-composite-equity-issuance/)**:
  the *annual, balance-sheet* issuance factor across the whole cross-section. Study 929 is
  an event study around a single, dated, deeply discounted deal type, not a slow factor.
- **[Study 367 — Closed-End Fund Discount](../../367-closed-end-fund-discount/)** and
  **[Study 910 — Managed-Distribution CEF](../../910-managed-distribution-cef/)**: the CEF
  *discount-to-NAV* and the distribution-policy premium. Those study the standing price
  of a fund; 929 studies what happens when the fund *issues more of itself* at a discount.
- **[Study 927 — Dutch-Auction Buyback](../../927-dutch-auction-buyback/)**: the exact
  mirror image — a pro-rata tender to *retire* shares at a premium. 929 is the pro-rata
  offer to *create* shares at a discount.
- **[Study 569 — SBC Dilution](../../569-sbc-dilution/)**: dilution through
  stock-based compensation — continuous and non-pro-rata, so no subscription right and no
  dated event.
- **[Study 219 — IPO Pop](../../219-ipo-pop/)** and
  **[Study 623 — IPO Long-Run Under-performance](../../623-ipo-long-run-underperformance/)**:
  first-time issuance, where the discount accrues to *new* buyers, not to existing holders.

## Method lineage

- **Market-model abnormal returns**, `(-250, -31)` estimation window — Brown & Warner
  (1985); [`strategy.market_model`](../rights_offering/strategy.py) and
  [`strategy.event_panel`](../rights_offering/strategy.py).
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite …
  Covariance Matrix*, Econometrica — [`strategy.newey_west_t`](../rights_offering/strategy.py)
  and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Return-difference (Sharpe comparison) t-stat.** Jobson & Korkie (1981), Journal of
  Finance — the HAC t on the daily portfolio-minus-benchmark difference in
  [`strategy.tradability`](../rights_offering/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_mean_ci`](../rights_offering/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Jackknife and permutation.** Efron & Tibshirani (1993), *An Introduction to the
  Bootstrap* — [`strategy.jackknife_issuer`](../rights_offering/strategy.py) and
  [`strategy.permutation_discount_test`](../rights_offering/strategy.py).
- **Beta-adjusted alpha with a HAC t.** Jensen (1968), Journal of Finance —
  [`strategy.alpha_vs_market`](../rights_offering/strategy.py). Used because an event
  book that is long-only equity a fraction of the time earns the equity premium it rents;
  its own Sharpe *t* is not evidence of an edge, and the vs-SPY race is
  exposure-mismatched in the opposite direction.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — as-of
  slicing plus a content fingerprint, mirrored by
  [`data.fingerprint`](../rights_offering/data.py).

## Data sources

- **20 issuer tapes + SPY (market) + BIL (cash)** — daily **total-return** closes via
  `yfinance` (`auto_adjust=True`), 2005-01-01 → 2026-06-30, in the shared
  `studies/_cache`. Total return matters enormously here: these funds pay out most of
  their return as (often return-of-capital) distributions, so a price-only series would
  show a false secular decline.
- **The rights-offering list is NOT a vendor feed.** It was compiled by hand from public
  announcements and is accurate to the **month**. There is no free, complete, machine
  readable US rights-offering database; SEC form 424B and N-2 filings carry the deals but
  are not indexed by deal type. This is the study's binding constraint and is treated as
  such throughout: wide windows, an anchor jitter, a timetable sweep, and a verdict that
  claims only what a month-precision list can support.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
