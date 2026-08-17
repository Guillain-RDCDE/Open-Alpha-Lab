# References & literature map — Study 932 (Trust Yield)

## The claim under test

- **The pre-deal SPAC as a T-bill in a box.** A US special-purpose acquisition company
  raises ~$10 a share into a trust invested in short Treasuries, and every public share
  carries a *redemption right*: at the deal vote (or at liquidation, if no deal closes) the
  holder may hand the share back for its pro-rata share of the trust plus accrued interest.
  The right is not contingent on voting against the deal, and it does not require the deal
  to fail. So a pre-deal SPAC quoted **below** trust is a Treasury bought at a discount, and
  that discount, annualised over the time left to the redemption date, is a yield **on top
  of** the bill yield — with the deal upside left over as a free option.
- **The steelman.** This is close to an arithmetic identity, and that is what makes it worth
  testing rather than asserting. The identity only pays if (a) shares actually traded below
  trust, (b) the trust was funded at the assumed level, (c) the deadline held, and (d) the
  redemption could be exercised. Each of those is an empirical question, and (b) turns out
  to be the load-bearing one.

## Where the claim comes from

- **Klausner, Ohlrogge & Ruan (2022), *A Sober Look at SPACs*, Yale Journal on Regulation
  39(1).** The canonical accounting of SPAC economics: dilution from sponsor promote,
  warrants and underwriting means the cash actually delivered per share at merger is far
  below $10, which is precisely *why* the non-redeeming holder loses and the redeeming
  holder does not. Their arithmetic is the source of the "redeem, don't ride" conclusion
  this study mechanises.
- **Gahng, Ritter & Zhang (2023), *SPACs*, Review of Financial Studies 36(9).** The largest
  SPAC return study: from IPO to merger, SPAC *units* earned an annualised return well above
  Treasuries for exactly the reason tested here (trust accretion plus the discount plus the
  warrant), while post-merger de-SPAC shares badly under-performed. Our result is the
  common-share, no-warrant, hold-to-redemption slice of theirs.
- **Ohlrogge (2023), *Why Have SPACs Underperformed?* / SPAC redemption-rate work.** Documents
  redemption rates that ran above 90% for the 2022 vintage — direct evidence that the exit
  modelled here was the exit the marginal holder actually took.
- **Practitioner framing.** The trade was widely marketed in 2022 as "SPAC arbitrage" or
  "a T-bill with a call option attached" (SPAC Research, Boaz Weinstein's Saba and
  Bulldog Investors were among the visible buyers of sub-trust shells). This study is the
  desk's flat, costed, excess-of-cash test of that pitch on the only tape that survives.

## Why it can fail

- **The trust is not $10.** Some 2021-2022 shells were over-funded ($10.10-$10.20) and some
  extension votes *added* to the trust; conversely, franchise tax and dissolution expenses
  are drawn from it. The whole result scales with this number, which is why the study sweeps
  it and reports the $9.90 case as the honest floor.
- **Deadline extension.** The sponsor can (and in 2022-2023 routinely did) call a vote to
  extend. Every extension is also a redemption opportunity — so it shortens the *realised*
  duration rather than lengthening it — but it makes the modelled deadline a fiction and
  changes the trade from a dated bill into an open-ended one.
- **Capacity and the death of the market.** Pre-deal shells are $200-300m trusts whose
  common often traded a few hundred thousand dollars a day. The 2023-2024 SPAC market is a
  handful of extended shells. An identity you cannot size is not an income stream.
- **Survivorship in any reconstructed SPAC tape.** Yahoo carries the pre-deal history forward
  under the *successor* ticker, so a reconstructed list is a list of shells that closed a
  deal and whose successor still trades under an unsplit symbol. Named on the Signal axis.
- **The Mitchell-Pulvino problem.** Mitchell & Pulvino (2001), *Characteristics of Risk and
  Return in Risk Arbitrage*, Journal of Finance — merger-arb returns look like an
  uncorrelated coupon in calm markets and a short put in bad ones. The SPAC version is
  gentler (the trust is Treasuries, not a target's equity) but the shape of the pitch is the
  same, and the mark-to-market book in this study shows the same lumpiness.

## Related desk studies (dedup)

- **[Study 931 — CEF IPO Decay](../../931-cef-ipo-decay/)**: the *other* $10-wrapper study in
  this lot, and the mirror image. There the day-one investor pays an underwriting load into
  a fund that then slides to a discount; here the investor *buys* the discount and is paid
  the NAV back by contract. 931 is about a wrapper that leaks value, 932 about one that
  legally returns it.
- **[Study 929 — Rights Offering Discount](../../929-rights-offering-discount/)** and
  **[Study 928 — Odd-Lot Tender](../../928-odd-lot-tender/)**: the neighbouring
  corporate-action plumbing studies. Both test a *discount granted by an issuer*; this one
  tests a *discount granted by the market* against a contractual put. Different mechanism,
  different tape.
- **[Study 930 — When-Issued Spinoff](../../930-when-issued-spinoff/)**: another hardcoded
  event-calendar study on a temporary two-price window; no trust, no redemption right.
- **[Study 921 — Bill Ladder vs ETF](../../921-bill-ladder-vs-etf/)** and
  **[Study 922 — Floating-Rate Front End](../../922-frn-vs-fixed-front-end/)**: the honest
  benchmarks for anything calling itself a cash substitute. This study's whole claim is a
  spread *over* the leg those two study, so BIL is the yardstick throughout.
- **[Study 610 — Fallen Angels](../../610-fallen-angels-premium/)** and
  **[Study 613 — Currency-Hedged Carry](../../613-currency-hedged-etf-carry/)**: the desk's two
  existing near-identity yields (a forced-seller premium and a covered-interest-parity
  identity). Study 932 belongs to that family — a mechanical return you can derive on paper —
  and inherits its characteristic weakness: the mechanism is sound, the capacity is not.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../trust_yield/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Cluster bootstrap.** Cameron, Gelbach & Miller (2008), *Bootstrap-Based Improvements for
  Inference with Clustered Errors*, Review of Economics and Statistics — the interval this
  study's verdict leans on, resampling whole SPACs rather than overlapping monthly entries:
  [`strategy.cluster_bootstrap_mean`](../trust_yield/strategy.py). The synthetic null shows
  the naive position-level *t* false-firing on 6/16 panels where the cluster CI fires on 1/16.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_sharpe_ci`](../trust_yield/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Overlapping-horizon inference.** Hodrick (1992) / Valkanov (2003) on the over-rejection
  of long-horizon overlapping-return tests — the reason the one-position-per-SPAC cut is
  reported alongside the full monthly panel.

## Data sources

- **31 hardcoded SPACs of the 2019-2022 vintage**, read as **unadjusted daily closes** via
  `yfinance` (`auto_adjust=False`) off each SPAC's *successor* ticker, which is where Yahoo
  carries the pre-deal history. Unadjusted is essential and deliberate: a $10 trust is a
  dollar quantity, and a post-deal reverse split would rescale the pre-deal tape (LCID's
  pre-deal $10 shows up as ~$200 on the adjusted series). Names that did split post-deal are
  therefore absent from the list, which is a documented selection.
- **BIL, SGOV** (total-return cash legs and the trust-accrual proxy), **^IRX** (13-week bill
  discount rate, used for the yield chart only) — `yfinance`, `auto_adjust=True`.
- **The deal-close calendar and the $10.00 trust level are hardcoded assumptions**, not tape.
  Both are swept. The redemption deadline is modelled as the close minus 30 days and swept
  over 15/30/60/90; the trust level over $9.90-$10.20.
- **As-of 2026-06-30.** The last pre-deal window in the list ends 2024-04-10, so the as-of
  binds only the cash legs; the partial current month is dropped either way.
