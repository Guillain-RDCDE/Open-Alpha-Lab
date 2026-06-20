# References & literature map — Study 340 (Bank-Loans)

## The claim under test

**Senior bank loans** — leveraged loans, and the ETFs that hold them (BKLN, SRLN, FLOT-style
floating-rate funds) — are sold as a *rate-proof* high-yield sleeve: because the coupon
**floats** with a short-term reference rate (LIBOR, now SOFR), the loan's price barely moves
when rates rise, so you get a fat yield with **almost no interest-rate (duration) risk**.
The pitch peaks in rate-hiking cycles: *"rotate out of duration into floating-rate loans."*

- Invesco's positioning for **BKLN** (*Invesco Senior Loan ETF*, launched 2011-03-03) and
  State Street / SPDR's **SRLN** describe senior loans as low-duration, senior-secured,
  higher-yielding instruments — a complement that "may benefit when rates rise."
- The "floating rate = no rate risk" framing is ubiquitous in income-investing media,
  advisor decks, and the 2022 rate-hike marketing wave.

## Why the steelman is genuinely correct (on its narrow point)

- **Floating coupons reset.** A leveraged loan's coupon = reference rate + a fixed spread,
  resetting every ~1–3 months, so its effective duration is near zero. Mechanically, the
  price is far less sensitive to the level of rates than a fixed-coupon bond.
- **Seniority and security are real.** Bank loans sit senior-secured in the capital
  structure, ahead of high-yield bonds, with historically higher recovery rates.
- **The 2022–2023 record vindicates the narrow claim.** While long Treasuries had a
  historic drawdown, floating-rate loan funds held their value — the "rate protection"
  worked exactly as advertised.

## Why it fails *as a "safe alternative to bonds"*

- **Floating-rate loans bear credit risk, not duration risk** — and the two are not
  interchangeable. Leveraged loans are sub-investment-grade corporate credit; their price is
  driven by *default/spread* risk, which spikes in recessions and risk-off shocks — the same
  moments a real bond (duration) would *help*. The literature treats leveraged loans as a
  credit/equity-correlated asset (see work on leveraged-loan and CLO risk; e.g. IMF and BIS
  financial-stability notes on the leveraged-loan market, 2019–2020).
- **Liquidity is the hidden risk.** The loan market is OTC and settles slowly (T+weeks);
  loan ETFs trade intraday on top of an illiquid underlying, so in a stress event the ETF
  can gap to a discount. BKLN lost ~24% in March 2020 — a liquidity, not a rate, event.
- **It is high yield wearing a low-duration label.** The fat coupon is compensation for
  credit and liquidity risk; it is not a free premium for giving up duration.

## Method lineage

- **Univariate / downside beta.** Conditioning beta on down-market days follows the
  downside-risk tradition (Bawa & Lindenberg 1977; Ang, Chen & Xing, *Downside Risk*, RFS
  2006) — the relevant measure when the claim is about behaviour *in stress*.
- **Newey–West HAC standard errors** for the mean of an autocorrelated influence series
  (here, the OLS-beta influence function): Newey & West (1987), Econometrica.
- **Circular block bootstrap** for a CI on a beta *difference* — block resampling preserves
  volatility clustering and cross-asset co-movement that i.i.d. resampling destroys
  (Politis & Romano, 1994).
- **Total-return adjustment.** For an income instrument whose return is mostly coupon, the
  fair series is dividend-and-split adjusted (`yfinance auto_adjust=True`); a price-only
  series would understate the loan's return and is *not* used.

## Data sources used

- **BKLN** (Invesco Senior Loan ETF), **TLT** (20y+ Treasuries), **IEF** (7-10y Treasuries),
  **SPY** (equity), daily, **total-return adjusted** via `quantlab.data` (yfinance
  `auto_adjust=True`); TLT/IEF/SPY from the shared cross-asset cache, BKLN cached
  study-local. BKLN lists **2011-03-03**, which bounds the joint window honestly — stated as
  a decision, not buried.

## Related desk studies

- [Study 338 — Preferred-Stocks](../../338-preferred-stocks/) — the same "is this asset what
  the brochure says?" lens on a perpetual junior equity-hybrid. **Distinct**: preferreds are
  a *duration + equity-tail* hybrid sold as safe; bank loans are a *zero-duration + credit*
  hybrid sold as rate-proof — opposite duration profiles, same credit-risk catch.
- [Study 97 — Balancing-Act](../../97-balancing-act/) — the fixed 60/40 stock/bond blend.
  **Distinct**: 97 is an *allocation* race; 340 is a single-instrument *duration-vs-credit
  identity* test.
- [Study 152 — Inflation-Hedge](../../152-inflation-hedge/) and other "does this asset do
  what the label promises?" teardowns.
