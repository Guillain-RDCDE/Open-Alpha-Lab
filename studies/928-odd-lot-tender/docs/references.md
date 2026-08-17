# References & literature map — Study 928 (Odd-Lot Priority)

## The claim under test

- **The folk trade.** A US issuer running a fixed-price or modified Dutch auction
  self-tender almost always includes an **odd-lot priority** clause: a holder of fewer
  than 100 shares who tenders *all* of them is purchased **first and in full**, ahead of
  the proration that hits every round-lot holder. The retail-forum conclusion is that this
  is a free arbitrage — buy 99 shares, tender them, collect the premium.
- **Where the clause comes from.** Odd-lot priority is not a courtesy: it is written into
  the offer to buy out small holders cheaply (their registrar and mailing costs exceed
  their economic weight) and it is expressly contemplated by the SEC's tender-offer rules.
  Rule 13e-4 (issuer tender offers) and Regulation 14E set the mechanics: **Rule 14e-1(a)**
  requires the offer to stay open at least **20 business days**, and **Rule 13e-4(f)(3)**
  requires *pro rata* purchase of over-tendered shares — the proration the odd-lot clause
  exempts you from. The 20-business-day minimum is the source of this study's assumed
  21-trading-day offer window.
- **What the claim needs to be true.** Three things at once: (i) the clearing price must
  exceed the price you can *buy* at after the announcement; (ii) the offer must actually be
  over-subscribed, else proration never bites and priority is worth nothing; (iii) the
  clearing price must exceed the *post-expiry* market price, else the round-lot holder's
  un-filled stub is worth more than the cash the odd-lot holder took. This study measures
  (i) and (iii) on the tape and treats (ii) as a declared, swept assumption.

## The tender-offer literature this study leans on

- **Comment & Jarrell (1991), *The Relative Signalling Power of Dutch-Auction and
  Fixed-Price Self-Tender Offers and Open-Market Share Repurchases*, Journal of Finance
  46(4).** The canonical comparison. Dutch-auction self-tenders are announced at premia in
  the low-to-mid teens and earn announcement abnormal returns around **+8%** (fixed-price
  tenders more). Our assumed 13% clearing premium sits in the middle of their range, and
  our measured **permanent abnormal reprice of +6.2%** is a close match to their
  announcement effect on an independent, much later sample.
- **Bagwell (1992), *Dutch Auction Repurchases: An Analysis of Shareholder Heterogeneity*,
  Journal of Finance 47(1).** The mechanism behind proration: shareholders have upward
  sloping supply curves, so the issuer clears at the top of the range far more often than
  the folklore assumes, and over-subscription (hence proration) is common but not
  universal. This is exactly why the proration factor is swept from 0.15 to 1.00 here.
- **Vermaelen (1981), *Common Stock Repurchases and Market Signalling*, Journal of
  Financial Economics 9(2)** and **Lakonishok & Vermaelen (1990), *Anomalous Price
  Behavior Around Repurchase Tender Offers*, Journal of Finance 45(2).** The founding
  signalling result and the first study of the *post-expiry* price — the leg that decides
  whether odd-lot priority is worth anything. Lakonishok & Vermaelen document that the
  price does **not** collapse back after expiry, which is what our −0.50% (HAC *t* = −1.17)
  post-expiry give-back reproduces on a modern sample.
- **Peyer & Vermaelen (2005), *The Many Facets of Privately Negotiated Stock
  Repurchases*, JFE** and **Peyer & Vermaelen (2009), *The Nature and Persistence of
  Buyback Anomalies*, Review of Financial Studies 22(4).** The long-run drift after
  repurchase announcements — relevant because a *permanent* reprice is the enemy of the
  odd-lot round trip: if the stock keeps the premium, the round-lot holder's stub is worth
  as much as the cash and priority buys nothing.
- **Odd lots and market structure.** O'Hara, Yao & Ye (2014), *What's Not There: Odd Lots
  and Market Data*, Journal of Finance 69(5) — odd lots are not a curiosity of retail
  folklore but a large and informative share of trading. Their point is orthogonal to ours
  (they study odd-lot *trades*, we study an odd-lot *entitlement*), but it is why the "an
  odd lot is a rounding error" intuition is wrong in the tape and right in the wallet.

## Why the folk version over-states the edge

- **The announcement pop is not yours.** The offer is public at the close of day *t*; the
  earliest an outside buyer trades is the close of *t+1*. Our tape says the market has
  already taken **+7.4%** (abnormal **+6.9%**, HAC *t* = +9.1) of a low-teens premium by
  then — and **+9.1%** in the post-2018 era. This is the standard event-study lesson
  (Fama, Fisher, Jensen & Roll, 1969, *The Adjustment of Stock Prices to New Information*,
  International Economic Review 10(1)): by the time the news is public, the price contains
  it.
- **Priority is only worth the gap between the tender price and the post-expiry price.**
  Because the post-expiry give-back is statistically absent on this sample, the value of
  odd-lot priority is essentially *the assumed premium minus the permanent reprice*, not a
  free lunch on the tape. This is why the study refuses to quote "priority is worth +3.4%"
  as a measurement: it is `(1 − f) × [(p_clear − p_post)/p_entry + cost]` with `p_clear`
  assumed, so it moves one-for-one with the premium and crosses zero at 7.4%. The
  literature (Comment-Jarrell, Bagwell) is where the 13% comes from — and literature is
  not evidence on this desk.
- **Flat fees on a capped position.** A broker's voluntary-corporate-action fee is a flat
  dollar charge; the odd-lot rule caps the position at 99 shares. That is a percentage cost
  that rises as the share price falls — the arithmetic that turns a headline percentage
  into a few hundred dollars a year.

## Related desk studies (dedup)

- **[Study 927 — Dutch Auction](../../927-dutch-auction-buyback/)**: the *same* 180-event
  tender list, asking the issuer-signalling question — does a self-tender mark the bottom,
  is there post-announcement drift for someone who buys the *stock*? Study 928 shares its
  sample deliberately and asks a different, mechanical question: what is the **odd-lot
  entitlement** worth to the holder who tenders, and how much of the premium survives the
  announcement pop, proration and a flat broker fee.
- **[Study 931 — CEF IPO Decay](../../931-cef-ipo-decay/)** and
  **[Study 929 — Rights Offering](../../929-rights-offering-discount/)**: other corporate
  actions where a headline discount or premium is quoted to retail; different mechanics
  (underwriting load, subscription rights), different tape.
- **[Study 926 — T+1](../../926-t-plus-one-settlement/)**: settlement plumbing, not an
  entitlement; no event list in common.
- Merger-arbitrage studies on the desk measure a *deal spread* that is contractually
  observable. This one deliberately does not pretend the clearing price is observable — it
  reports a **breakeven premium** instead.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../odd_lot/strategy.py), applied to the **calendar-ordered**
  event series because self-tenders arrive in waves.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA
  — [`strategy.block_bootstrap_mean_ci`](../odd_lot/strategy.py), blocks of 5 consecutive
  *events* rather than days.
- **Wilson score interval** for the hit rate. Wilson (1927), JASA.
- **Calendar-time portfolio** as the deployable check on an event study: Fama (1998),
  *Market Efficiency, Long-Term Returns, and Behavioral Finance*, JFE 49(3) — the standard
  answer to cross-sectional dependence in overlapping event windows.

## Data sources

- **The event list**: one EDGAR full-text search, `q="modified Dutch auction"`,
  `forms=SC TO-I`, 2010–2025, clustered per registrant; the earliest SC TO-I filing date of
  each cluster is the commencement date. Every row carries its SEC accession number.
  Shared verbatim with Study 927.
- **Prices**: daily **total-return** closes via `yfinance` (`auto_adjust=True`) for the 129
  distinct issuer tickers — **128 of which have a recoverable tape** — plus **SPY** (market
  leg) and **BIL** (cash leg), as-of **2026-06-30** with the partial current month dropped.
- **Not from the tape** (declared PROXIES, all swept): the clearing premium, the offer
  length, the round-lot proration factor, the broker corporate-action fee and the assumed
  $5,000 odd-lot position size. The SC TO-I/A amendments and closing 8-Ks that carry the
  true clearing prices and proration factors are not machine-readable at this sample size;
  saying so and sweeping is the honest alternative to inventing them.
