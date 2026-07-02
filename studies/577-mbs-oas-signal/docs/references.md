# References & literature map — Study 577 (MBS-OAS-Signal)

## The claim, at full strength

- **Option-adjusted spread on agency MBS.** The OAS is the constant spread over the Treasury curve
  that, once the embedded **prepayment option** is priced by a term-structure/prepayment model,
  reprices the mortgage cash flows to their market value. It isolates the *risk/liquidity*
  compensation in mortgage bonds from the option value. The canonical tracked series are the **ICE
  BofA US Mortgage-Backed Securities Index OAS** and Bloomberg's MBS OAS — both licensed vendor
  products, which is exactly why this study is synthetic-only.
- **Gabaix, Krishnamurthy & Vigneron (2007)**, *"Limits of Arbitrage: Theory and Evidence from the
  Mortgage-Backed Securities Market."* *Journal of Finance* 62(2). MBS OAS is driven by the
  risk-bearing capacity of specialised mortgage investors — so it moves with intermediary balance
  sheets and spikes in stress, the mechanism behind the "OAS widening = risk-off" folklore.
- **Boyarchenko, Fuster & Lucca (2019)**, *"Understanding Mortgage Spreads."* *Review of Financial
  Studies* 32(10). Decomposes mortgage spreads (OAS included) into prepayment risk, credit/liquidity
  and a residual; shows spreads widen sharply in stress episodes (2008, 2011, 2020).
- **He, Kelly & Manela (2017)**, *"Intermediary Asset Pricing: New Evidence from Many Asset Classes."*
  *Journal of Financial Economics* 126(1). Intermediary capital is a priced state variable across
  MBS, credit and equities — the cross-asset channel a mortgage-spread lead would ride.

## The cross-asset "spread as a leading indicator" tradition

- **Gilchrist & Zakrajšek (2012)**, *"Credit Spreads and Business Cycle Fluctuations."* *American
  Economic Review* 102(4). The excess bond premium (a corporate-spread component) *leads* real
  activity and equity returns — the closest published cousin to the MBS-OAS-leads-risk claim, on the
  corporate side.
- **Adrian, Boyarchenko & Giannone (2019)**, *"Vulnerable Growth."* *American Economic Review*
  109(4). Financial conditions (spreads prominent among them) shift the *downside* of the forward
  return/growth distribution — the risk-off asymmetry the timing overlay exploits.

## Neighbours on this bench (the dedup map)

- **[Study 115 — Credit-Spreads](../../115-credit-spreads/)** — the *corporate* HY/IG spread as a
  state variable. Study 577 is specifically the **agency-MBS option-adjusted spread** as a
  cross-asset *lead*, a different instrument and a different (prepayment-option-adjusted) measure.
- **[Study 05 — Twin-Spread](../../05-twin-spread/)** — a relative-value *pair* trade on two related
  instruments, not a macro risk-off leading indicator.
- **[Study 111 — VIX-Term-Structure](../../111-vix-term-structure/)** /
  **[Study 131 — Utilities-Canary](../../131-utilities-canary/)** — other "canary leads risk-off"
  studies; the MBS-OAS version is the mortgage-market analogue, and shares the synthetic-only wall
  when the canonical series is licensed.

## Shared method

- **OLS predictive regression + t-stat** — the forward-return-on-signal slope whose *sign* is the
  claim (negative = risk-off lead).
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  signal against forward returns and read the slope-*t*'s tail probability.
- **Ornstein-Uhlenbeck mean-reversion with jumps** — the synthetic OAS path (mean-reverting level +
  clustered Poisson stress spikes), a standard spread/rate generator.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥ 2
  on a **real** tape for `REAL`; synthetic-only is capped at `WEAK`), one explicit execution lag,
  costs one-way × NAV, and the data-availability caveat stated on the SIGNAL axis (as in the desk's
  lego-returns / whisky-cask / sneaker-resale studies).
