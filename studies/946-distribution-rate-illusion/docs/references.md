# References & literature map — Study 946 (Distribution is not Return)

## The claim under test

- **The distribution-rate sticker.** Every high-payout ETF is marketed on one number: a
  "distribution rate" or "12-month yield", printed on the fact sheet, aggregated by screeners
  and ranked in every income-investing forum. The implicit reading — the one this study takes
  literally and tests cross-sectionally — is that a bigger sticker means more money: rank the
  income universe by advertised payout and the top of the list should be the better place to
  put a dollar.
- **The arithmetic counter-claim.** A distribution is a transfer, not a return. On the
  ex-date the fund's NAV falls by the amount distributed; a fund can therefore manufacture any
  headline rate it likes by handing back capital (return of capital, ROC) or by
  systematically monetising volatility. If that is all the sticker measures, the payout rank
  should predict a *falling quoted price* one-for-one and predict *total* return not at all.
  That is a sharp, falsifiable cross-sectional prediction — and the one we test.

## Where the mechanism comes from

- **The ex-dividend price drop.** Campbell & Beranek (1955), *Stock Price Behavior on
  Ex-Dividend Dates*, Journal of Finance; Elton & Gruber (1970), *Marginal Stockholder Tax
  Rates and the Clientele Effect*, Review of Economics and Statistics. The price falls by
  (very nearly) the dividend on the ex-date — the founding empirical fact behind "distribution
  is not return". Our price-only versus total-return tapes are the fund-level version of it.
- **Dividend irrelevance.** Miller & Modigliani (1961), *Dividend Policy, Growth, and the
  Valuation of Shares*, Journal of Business. In frictionless markets the split of return into
  payout and price appreciation carries no value information. The income-ETF sticker is a bet
  that investors behave as if it does.
- **Why investors act otherwise.** Shefrin & Statman (1984), *Explaining Investor Preference
  for Cash Dividends*, JFE (mental accounting and self-control); Hartzmark & Solomon (2019),
  *The Dividend Disconnect*, Journal of Finance — investors track price returns separately
  from dividends and systematically misjudge total return as a result. This is the demand-side
  explanation for why a sticker that means nothing sells so well.
- **Free cash flow as a return of capital.** Baker & Wurgler (2004), *A Catering Theory of
  Dividends*, Journal of Finance — sponsors supply the payout characteristic investors are
  currently paying up for. The 2020-2026 derivative-income ETF boom is the cleanest modern
  instance: the payout is engineered, not earned.

## Why a covered-call wrapper's payout is especially not return

- **Option-writing does not create return, it reshapes it.** Whaley (2002), *Return and Risk
  of CBOE Buy Write Monthly Index*, Journal of Derivatives; Israelov & Nielsen (2015),
  *Covered Call Strategies: One Fact and Eight Myths*, Financial Analysts Journal — the
  premium is compensation for a short-volatility, capped-upside exposure, not an income
  stream, and its risk-adjusted advantage largely disappears once the equity beta is taken
  out. Our CAPM control (β = 0.65 on the high-payout leg, α = −5.3 bps/mo, *t* = −0.59) is the
  same conclusion drawn on the fund tape rather than the index.
- **Return of capital in the wrapper.** Investment Company Act §19(a) requires a fund to
  disclose when a distribution includes a return of capital. The 19(a) notice is the
  regulatory admission that a headline "yield" and an economic return are different objects —
  the paperwork behind the number this study reconstructs.

## Related desk studies (dedup)

- **[Study 337 — Covered-Call-ETF](../../337-covered-call-etf/)** races **each fund
  individually against SPY total return** and decomposes its distribution into a
  return-of-capital share. That is a per-fund verdict on five named products. Study 946 is
  the **cross-sectional** question those per-fund races cannot answer: across a whole income
  universe, does the *rank* on payout carry information about the *rank* on subsequent total
  return? 337 says QYLD is a bad buy; 946 says the payout number itself is uninformative
  about which income fund to buy — including when the fat payers happen to win.
- **[Study 910 — Managed-Distribution CEF](../../910-managed-distribution-cef/)** tests the
  **closed-end** wrapper's double-carry story (discount pull plus managed distribution) on a
  hand basket held long, with a CAPM control. It is a hold-the-payout total-return study on
  seven CEFs. Study 946 never holds anything for its headline: it sorts an **ETF**
  cross-section on the payout rate and measures what the *sort* predicts, with the price-only
  leg — the erosion channel 910 could not see, because 910 uses total-return closes only —
  reconstructed and measured directly.
- **[Study 62 — Premium-Seller](../../62-premium-seller/)** races a buy-write fund against
  *its own underlying* on upside/downside capture; **[Study 900 — Quality-Income](../../900-quality-income/)**
  screens dividend payers on quality rather than on payout level. Neither builds the payout
  cross-section or separates the price and total tapes.

## Method lineage

- **Cross-sectional slope averaging.** Fama & MacBeth (1973), *Risk, Return, and Equilibrium:
  Empirical Tests*, Journal of Political Economy — [`strategy.fama_macbeth`](../dist_illusion/strategy.py).
- **HAC / Newey-West *t*.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../dist_illusion/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py). Six lags, because a
  rolling-12-month ranking variable induces exactly this kind of persistence.
- **Moving-block bootstrap.** Künsch (1989), *The Jackknife and the Bootstrap for General
  Stationary Observations*, Annals of Statistics; Politis & Romano (1994), *The Stationary
  Bootstrap*, JASA — [`strategy.block_bootstrap_ci`](../dist_illusion/strategy.py).
- **Beta control.** Sharpe (1964) / Lintner (1965) CAPM in excess-of-cash form —
  [`strategy.capm`](../dist_illusion/strategy.py). Required here because the high-payout
  cohort is structurally lower-beta (0.65 versus 0.80), so a raw spread in a bull decade is a
  beta bet until proven otherwise.

## Data sources, proxies and assumptions

- **Two tapes, both from `yfinance`, daily, 2003-01-02 → 2026-06-30 (as-of 2026-06-30):**
  `auto_adjust=True` gives the **total-return** close (distributions reinvested);
  `auto_adjust=False` gives the **price-only** close (split-adjusted, not
  distribution-adjusted). Every number in this study is labelled with which of the two it
  came from.
- **PROXY — the distribution rate.** Yahoo! does not publish a fund's marketed distribution
  rate, so it is reconstructed as the compounded trailing-12-month product of
  `(1+total)/(1+price) − 1`. It differs from a fact-sheet sticker in three named ways: it is
  *realised trailing* rather than last-payment-annualised; it lumps **capital-gains
  distributions** in with income (PBP's 12.94% last reading is mostly one); and it inherits
  Yahoo!'s adjustment conventions. It is nonetheless the honest cash-out-the-door rate, which
  is the quantity the claim is about — and it reproduces the published stickers to within a
  few tenths of a point across the universe.
- **ASSUMPTION, and a hindsight one — the corporate-action guard.** Fund-months with an
  absolute total return above 0.50 are dropped as unadjusted corporate actions. The filter
  reads the return of the month being predicted, so it removes a fund from the sort formed at
  *t* on the strength of its *t+1* print: a data clean, not a rule a live trader could run,
  and labelled as such in `data.monthly_panel`. Exactly one fund-month fires: NUSI's
  2025-02-18 1-for-2 reverse split, which Yahoo! applied to neither tape. Swept at 0.40 and
  0.50 and **off** (the no-guard column is the live-tradable read: price leg −57.0, *t* =
  −2.79; total leg still a null), and cross-checked by deleting NUSI outright. Neither stamp
  moves.
- **AN IDENTITY, not a third result — the price leg.** The payout is *defined* as the
  total/price gap, so `hml_price ≡ hml_total − hml_payout` (correlation 0.99995 month by
  month). The erosion *t* of −4.53 is therefore the payout-persistence *t* carried through a
  total-return null, and the Signal stamp is claimed on that pair rather than on the erosion
  counted twice. `verify.py` prints the identity check; `tests/test_data.py` asserts it.
- **ASSUMPTIONS — friction.** 5 bps one-way cost × NAV traded (swept 0 → 25) and a short-leg
  borrow fee of 0 → 200 bps/yr (swept; the tape carries no borrow data). Both are labelled
  and neither changes the verdict.
- **Cash and benchmark.** BIL (1-3 month T-bill ETF) total return is the cash leg of every
  excess-of-cash number; SPY total return is the CAPM benchmark.
- **Survivorship, named.** The fifteen funds are those that gathered assets and remained
  quoted through the as-of. Income products that closed are absent, so any positive read on
  the high-payout leg is an upper bound. NUSI stopped trading in July 2026 — one month after
  the as-of — which is a reminder of what the survivor filter removes. A second, smaller
  forward-looking filter sits inside `strategy.sorted_legs`: it ranks only funds that have a
  *next*-month print, so a fund that stops quoting is skipped rather than realised. It pushes
  the same way as the universe choice — the high-payout leg is flattered, not penalised.
- **Reproducibility of the data stamp.** The fingerprints in `docs/results.md` are taken on
  the tapes' **returns**, not their levels. `auto_adjust=True` back-adjusts the entire
  history on every re-fetch (any new distribution rescales every past close), so a level
  fingerprint of the total-return tape drifts without a single return changing. See
  `data.returns_fingerprint`.
