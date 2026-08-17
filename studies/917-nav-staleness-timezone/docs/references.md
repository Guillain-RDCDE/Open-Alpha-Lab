# References & literature map — Study 917 (Stale NAV)

## The claim under test

- **The stale-price thesis.** Tokyo, Frankfurt, Hong Kong, Sydney and London are all shut
  while Wall Street trades. A fund holding those shares is therefore priced off *yesterday's*
  local closes, so a strong US session leaves it "owing" the holder a move — which it should
  pay the next day. The folk version aimed at US-listed country ETFs: after a big up day in
  SPY, buy EWJ/EWG/FXI/EWA/EWU for tomorrow.
- **The steelman.** This is not folklore in origin: it is one of the best-documented
  predictabilities in the mutual-fund literature, it was large, and it was *traded* — until
  the 2003 market-timing scandal and the fair-value-pricing rules that followed. The
  question this study asks is whether any of it is left in the *ETF* wrapper in 2026, when
  the fund itself trades continuously in New York and authorised participants arbitrage it
  against futures round the clock.

## Where the effect comes from — the original evidence

- **Bhargava, Bose & Dubofsky (1998),** *Exploiting International Stock Market Correlations
  with Open-End International Mutual Funds*, Journal of Business Finance & Accounting — the
  early demonstration that lagged US returns predict next-day international fund NAVs.
- **Chalmers, Edelen & Kadlec (2001),** *On the Perils of Financial Intermediaries Setting
  Security Prices: The Mutual Fund Wild Card Option*, Journal of Finance — stale NAVs as a
  free option handed to fast traders by the fund's own pricing convention.
- **Goetzmann, Ivković & Rouwenhorst (2001),** *Day Trading International Mutual Funds:
  Evidence and Policy Solutions*, Journal of Financial and Quantitative Analysis — quantifies
  the timing profits available from exactly this lag, and proposes the fixes (fair-value
  pricing, redemption fees) that were subsequently adopted.
- **Zitzewitz (2003),** *Who Cares About Shareholders? Arbitrage-Proofing Mutual Funds*,
  Journal of Law, Economics & Organization — the dilution cost borne by long-term holders;
  the paper that helped end the practice.
- **Boudoukh, Richardson, Subrahmanyam & Whitelaw (2002),** *Stale Prices and Strategies for
  Trading Mutual Funds*, Financial Analysts Journal — the practitioner statement of the trade
  and of its capacity limits.

## Why it should be dead in an ETF

- **Fair-value pricing.** SEC guidance after 2001 and the post-2003 reforms require funds to
  mark foreign holdings to a *fair value* estimate that already incorporates the US session,
  which removes the very staleness the trade fed on.
- **Continuous secondary-market pricing.** A US-listed country ETF is a *security*, not a
  once-a-day NAV strike: it trades in New York alongside SPY and index futures, so the US
  session is impounded into its price **as it happens**. Anything left to "catch up" must
  survive the ETF's own arbitrage, not merely the fund's accounting.
- **Engle & Sarkar (2006),** *Premiums-Discounts and Exchange Traded Funds*, Journal of
  Derivatives — international ETFs carry visibly larger and more persistent
  premium/discount deviations than domestic ones, which is where any residual would live.
- **Levy & Lieberman (2013),** *Overreaction of Country ETFs to US Market Returns:
  Intraday vs Daily Horizons and the Role of Synchronized Trading*, Journal of Banking &
  Finance — the key modern result and the one this study's tape agrees with: when the local
  market is **closed**, country ETFs *over*-react to the US session and partially reverse
  afterwards. Over-reaction and reversal, not staleness and catch-up.

## The confound this study insists on

- **Short-horizon reversal in the US market itself.** Lehmann (1990), *Fads, Martingales and
  Market Efficiency*, Quarterly Journal of Economics, and Lo & MacKinlay (1990), *When Are
  Contrarian Profits Due to Stock Market Overreaction?*, Review of Financial Studies. A
  country ETF with a US beta near 1 inherits any daily reversal in SPY mechanically. That is
  why the headline table reports the **domestic control** (SPY regressed on itself) and a
  **net-of-SPY** slope — without them, the market's own reversal masquerades as a timezone
  effect. Multiple testing across five funds is handled with a Bonferroni bar of |*t*| ≥ 2.58.

## Related desk studies (dedup)

- **[Study 379 — ETF Lead-Lag](../../379-etf-lead-lag/)**: does an ETF lead its *own member
  stocks* intraday — a within-market microstructure question, same session, no closed market.
- **[Study 865 — Credit → Equity Lead-Lag](../../865-credit-equity-lead-lag/)** and
  **[Study 870 — Industry-Leader Lead-Lag](../../870-industry-leader-lead-lag/)**: lead-lag
  across *asset classes* and *within a sector*, both US-hours; neither turns on a market
  being shut.
- **[Study 01 — Overnight Anomaly](../../01-overnight-anomaly/)** and
  **[Study 788 — Overnight/Intraday Tug-of-War](../../788-overnight-intraday-tug-of-war/)**:
  decompose one instrument's own session into overnight and intraday legs. Study 917 does not
  decompose a session — it asks whether *one market's finished session* predicts *another
  market's next session* through a US-listed wrapper.
- **[Study 146 — Country Momentum](../../146-country-momentum/)**: multi-month trend rotation
  across the same country ETFs — a slow cross-sectional signal, not a one-day spillover.
- **[Study 613 — Currency-Hedged ETF Carry](../../613-currency-hedged-etf-carry/)** and
  **[Study 916 — Withholding Drag](../../916-withholding-drag-international/)**: the *level*
  costs and identities of owning a foreign market through a US wrapper (hedge carry, dividend
  withholding). Study 917 is about the *timing* of the wrapper's price, not its level.

## Method lineage

- **HAC / Newey-West standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica — [`strategy.ols_hac`](../stale_nav/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_mean_ci`](../stale_nav/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  slice and content fingerprint carried at the top of `docs/results.md`.

## Data sources

- **SPY** (US benchmark), **EWJ, EWG, FXI, EWA, EWU** (single-country iShares ETFs whose home
  markets are closed during the US session), **BIL** and **`^IRX`** (cash legs) — daily
  **total-return** closes via `yfinance` (`auto_adjust=True`), through **2026-06-30**.
- **`^IRX` is a PROXY**: it is the 13-week Treasury bill *discount rate* index, compounded
  daily and lagged one day, not an investable fund. It buys the long 1996-start sample; the
  headline is re-run on **BIL**'s actual total return over 2007+ as the tradable check.
- **Cost and borrow are ASSUMPTIONS**, not tape: 10 bps one-way × NAV as the baseline for a
  single-country ETF (wider than SPY's ~1 bp), swept 0–50 bps; borrow on the short mirror
  swept 0–5%/yr. Both sweeps are printed in `docs/results.md` rather than buried. Costs are
  **one-sided** — the rule pays on every position change, buy-and-hold pays nothing — an
  asymmetry that runs *against* the claim and is therefore left in rather than tuned away.
- **The unit hedge is an ASSUMPTION.** The headline "net of SPY" column subtracts one unit
  of SPY, i.e. assumes each fund's contemporaneous US beta is 1. It is not (0.75–1.16), so
  `docs/results.md` §1a re-runs every slope at a **fitted** beta. That fitted beta is
  estimated in-sample and applied in-sample — a **diagnostic, not a tradable rule**, and it
  stamps no badge; it is reported because it is load-bearing (it moves EWJ from *t* = −1.51
  to −3.19, still negative). The verdict is the same under both hedges.
- **Survivorship:** all six tickers still trade; no closed country fund enters the panel,
  and the five funds are the large, long-lived survivors of the single-country ETF cohort —
  an ex-post universe choice that flatters the tape rather than the null being reported.
