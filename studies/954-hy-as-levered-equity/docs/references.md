# References & literature map — Study 954 (High Yield in Disguise)

## The claim under test

- **"High yield is levered equity wearing a bond costume."** The trade-desk aphorism: a
  junk bond is a senior claim on a leveraged firm, so its payoff is a short put on the
  firm's assets — equity-like in a downturn, capped on the upside — and a high-yield fund
  is therefore a repackaging of equity beta plus interest-rate duration, sold at a bond
  fund's fee. The testable form: fit `w · SPY + (1 − w) · IEF` to a high-yield fund out of
  sample, and ask (a) whether the blend actually replicates it and (b) whether, at the same
  realised volatility, the fund or the blend pays more excess-of-cash return.
- **The steelman for the fund.** High yield is *not* a linear blend: default risk,
  covenant recovery, call features and a shorter duration than IEF all make its residual
  economically real. If that residual is a compensated risk premium — the classic credit
  risk premium — the fund should hold its own on a vol-matched basis even when its R²
  against the blend is low. That is exactly the null this study's synthetic control plants.

## The theory — why credit looks like equity

- **Merton (1974), *On the Pricing of Corporate Debt: The Risk Structure of Interest
  Rates*, Journal of Finance.** The structural model that makes the whole claim precise:
  risky debt = riskless debt minus a put on firm value, so corporate spreads inherit equity
  risk mechanically. The lower the rating, the larger the equity loading — which is why the
  fitted `w` here lands near 0.45 for high yield and would sit far lower for investment grade.
- **Schaefer & Strebulaev (2008), *Structural Models of Credit Risk are Useful: Evidence
  from Hedge Ratios on Corporate Bonds*, Journal of Financial Economics.** Structural models
  predict corporate-bond *hedge ratios* against equity well even when they price spreads
  badly — direct support for the replication design used here (fit the beta, not the level).
- **Kapadia & Pu (2012), *Limited Arbitrage Between Equity and Credit Markets*, Journal of
  Financial Economics.** Documents how far and how long the two markets can come apart —
  the reason the residual in this study is large and persistent rather than noise.

## The empirical counterweight — the residual is real but poorly paid

- **Asness, Israelov & Liew** and the AQR strand on **credit's low realised alpha to
  equity + duration**: once you hedge out the equity and rate loadings, the leftover credit
  exposure has historically delivered little. Asvanunt & Richardson (2017), *The Credit Risk
  Premium*, Journal of Fixed Income, is the careful long-sample version: a positive but
  small credit premium, and one that is largely absorbed by term and equity factors.
- **Ilmanen (2011), *Expected Returns*, Wiley, ch. on credit.** The canonical statement
  that the credit spread is mostly compensation for default losses, illiquidity and
  downgrade risk, leaving a thin residual once the equity-beta and duration components are
  paid for separately — and that the residual is worst-timed (it disappears precisely when
  equity risk is realised).
- **Ng & Phelps (2011), *Capturing Credit Spread Premium*, Financial Analysts Journal.**
  The excess-return-over-Treasuries decomposition that this study performs on ETF total
  returns rather than index data.
- **Fund frictions.** Choi, Hoseinzade, Shin & Tehranian (2020), *Corporate Bond Mutual
  Funds and Asset Fire Sales*, Journal of Financial Economics — the flow-driven costs a
  bond fund bears that an index blend of two mega-liquid equity/Treasury ETFs does not.

## Related desk studies (dedup)

- **[Study 115 — Credit Spreads](../../115-credit-spreads/)**: uses the credit spread as a
  *macro timing signal* for other assets. Study 954 never times anything — it decomposes
  the high-yield fund itself into two tradable legs.
- **[Study 832 — High-Yield Credit Momentum](../../832-high-yield-credit-momentum/)**: a
  *timing rule* inside high yield. This study is a static, held-out replication, not a rule.
- **[Study 865 — Credit-Equity Lead-Lag](../../865-credit-equity-lead-lag/)**: asks whether
  credit moves *before* equity. Study 954 asks whether credit is *the same thing as* a
  mixture of equity and Treasuries, contemporaneously.
- **[Study 907 — Senior Loans vs HY](../../907-senior-loans-vs-hy/)** and
  **[Study 885 — Ultra-Short Credit Pickup](../../885-ultra-short-credit-pickup/)** and
  **[Study 887 — High-Yield Muni Premium](../../887-high-yield-muni-premium/)**: sleeve-vs-sleeve
  races *within* credit. Study 954's benchmark is deliberately *outside* credit entirely.
- **[Study 951 — The Crossover Rung](../../951-crossover-credit-bbb-bb/)**: where on the
  rating ladder the pay-off is best. Study 954 asks whether the ladder is worth climbing at
  all versus two non-credit legs.
- **[Study 953 — Replicating the Convert](../../953-convertible-replication/)**: the same
  replication *method* applied to convertibles (equity + investment-grade credit). Study 954
  is the credit-side sibling and its benchmark contains no credit at all.
- **[Study 945 — The Hidden Financing](../../945-leverage-financing-cost/)**: replication of
  a *leveraged* fund to back out its financing rate. Same held-out-weights machinery, an
  entirely different question and asset.

## Method lineage

- **Constrained (fully funded) replication.** Regressing `r_HY − r_IEF` on `r_SPY − r_IEF`
  imposes weights summing to one by construction — the standard two-asset style-analysis
  trick from Sharpe (1992), *Asset Allocation: Management Style and Performance
  Measurement*, Journal of Portfolio Management —
  [`strategy.rolling_beta`](../hy_replication/strategy.py).
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../hy_replication/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Sharpe-difference test.** Jobson & Korkie (1981), *Performance Hypothesis Testing with
  the Sharpe and Treynor Measures*, Journal of Finance; Ledoit & Wolf (2008), *Robust
  Performance Hypothesis Testing with the Sharpe Ratio*, Journal of Empirical Finance —
  implemented here as the HAC *t* on the vol-matched daily return difference,
  [`strategy.vol_matched_diff`](../hy_replication/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_sharpe_ci`](../hy_replication/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources

- **HYG, JNK, USHY** (high-yield ETFs), **SPY** (equity leg), **IEF** (7-10y Treasury
  duration leg), **SHY / IEI / TLT** (alternative Treasury maturities, used only to sweep
  the duration-leg choice) and **BIL** (1-3M T-bill cash proxy) — daily **total-return** closes via
  `yfinance` (`auto_adjust=True`), 2004 → 2026-06-30. Total return is not optional here: a
  high-yield fund pays out most of its return as coupon, so a price-only tape would show a
  permanently falling line and answer a different question.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  The held-out record starts 2008-06-02, after the first 252-day estimation window on the
  HYG∩SPY∩IEF∩BIL common history (BIL's May-2007 inception gates the cash leg).
- **Non-tape inputs, all labelled PROXY and swept in `docs/results.md`:** the 2 bps one-way
  rebalance cost, the 50 bps/yr borrow spread (inert — the fitted weight never went short),
  the three fixed crisis calendar windows, the 252-day estimation window, the choice of
  **IEF** as the duration leg (swept over SHY / IEI / IEF / TLT — the sign holds at every
  maturity, the |*t*| does not), and the quoted expense ratios used only to decompose the
  measured gap. The vol match behind the headline *t* uses full-sample realised
  volatilities: it is an **ex-post test statistic**, not a portfolio anyone could have run.
