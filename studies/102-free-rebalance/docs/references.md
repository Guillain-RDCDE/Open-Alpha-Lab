# References & literature map — Study 102 (Free-Rebalance)

## The claim under test

The "rebalancing bonus" — also sold as **volatility harvesting**, the **diversification
return**, or, in its punchiest form, **Shannon's Demon** — is the idea that periodically
rebalancing a fixed-weight portfolio back to its targets is a *free lunch*: it
**mechanically adds return** on top of the risk control, by systematically "selling high
and buying low" across the assets. The strong, sold-at-full-strength version is that
rebalancing pays you to control your risk.

- Investopedia / popular finance framing of the **"rebalancing premium"** and **Shannon's
  Demon**: <https://www.investopedia.com/terms/r/rebalancing.asp>
- The Shannon's-Demon thought experiment (Claude Shannon's lecture, popularised by
  William Poundstone, *Fortune's Formula*, 2005): two assets, one a flat 50% cash, both
  re-set to 50/50 each period, can compound positively even when each leg alone does not.

## Why the steelman is almost coherent

- **Fernholz & Shay (1982), *Stochastic Portfolio Theory and Stock Market Equilibrium*,
  Journal of Finance 37(2).** Introduces the *excess growth rate* of a continuously
  rebalanced portfolio — the formal root of the "diversification return": a fixed-weight
  portfolio's geometric growth exceeds the weighted average of the assets' growth rates by
  half the difference between the average variance and the portfolio variance.
- **Booth & Fama (1992), *Diversification Returns and Asset Contributions*, Financial
  Analysts Journal 48(3).** Names and measures the *diversification return*
  `DR = 0.5 * (Σ wᵢσᵢ² − σ_p²)` — the identity this study computes. It is **≥ 0 by
  construction** because rebalancing reduces portfolio variance.
- **Willenbrock (2011), *Diversification Return, Portfolio Rebalancing, and the
  Commodity Return Puzzle*, Financial Analysts Journal 67(4).** Shows the diversification
  return *is* the rebalancing return and clarifies the baseline against which it is
  measured — the weighted average of constituents, **not** a buy-and-hold drift portfolio.

## Why it is likely to fail *as stated* ("a free lunch that adds return")

- **The identity's baseline is not the honest benchmark.** Booth-Fama's DR compares a
  rebalanced book to the *weighted average of the assets*. A real investor's alternative
  is the **drift (buy-and-hold)** portfolio, which *also* diversifies **and** lets the
  winners run. Measured against drift, the realised bonus can be small or **negative**.
- **Chambers & Zdanowicz (2014), *The Limitations of Diversification Return*, Journal of
  Portfolio Management 40(4).** Argues the diversification return is largely an artefact
  of the variance-vs-geometric-mean relationship and is **not** a source of excess return
  you can reliably bank; it is "return you would have had anyway", re-labelled.
- **Trending markets punish rebalancing.** When one asset persistently out-trends the
  others, rebalancing keeps trimming the winner and topping up the laggard — a drag, not
  a bonus (the negative case our synthetic control plants and the post-2014 sub-period
  shows on the real tape).
- **Costs.** Every rebalance pays a spread/commission on the traded turnover; more
  frequent rebalancing trades more and harvests little extra. Charged one-way × NAV here.

## Method lineage

- **Diversification-return identity** `DR = 0.5 (Σ wᵢσᵢ² − σ_p²)` — Booth & Fama (1992),
  Willenbrock (2011); computed from the annualised covariance of daily total returns.
- **Newey–West HAC standard errors** for the mean of an autocorrelated daily-bonus
  series: Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica.
- **Circular block bootstrap** for a CI on the daily bonus (Politis & Romano, 1992) —
  i.i.d. resampling would destroy the autocorrelation the inference must respect; blocks
  of ~63 days (one quarter) match the rebalancing horizon.

## Data sources used

- **SPY** and **TLT**, daily, **total-return adjusted** (dividends/coupons folded in) via
  `quantlab.data` (Yahoo Finance), cached to parquet under `_cache/`. Total return is the
  fair benchmark because a drift portfolio compounds distributions exactly as a held
  basket does. The aligned window honestly starts at TLT's first date (**2002-07-30**);
  GLD (from 2004-11) is available in the loader for a 3-asset variant.

## Related desk studies

- [Study 68 — All-Weather](../../68-all-weather/) — multi-asset risk balancing; the bar a
  de-risking story must clear, and where risk control *is* the real product.
- [Study 91 — Death-Cross](../../91-death-cross/) — another "controls risk but doesn't add
  return" verdict, and the same alpha-vs-beta discipline.
