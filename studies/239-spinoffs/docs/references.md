# References & literature map — Study 239 (Spinoffs)

## The claim under test

- **Cusatis, Miles & Woolridge (1993).** *Restructuring through Spinoffs: The Stock
  Market Evidence* — Journal of Financial Economics 33(3), 293–311.  The foundational
  study: spin-off children and their parents both outperform the market in the 3 years
  following the spin-off, by roughly +25% on average.  The interpretation: spin-offs
  eliminate the conglomerate discount and let both entities be run by focused managers;
  the market under-reacts initially.  This is the primary claim our desk tests.

- **Greenblatt (1997).** *You Can Be a Stock Market Genius* — Simon & Schuster.  The
  popular presentation of the spin-off anomaly: institutional investors who receive
  spin-off shares may be forced sellers (the child is too small for their mandate),
  creating artificial selling pressure and a buying opportunity.  Greenblatt's framework
  is the "folk" version most retail investors have encountered.

## Why the spin-off premium is (almost) coherent

- **Conglomerate discount elimination.** Berger & Ofek (1995), *Diversification's
  Effect on Firm Value* (Journal of Financial Economics), document that diversified
  conglomerates trade at a 13–15% discount to standalone peers; spin-offs close this
  gap.  The closing of the discount provides a structural rationale for post-spin
  outperformance.

- **Forced seller hypothesis.** Greenblatt (1997, above) and Chemmanur & Yan (2004),
  *A Theory of Corporate Spin-offs* (Journal of Financial Economics), formalise the
  idea that institutional investor mandates (minimum market-cap requirements, sector
  constraints) force selling of newly distributed shares regardless of valuation.
  This selling pressure creates a temporary undervaluation.

- **Management incentive alignment.** Hite & Owers (1983), *Security Price Reactions
  Around Corporate Spin-off Announcements* (Journal of Financial Economics), show that
  spin-offs create focused management teams with clearer incentive structures, which
  the market may take time to reflect in prices.

## Evidence that the anomaly is faded or fragile

- **McConnell & Ovtchinnikov (2004).** *Predictability of Long-Run Spinoff Returns*
  — Journal of Investment Management 2(3).  Using a broader sample, they find that
  the spin-off premium documented by Cusatis et al. is concentrated in the first
  12–18 months and is largely explained by size and value factors (Fama-French
  three-factor model).  After factor adjustment, the alpha is substantially smaller.

- **Desai & Jain (1999).** *Firm Performance and Focus: Long-Run Stock Market
  Performance Following Spinoffs* — Journal of Financial Economics 54(1), 75–101.
  They find positive returns but attribute much of the gain to focus-enhancing
  spin-offs (where management cites improved focus as the rationale); non-focus spins
  are weaker.

- **Post-2000 institutional changes.** Index inclusion rules have tightened; many
  institutional mandates now accommodate small-cap inclusions; the forced-selling
  mechanism may be weaker than in the 1980s.  Index fund proliferation means the
  parent's index weight is transferred to the child almost automatically.

- **Selection bias in our study.** The 14 events in `data.SPINOFF_TABLE` are curated
  from widely-reported transactions.  Notorious successes (CARR, CEG, GEV, APTV) and
  notable failures (KVUE, BHF, FOXA) are both over-represented relative to the
  full universe of spin-offs, which includes many small-cap transactions.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica)
  — implemented in [`strategy.hac_tstat`](../spinoffs/strategy.py).

- **Event-study framework.** Fama, Fisher, Jensen & Roll (1969), *The Adjustment of
  Stock Prices to New Information* (International Economic Review) — the foundational
  event-study design we adapt for post-ex-distribution forward returns.

- **Benchmark choice.** We use SPY (the S&P 500 ETF) as the benchmark for simplicity;
  a factor-matched benchmark (size + value + momentum) would be more rigorous and would
  likely reduce the measured alpha.  Fama & French (1993), *Common Risk Factors in the
  Returns on Stocks and Bonds* (Journal of Financial Economics), for the three-factor
  model that properly accounts for the small-cap tilt of newly-spun children.

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`) — split-adjusted auto-adjusted closes
  for each child ticker and SPY.  Ex-distribution dates from public SEC Form 10-12B
  filings and financial press coverage.  Window: 2011-11-17 to 2026-06-16;
  14 events across 13 child tickers.

## Related desk studies

- **[Study 142 — Split-Drift](../../142-split-drift/)**: post-split-effective-date
  drift — the same event-study machinery applied to a different corporate action;
  also finds no post-effective-date signal.
- **[Study 138 — Random-Forest](../../138-random-forest/)**: ML applied to corporate
  events — the limits of small samples are equally relevant there.
- **[Study 119 — Magic-Formula](../../119-magic-formula/)**: fundamental value screen
  that implicitly captures some of the same "unloved asset" theme as the forced-seller
  hypothesis.
