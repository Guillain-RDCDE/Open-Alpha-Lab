# References & literature map — Study 842 (Implementation Shortfall)

## The claim under test

- **The source concept.** André F. **Perold (1988)**, *"The Implementation Shortfall: Paper
  versus Reality"* (Journal of Portfolio Management, 14(3)). The founding statement of the gap
  this study is about: the return of the *paper portfolio* (the frictionless backtest, positions
  taken at the decision price) and the return of the *real portfolio* differ by the cost of
  actually getting into the trades — commissions, bid-ask spread, market impact, and the
  opportunity cost of delay. The implementation shortfall is that difference, and it is *not* a
  rounding error: it is often the whole edge. A backtest that reports only the paper number is,
  by Perold's definition, reporting a fiction.
- **The steelman we reproduce.** We build a strategy whose paper performance is genuinely good —
  a moderate-turnover cross-sectional long-short with a *planted* gross edge, so its 0-cost
  Sharpe honestly dazzles — and then charge the friction of trading it. The point is not that the
  signal is fake (it isn't); it is that the tradable alpha is what survives the cost of turning
  the book over, and that survivor shrinks with turnover.

## Transaction costs and market impact — the cost model

- **Almgren & Chriss (2000)**, *"Optimal Execution of Portfolio Transactions"* (Journal of Risk
  3). The canonical decomposition of trading cost into a *temporary* (impact you pay and recover)
  and *permanent* component, and the framework in which impact grows with the **rate** of trading
  — the theoretical basis for a turnover-scaled impact term.
- **Almgren, Thum, Hauptmann & Li (2005)**, *"Direct Estimation of Equity Market Impact"* (Risk
  18). The empirical square-root/participation law: impact rises with the fraction of volume you
  demand (participation). Our impact term is a deliberately simple, monotone participation proxy —
  impact per unit turnover rises with turnover, so the daily drag is super-linear (∝ turnover²) —
  which captures the qualitative fact that trading more costs disproportionately more.
- **Kyle (1985)**, *"Continuous Auctions and Insider Trading"* (Econometrica). The origin of
  *price impact* (Kyle's λ): informed order flow moves prices linearly in size — the micro-
  foundation for why a bigger footprint costs more per share.
- **Frazzini, Israel & Moskowitz (2018)**, *"Trading Costs"* (working paper / AQR). Real-world
  live-trading cost estimates for equity factor strategies, showing that realistic costs erode a
  large share of paper factor returns — and that high-turnover signals suffer most. The empirical
  motivation for grading a factor's *net* Sharpe, not its gross.

## Why turnover is the axis

- **Novy-Marx & Velikov (2016)**, *"A Taxonomy of Anomalies and Their Trading Costs"* (Review of
  Financial Studies 29(1)). The definitive cross-sectional result: after realistic trading costs,
  the surviving anomalies are overwhelmingly the *low-turnover* ones; many celebrated
  high-turnover signals have net returns indistinguishable from zero. Exactly the turnover curve
  this study draws on a controlled tape.
- **Korajczyk & Sadka (2004)** and **Lesmond, Schill & Zhou (2004)**, on the price-impact and
  round-trip costs that sink momentum and other high-turnover strategies once you size them —
  the empirical shape our synthetic turnover curve reproduces.

## Method lineage (the desk's shared engine)

- **Newey & West (1987)**, *"A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix"* (Econometrica) — the HAC *t* used on the daily
  gross and net spread series (`strategy.newey_west_t`).
- **Wilson (1927)** — the score interval for a binomial share (`strategy.wilson_interval`).
- **Sharpe (1994)**, *"The Sharpe Ratio"* (Journal of Portfolio Management) — the risk-adjusted
  return the whole ladder is measured in.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — a synthetic control is a
  machinery proof, never market evidence; `REAL` needs a robust *t* ≥ 2 on a real tape (which a
  synthetic-only demo can never provide); costs are one-way × NAV per rebalance leg; ≥ 20 seeds
  for any synthetic-dependent claim.

## Data sources used here

- **None.** This is a synthetic-only research-method demo. Every number is produced offline and
  deterministically by [`cost_gap/data.py`](../cost_gap/data.py) (seed 842); the entire test suite
  and both notebooks run without the network. All headline numbers are pinned with an as-of date
  (2026-06-30) and content fingerprints in [`docs/results.md`](results.md), reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[Study 30 — House-Edge (retail markup)](../../30-house-edge/)** — the fixed retail *markup /
  spread* a dealer charges a customer on a single instrument. Study 842 is about the *dynamic,
  turnover-scaled* cost of a **strategy** rotating a whole book — impact that grows with how much
  and how fast you trade, not a static per-ticket markup.
- **[Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/)** — a backtest looks
  great because the *search* manufactured it (selection over many rules on one tape), corrected by
  the Deflated Sharpe Ratio and PBO. Study 842's strategy is **not** overfit — its gross edge is
  genuine and holds out-of-seed; the edge dies from **costs**, not from selection. The two are the
  paired reasons a paper Sharpe misleads: 344 = the edge was never there; 842 = the edge was real
  on paper but you cannot afford to trade it.
- **[Study 619 — BITO Roll-Drag](../../619-bito-roll-drag/)** — a specific *structural* cost (the
  roll cost / contango drag of a futures-based ETF) eroding a single product's return. Study 842
  is the general *implementation* cost (spread + impact) of a **cross-sectional trading strategy**
  as a function of its turnover — a portfolio-level friction, not one instrument's carry.

None of the siblings model the **paper-vs-live gap of a strategy as an explicit function of
turnover with a super-linear market-impact term** — this study's own axis.
