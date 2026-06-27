# References & literature map — Study 502 (Betting-Against-Correlation)

## The claim, at full strength

- **Asness, Frazzini, Gormsen & Pedersen (2020)**, *"Betting Against Correlation: Testing
  Theories of the Low-Risk Effect."* *Journal of Financial Economics* 135(3), 629–652. The
  paper this study replicates. Since `beta = correlation × (vol_stock / vol_market)`, they split
  Betting-Against-Beta into a **Betting-Against-Correlation (BAC)** leg and a
  **Betting-Against-Volatility (BAV)** leg. They find the **correlation** component carries the
  low-risk premium — consistent with leverage-constraint theories (Frazzini–Pedersen) and
  *inconsistent* with pure lottery-demand/volatility stories. Our decomposition (sort on corr
  vs beta vs vol on the same panel) is the direct test.
- **Frazzini & Pedersen (2014)**, *"Betting Against Beta."* *Journal of Financial Economics*
  111(1), 1–25. The parent result: leverage-constrained investors bid up high-beta assets,
  flattening the security-market line; a beta-neutral long-low/short-high book earns a premium.
  The construction behind both BAB and our beta-neutralised BAC book. This desk's
  [Study 238 — Betting-Against-Beta](../../238-betting-against-beta/) replicates BAB directly;
  Study 502 isolates the *correlation* slice AFGP say is the real driver.
- **Baker, Bradley & Wurgler (2011)**, *"Benchmarks as Limits to Arbitrage: Understanding the
  Low-Volatility Anomaly."* *Financial Analysts Journal* 67(1). The low-risk anomaly's canonical
  statement and the benchmark-relative-mandate mechanism; the backdrop AFGP refine.
- **Bali, Brown, Murray & Tang (2017)**, *"A Lottery-Demand-Based Explanation of the Beta
  Anomaly."* *Journal of Financial and Quantitative Analysis* 52(6). The competing demand-side
  story (investors overpay for lottery-like high-vol stocks) that AFGP's BAC/BAV split is
  designed to discriminate against — they argue the *correlation* result favours the
  leverage-constraint channel over the lottery channel.

## Shared method

- **Newey & West (1987)** — heteroskedasticity- and autocorrelation-consistent (HAC) standard
  errors; the *t*-stat the inference bar requires on the monthly book.
- **Politis & Romano (1994)** — circular block bootstrap; preserves the short-run
  autocorrelation that i.i.d. resampling destroys (this study's CI on the BAC spread).
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005, *Permutation, Parametric, and
  Bootstrap Tests of Hypotheses*) — permute the correlation ranks across names to build the null
  the real spread is scored against (this study's placebo p-value).

## Neighbours on this bench (the dedup map)

- **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta/)** — the parent BAB
  factor: sort on **beta**, leverage the low-beta leg to unit beta, short the high-beta leg.
  Study 502 sorts on **correlation** (holding the vol slice out), and directly contrasts the
  correlation sort against the beta and volatility sorts on the same panel — the AFGP
  decomposition 238 does not do.
- **[Study 330 — Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)** — the retail ETF
  embodiment (SPLV vs SPHB) of the low-*vol* effect. Study 502 is the cross-sectional
  *correlation*-vs-vol-vs-beta decomposition on single names, not the ETF race.

## House methodology

- [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (HAC *t* ≥ 2 + placebo +
  seed-robustness), the excess-of-cash Sharpe rule, one execution lag documented exactly, costs
  one-way × NAV × turnover with shorts paying borrow, and the survivorship-bias naming rule.
