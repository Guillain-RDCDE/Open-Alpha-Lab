# References & literature map — Study 953 (Replicating the Convert)

## The claim under test

- **The convertible-fund pitch.** A convertible bond is sold as a hybrid you cannot build
  from parts: a bond that converts into equity, so you ride the stock up while a bond floor
  catches you on the way down. The fund wrapper (CWB, ICVT) inherits the pitch — "equity-like
  returns with bond-like drawdowns", a *convex* payoff worth a 20-40 bps management fee.
- **The steelman, and the null it implies.** Standard convertible-arbitrage theory says a
  convertible *is* decomposable: straight debt plus a long equity call. If the decomposition
  is exact, then a fund holding hundreds of them is, in aggregate, a static equity + credit +
  cash mix — with one twist. The twist is **path dependence**: each embedded call's delta
  rises as its underlying rallies, so an aggregate convertible portfolio should behave like a
  mix whose *equity weight ratchets up in rallies and down in selloffs*. That path dependence
  is precisely what a **static** replication cannot copy, and it is what would show up as a
  positive residual and a tail smile. So the null ("a costume") and the alternative ("genuine
  convexity") make different, testable predictions — which is why this study fits the static
  mix out-of-sample and then interrogates the residual's *shape*, not just its mean.

## The valuation and decomposition literature

- **Ingersoll (1977)**, *A Contingent-Claims Valuation of Convertible Securities*, Journal of
  Financial Economics; **Brennan & Schwartz (1977, 1980)**, *Convertible Bonds: Valuation and
  Optimal Strategies for Call and Conversion*, Journal of Finance / JFQA. The founding
  decomposition: a convertible is straight debt plus a warrant, with the issuer's call
  feature capping the upside. Everything in this study's replication is that decomposition
  taken literally and priced with liquid ETFs.
- **Ammann, Kind & Wilde (2003)**, *Are Convertible Bonds Underpriced? An Analysis of the
  French Market*, Journal of Banking & Finance; **Chan & Chen (2007)**, *Convertible Bond
  Underpricing: Renegotiable Covenants, Seasoning and Convergence*, Management Science. The
  classic "new-issue underpricing" evidence, which is where any genuine convertible alpha is
  supposed to come from — a primary-market effect that a **secondary-market index fund** like
  CWB has little claim on. Our blank hold-out is consistent with that distinction.
- **Batta, Chacko & Dharan (2010)**, *A Liquidity-Based Explanation of Convertible Arbitrage
  Crashes*, Financial Analysts Journal. Why the "bond floor" is weakest exactly in a crisis:
  when arbitrageurs delever, convertibles trade below the value of their parts. Our
  March-2020 month (fund −13.3% vs its own replica −6.6%) is a textbook instance.

## Factor replication as a method

- **Hasanhodzic & Lo (2007)**, *Can Hedge-Fund Returns Be Replicated? The Linear Case*,
  Journal of Investment Management. The canonical exercise: fit a fund to a handful of liquid
  factors in-sample, freeze the loadings, and score the hold-out. We copy the discipline —
  including the part practitioners skip, which is that the weights must be **frozen** before
  the window they are judged on.
- **Sharpe (1992)**, *Asset Allocation: Management Style and Performance Measurement*,
  Journal of Portfolio Management. Returns-based style analysis with **non-negative weights
  summing to at most one** — the exact constrained least-squares problem solved in
  [`strategy.constrained_ls`](../convert_repl/strategy.py), which keeps the replica a
  portfolio a human could actually hold.
- **Fung & Hsieh (2001, 2004)**, *The Risk in Hedge Fund Strategies* / *Hedge Fund
  Benchmarks: A Risk-Based Approach*, Review of Financial Studies / FAJ. The warning that
  option-like strategies have non-linear exposures a linear replication misses — which is why
  this study does not stop at the residual's mean but tests its **curvature** as well.

## Testing the shape, not the level

- **Treynor & Mazuy (1966)**, *Can Mutual Funds Outguess the Market?*, Harvard Business
  Review. The quadratic-in-the-market regression whose curvature term γ is the standard
  convexity statistic; here it is run on the **fund-minus-replica** difference, so it asks
  whether the wrapper is convex *relative to its own parts* rather than relative to cash.
- **Henriksson & Merton (1981)**, *On Market Timing and Investment Performance II*, Journal
  of Business. The piecewise (up-market / down-market beta) alternative — the source of the
  up-capture / down-capture asymmetry reported alongside γ.
- **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../convert_repl/strategy.py) and
  [`strategy.hac_ols`](../convert_repl/strategy.py), plus
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Politis & Romano (1994)**, *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_mean_ci`](../convert_repl/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py). Blocks of 21 days
  preserve the autocorrelation of a hybrid instrument's tracking error.

## Related desk studies (dedup)

- **[Study 339 — Convertible-Bonds](../../339-convertible-bonds/)**: tests CWB's convexity
  against **SPY directly** (a Treynor-Mazuy quadratic on the market, plus a *beta-matched*
  SPY/AGG blend). Study 953 changes the counterfactual and the discipline in three ways that
  matter: (1) the benchmark is a **fitted, three-factor, long-only replication** with a
  QQQ growth leg and a cash weight the data chooses — not a two-asset blend matched on beta;
  (2) the weights are estimated **in-sample and frozen**, so the alpha is scored on a genuine
  hold-out rather than in-sample; (3) the convexity question is asked **against the replica**,
  in tail terciles, rather than against the market. 339 asked "is CWB convex?"; 953 asks
  "is CWB *anything* its own parts are not?"
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: linear stock/bond diversification —
  the mix itself, not a wrapper claiming to improve on it.
- **[Study 912 — Gold + Trend](../../912-gold-trend-managed/)**: the same house discipline
  (excess-of-cash race, HAC *t* on the difference, era cut, cost sweep, synthetic control)
  applied to a *timing rule* rather than a *replication*; the shared inference spine is
  deliberately identical.

## Data sources

- **CWB** (SPDR Bloomberg Convertible Securities, the $4bn liquid-convertible tracker) and
  **ICVT** (iShares Convertible Bond) — the funds under test. **SPY, QQQ, LQD** — the
  replication kit. **BIL** (1-3M T-bills) — the cash leg. All daily **total-return** closes
  via `yfinance` (`auto_adjust=True`), 2009-04-16 → 2026-06-30 for the CWB panel.
- **Total return, never price only.** Both convertible funds distribute coupons (CWB's yield
  has run 1-3%/yr) and LQD is mostly coupon, so a price-only comparison would understate the
  fund and the credit leg alike. Every number in this study is total return.
- **Expense ratios (0.40% CWB, 0.20% ICVT) are quoted from the issuers, not measured** —
  they are labelled a PROXY in the README and matter only as context; the fund's tape is
  already net of them. The replica's own 10 bps/yr fee drag is likewise an assumption, and
  is swept 0-20 bps in [`docs/results.md`](results.md).
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
