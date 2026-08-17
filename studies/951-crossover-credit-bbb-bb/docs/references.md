# References & literature map — Study 951 (The Crossover Rung)

## The claim under test

- **The crossover-rung thesis.** Credit is sold to investors as a ladder — aggregate,
  investment grade, crossover (BBB−/BB+), broad high yield, CCC — and the folklore holds
  that the *boundary* rung is the best-paid step on it. The argument is institutional, not
  statistical: investment-grade mandates, insurance capital charges (NAIC designations) and
  index-tracking rules force holders to sell when an issuer is downgraded out of IG, while
  the natural buyers on the other side (high-yield funds) are constrained in size and often
  cannot buy until the bond enters *their* index. The result is supposed to be a
  price-pressure discount at the boundary that no one on either side is free to arbitrage —
  yield without the default experience of deep high yield.
- **The steelman.** If the story is true, the crossover rung should out-earn *both* of its
  neighbours after you strip out the two risks it obviously carries more of: interest-rate
  duration and equity beta. That is a testable, mechanical claim, and this study tests it on
  the tradable tape — the fallen-angel ETFs against investment-grade and broad high-yield
  ETFs — rather than on an index back-test that no one could have held.

## Why the premium *should* exist — the mechanism

- **Ben Dor & Xu (2011, updated 2015),** *Fallen Angels: Characteristics, Performance and
  Implications for Investors*, Journal of Fixed Income. The canonical index study: bonds
  downgraded out of investment grade underperform sharply *into* the downgrade and then
  recover strongly afterwards, with the abnormal return concentrated in the months following
  index exclusion. The forced-seller reading of the credit ladder starts here.
- **Ambastha, Ben Dor, Dynkin, Hyman & Konstantinovsky (2010),** *Empirical Duration of
  Corporate Bonds and Credit Market Segmentation*, Journal of Fixed Income. Documents that
  effective duration and equity sensitivity vary systematically along the quality ladder —
  which is precisely why any rung race must be run against a duration factor *and* an equity
  factor, as this study does, rather than on raw yield or raw excess return.
- **Chen, Lookman, Schürhoff & Seppi (2014),** *Rating-Based Investment Practices and Bond
  Market Segmentation*, Review of Asset Pricing Studies. Shows that rating-contingent
  mandates segment the market and generate price pressure around the IG/HY boundary — the
  cleanest academic statement of the mechanism the crossover thesis relies on.
- **Ellul, Jotikasthira & Lundblad (2011),** *Regulatory Pressure and Fire Sales in the
  Corporate Bond Market*, Journal of Financial Economics. Insurance-company capital rules
  produce measurable fire sales on downgrade, with prices reverting over the following
  quarters. This is the causal engine behind the fallen-angel effect and the reason the
  premium should be *episodic* — it fires in downgrade waves, not evenly through time.

## Why it can be smaller than advertised

- **Ng & Phelps (2011),** *Capturing Credit Spread Premium*, Financial Analysts Journal.
  Realised credit excess returns are far smaller than headline spreads once downgrades,
  defaults and the transaction costs of index turnover are charged — the general warning
  that spread ≠ premium applies to the crossover rung too.
- **Asvanunt & Richardson (2017),** *The Credit Risk Premium*, Journal of Fixed Income. Once
  credit returns are hedged of duration and equity exposure the residual premium is real but
  modest and concentrated in crises — consistent with this study's finding that two
  downgrade-wave years (2016, 2020) carry the entire crossover premium.
- **Post-publication decay.** McLean & Pontiff (2016), *Does Academic Research Destroy Stock
  Return Predictability?*, Journal of Finance. Fallen-angel index research was published in
  2011 and productised as an ETF in 2012; this study's era cut (alpha +4.47%/yr before 2019
  against +1.50%/yr after, on the same fund) is exactly the shape that literature predicts,
  and is the reason the Signal stamp is Weak rather than Real.
- **Fund-proxy risk.** A fallen-angel ETF is not the crossover rung: it is an event-tilted,
  longer-duration slice of it, and two such funds (ANGL, FALN) hold different vintages of
  the same wave. The FALN − USHY cross-check (+0.63%/yr, *t* = +0.48) in
  [`results.md`](results.md) is the honest measure of how much of the headline is the rung
  and how much is one fund's particular composition.

## Related desk studies (dedup)

- **[Study 610 — Fallen-Angels-Premium](../../610-fallen-angels-premium/)**: the closest
  neighbour and the direct ancestor. It asks a *within-high-yield selection* question — do
  bonds ejected from investment grade beat broad high yield? — on **monthly** ANGL vs HYG
  with duration and quality *controls added one at a time*, and stamps it Real/Investable.
  Study 951 asks a different question: not "is the fallen-angel slice better than HY?" but
  "**is the boundary the peak of the whole ladder?**" — a four-rung race (AGG → LQD → ANGL →
  HYG) on **daily** data with duration *and* equity beta adjusted **jointly and always**, and
  it adds the leg 610 never ran: **ANGL − LQD**, the investment-grade side of the boundary
  (+1.51%/yr, *t* = +1.15 — absent). It also puts the shared ANGL − HYG leg through a
  robustness battery 610 did not run: a one-year-out jackknife (survives 5/15), a
  crisis-year deletion (2016 + 2020 → *t* = +1.61), an HAC-bandwidth sweep, and the
  both-legs-swapped FALN − USHY pair (*t* = +0.48). Those tests are why this study stamps
  the ladder claim **Weak** where 610 stamps its narrower claim Real — the two verdicts are
  about different propositions, and the disagreement on the overlapping leg is a robustness
  disagreement, stated openly here.
- **[Study 115 — Credit-Spreads](../../115-credit-spreads/)**: whether high-yield spreads
  *predict equities* — cross-asset timing, not a within-credit rung comparison.
- **[Study 832 — High-Yield Credit Momentum](../../832-high-yield-credit-momentum/)**:
  timing the SPY↔IEF switch on the credit *trend* — a timing rule, whereas 951 races static
  holdings.
- **[Study 885 — Ultra-Short Credit Pickup](../../885-ultra-short-credit-pickup/)**: the
  bottom rung of the same ladder (JPST/ICSH/MINT versus bills) — the IG-to-cash step, not the
  IG-to-HY boundary.
- **[Study 892 — Corporate Bond Ladder](../../892-corporate-bond-ladder/)**: a *maturity*
  ladder inside one credit quality; 951's ladder runs along the *quality* axis.
- **[Study 795 — Corporate Bond Momentum](../../795-corporate-bond-momentum/)** and
  **[Study 796 — Corporate Bond Low-Risk](../../796-corporate-bond-low-risk/)**: factor
  sorts *within* the corporate universe, not a rung-versus-rung race.

## Method lineage

- **HAC / Newey-West standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica — [`strategy.ols_hac`](../crossover_credit/strategy.py) and
  [`strategy.newey_west_t`](../crossover_credit/strategy.py). Bandwidth from the standard
  `4 (n/100)^(2/9)` rule, and swept in [`results.md`](results.md).
- **Alpha on a return difference.** Jobson & Korkie (1981), *Performance Hypothesis Testing
  with the Sharpe and Treynor Measures*, Journal of Finance — the head-to-head form used in
  [`strategy.pair_alpha`](../crossover_credit/strategy.py).
- **Factor-adjusted performance evaluation.** Jensen (1968), *The Performance of Mutual Funds
  in the Period 1945-1964*, Journal of Finance — the intercept-as-alpha convention this study
  applies with a duration and an equity factor.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_alpha_ci`](../crossover_credit/strategy.py), resampling the dependent
  series and the factor tape jointly.
- **Jackknife.** Quenouille (1949) / Tukey (1958) — the one-year-out deletion in
  [`strategy.year_jackknife`](../crossover_credit/strategy.py), used here as an
  influential-period diagnostic rather than a variance estimator.

## Data sources

- **AGG** (broad aggregate), **LQD** (investment grade), **ANGL** and **FALN**
  (fallen-angel / crossover proxies), **HYG** and **USHY** (broad high yield), **IEF**
  (7-10y Treasuries, the duration factor), **SPY** (the equity factor) and **BIL** (1-3M
  T-bills, the cash leg) — daily **total-return** closes via `yfinance`
  (`auto_adjust=True`), 2002 → 2026-06-30, cached in the shared `studies/_cache`.
  Total return is non-negotiable here: a bond fund's entire return is coupon, and a
  price-only tape would rank the ladder upside-down.
- The cash leg is BIL's *actual* total return, so every Sharpe is excess of the real path of
  short rates (~0% in 2012-2015, ~5% in 2023-2026) rather than a flat proxy.
- **As-of 2026-06-30**; the partial current month is dropped so the sample never creeps.
  ANGL's 2012-04 inception gates the headline window — the 1997-2011 index evidence for the
  fallen-angel effect pre-dates any tradable crossover fund and is *context only*, never
  evidence for the stamp.
