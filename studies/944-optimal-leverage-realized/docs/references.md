# References & literature map — Study 944 (How Much Leverage)

## The claim under test

- **The growth-optimal-leverage thesis.** For a log-utility investor holding a single
  risky asset financed at the risk-free rate, the growth-optimal exposure is
  `L* = mu_excess / sigma^2` — the Kelly fraction in continuous time, and Merton's
  solution for `gamma = 1`. On US equities the textbook plug-in (excess return ~6%/yr,
  vol ~16%/yr) lands near **2.3×**, which is why "2× the index, rebalanced daily" is the
  most repeated leverage prescription on the retail internet. The claim this study tests
  is not that the formula is wrong — it is exactly right, given the parameters — but the
  *practical* corollary everyone draws from it: that the optimal multiple is a number you
  can estimate and then use.
- **The steelman.** The growth curve `g(L) = L*mu − L^2*sigma^2/2 − (L−1)*spread` is
  smooth and concave, so being a little wrong about `L*` costs only second-order growth.
  If the peak is flat, a rough estimate should be good enough — which would make
  "somewhere around 2×" a perfectly serviceable answer. This study measures how flat the
  peak actually is, and how far it moves.

## The theory

- **Kelly (1956).** J. L. Kelly Jr., *A New Interpretation of Information Rate*, Bell
  System Technical Journal 35(4) — the growth-optimal betting fraction, from which
  `mu/sigma^2` follows in the continuous limit.
- **Merton (1969).** Robert C. Merton, *Lifetime Portfolio Selection under Uncertainty:
  The Continuous-Time Case*, Review of Economics and Statistics 51(3) — the same multiple
  as the risky-asset weight `mu/(gamma*sigma^2)`, with log utility at `gamma = 1`.
- **MacLean, Thorp & Ziemba (2011), eds.** *The Kelly Capital Growth Investment Criterion*,
  World Scientific — the standard collection, including the practitioner correctives
  (fractional Kelly, drawdown constraints) that exist precisely because full Kelly is
  brutal to live through. Our −94.5% drawdown at the realised optimum is the empirical
  statement of that folklore.
- **Variance drag / the daily-reset identity.** The `0.5*L*(L−1)*sigma^2` gap between
  `L × CAGR` and the CAGR of a daily-reset `L×` fund is the mechanical reason the growth
  curve turns over; the desk's [Study 61](../../61-slow-burn/) and
  [Study 100](../../100-melting-ice/) verify that identity against real 3× ETF tapes.

## Why the practical corollary can fail

- **Merton (1980).** *On Estimating the Expected Return on the Market*, Journal of
  Financial Economics 8(4) — the load-bearing citation for this study. Variance is
  estimable from high-frequency data at almost any horizon; the **mean is not**, and its
  standard error shrinks only with the calendar *span* of the sample. Since
  `L* = mu/sigma^2` is linear in the un-estimable quantity and inverse in the estimable
  one, the optimal multiple inherits all of the mean's imprecision. With `sigma = 18%` and
  23 years, `se(mu) ~ 3.8%/yr`, so `se(L*) ~ 1.2` — which is what our bootstrap CI of
  [1.00, 3.00] says in a different language.
- **Michaud (1989).** *The Markowitz Optimization Enigma: Is 'Optimized' Optimal?*,
  Financial Analysts Journal 45(1) — optimisers are error maximisers; the argmax of a
  noisily-estimated objective is far less stable than the objective itself. Our rolling
  five-year hindsight optimum (24% of windows at the floor, 54% at the cap) is that
  phenomenon in one dimension.
- **Estrada (2010), *Geometric Mean Maximization: An Overlooked Portfolio Approach?*,
  Journal of Investing** — geometric-return maximisation is theoretically clean and
  empirically unstable out of sample; the recommended allocations swing violently with the
  estimation window. Our cap and window sweeps reproduce this.
- **Bansal & Yaron (2004)**, *Risks for the Long Run*, Journal of Finance, and the
  broader time-varying-expected-return literature — if `mu` genuinely moves, the target is
  not merely noisy but non-stationary, and the era hand-off (the late decade's optimum
  under-performing *no leverage* in the early decade) is the direct consequence.

## Related desk studies (dedup)

- **[Study 157 — Kelly-Sizing](../../157-kelly-sizing/)** is the nearest neighbour and
  must be read alongside this one. It tests the *rule*: walk-forward full- and half-Kelly
  sizing on SPY 1993–2026, monthly rebalance, and grades it Weak/Fragile. Study 944 does
  not re-litigate that rule — it maps the **object** the rule is chasing: the realised
  terminal-wealth / Sharpe / drawdown surface across the whole 1.0–3.0 grid, a block
  bootstrap of the **argmax itself**, a rolling five-year hindsight optimum, and an
  era hand-off test. The ex-ante Kelly arm here is deliberately an *independent
  replication* of 157's headline on a different engine (daily reset, ^IRX financing plus a
  swept spread, turnover costed), and it agrees: positive sign, sub-2 *t*.
- **[Study 61 — Slow-Burn](../../61-slow-burn/)** and
  **[Study 100 — Melting-Ice](../../100-melting-ice/)**: the *instrument* studies — does a
  3× daily-reset ETF decay, and does the synthetic identity match the real fund tape?
  944 assumes that mechanic is correct (61 and 100 verified it) and asks the sizing
  question above it.
- **[Study 593 — HFEA](../../593-hfea-leveraged-6040/)** and
  **[Study 594 — Leverage-Rotation-200SMA](../../594-leverage-rotation-200sma/)**: levered
  *portfolios* and levered *timing rules*. 944 holds one asset at a constant multiple with
  no timing at all — deliberately the simplest possible leverage decision, so that the
  instability that shows up cannot be blamed on a signal.
- **[Study 590 — Sharpe-Hacking](../../590-sharpe-hacking/)**: shows that leverage does
  nothing to Sharpe. 944 measures that invariance exactly on the real tape (0.5752 at 1×
  and at 3×, gross of the spread) and uses it to rule the Sharpe axis out of the question
  before the growth analysis starts.
- **[Study 633 — BTC-Vol-Targeting](../../633-btc-vol-targeting/)**: a *time-varying*
  exposure rule (constant vol target). 944's exposure is constant by construction; the
  ex-ante Kelly arm is the only time-varying leg and is presented as a cross-check, not
  the headline.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../optimal_leverage/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1992, 1994) — the moving-block and
  stationary bootstraps for dependent data;
  [`strategy.block_bootstrap_ci`](../optimal_leverage/strategy.py) and
  [`strategy.bootstrap_optimum`](../optimal_leverage/strategy.py), which bootstraps the
  *argmax* rather than a mean.
- **Testing a growth (not mean) difference.** Because terminal wealth is a product,
  growth comparisons are run on daily **log**-return differences, not arithmetic ones;
  [`strategy.growth_diff_test`](../optimal_leverage/strategy.py) reports both so the
  divergence between them (arithmetic *t* = +2.54, log *t* = +1.20) is visible rather
  than exploited.

## Data sources & assumptions

- **SPY** — daily **total-return** closes via `yfinance` (`auto_adjust=True`), from the
  shared desk cache. Price-only closes are never used anywhere in this study; a
  price-only SPY would understate every point on the growth curve by the dividend yield.
- **^IRX** — the 13-week Treasury-bill **discount rate**, in percent. Converted to a daily
  accrual on an act/360 basis using the previous close's level over the calendar days the
  bar spans. It is the financing base for the borrowed sleeve *and* the cash leg that
  every excess return subtracts.
- **BIL** — the tradable 1–3 month T-bill ETF, used only as a **cross-check** on the ^IRX
  construction (^IRX-implied 1.47%/yr vs BIL 1.36%/yr over 2007–2026, a +11 bps/yr gap).
- **PROXY — financing spread, 50 bps/yr.** Not a measured quantity. It stands for the
  all-in cost of the borrowed sleeve above bills (E-mini roll cheapness, box-spread
  financing, or a levered-ETF fee load). Swept 0 / 25 / 50 / 100 / 200 bps; the realised
  optimum moves 3.00 → 2.45 across that range. Retail margin lending runs far wider than
  200 bps, at which point the optimum falls further still.
- **PROXY — reset cost, 1 bp one-way × notional traded.** Charged on the turnover the
  daily reset generates (10.1× NAV/yr at the optimum). Swept 0 → 5 bps; it barely moves the
  answer.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  The window opens 2003-06-04 because that is where the cached ^IRX history begins — it
  therefore contains the GFC and 2022 but **not** the 2000–2002 bust, and the realised
  optimum should be read as conditional on that. That conditionality is **measured, not
  merely confessed**: [`strategy.start_sensitivity`](../optimal_leverage/strategy.py)
  re-runs the headline from five different start dates and reports how far the answer
  moves (optimum 2.60 → 3.00; the era hand-off swinging from +1.73%/yr to −4.32%/yr
  against not levering at all). An earlier draft of this study ran on a cache whose ^IRX
  began 2004-01-06 and reported the hand-off with the opposite sign; that near-miss is
  what the section exists to make impossible to repeat quietly.
- **Survivorship.** No cross-section is screened here, so there is no name-level
  survivorship. The macro form remains and is load-bearing: SPY is one realisation of one
  index, in one country, that did not suffer a terminal decade inside the sample. A
  growth-optimal multiple estimated on a tape that happened to compound is the definition
  of a quantity conditioned on survival.
- No shorting appears anywhere in this study, so no borrow fee on a short leg arises; the
  financing spread on the borrowed sleeve is the leverage analogue and is swept in its
  place (0 / 25 / 50 / 100 / 200 bps/yr).
