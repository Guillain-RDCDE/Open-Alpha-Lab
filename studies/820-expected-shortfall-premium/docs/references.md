# References & literature map — Study 820 (Expected-Shortfall Premium)

## The claim under test

- **The source idea.** Turan **Bali, Nusret Cakici & Robert Whitelaw** develop the modern
  cross-sectional test of **downside tail risk** as a priced characteristic. In *"Hybrid Tail
  Risk and Expected Stock Returns"* (Review of Asset Pricing Studies, 2014) and the related
  **Value-at-Risk** work (*"Maxing Out: Stocks as Lotteries...",* JFE 2011, and Bali-Cakici's
  VaR-and-returns studies), they sort stocks on left-tail severity measured directly from the
  return tape and ask whether the fat-left-tail names are compensated. **Expected Shortfall**
  (CVaR — the *mean* of the returns beyond the VaR quantile) is the coherent, sub-additive
  refinement of VaR (Artzner, Delbaon, Eber & Heath, 1999; Rockafellar & Uryasev, 2000): it
  reads the *depth* of the worst days, not just their threshold.
- **The two readings.** A **rational** reading says a fat left tail is genuine crash exposure and
  must earn a risk premium (high ES → high forward return). A **behavioural** reading (the MAX /
  lottery literature) warns that tail-heavy names are also lottery-like and can be *over*-priced,
  giving the opposite sign. Which one wins is an empirical, universe-and-regime question — the
  point of this teardown.
- **The specific test here.** We take the self-contained daily version: sort a liquid US
  cross-section on its **trailing-252-day historical Expected Shortfall at 5%** (the mean of the
  worst 5% of daily returns) and measure the forward return of the equal-weight long-high-ES /
  short-low-ES book, with a Newey-West *t*, a permutation placebo, a two-era robustness cut, a
  costed timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Expected Shortfall, no free model.** For each name, the rolling `window`-day historical ES:
  the negative mean of the worst `ceil(0.05·window)` daily simple returns (a positive
  magnitude). Non-parametric — no distributional assumption, no GARCH.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing ES **known at
  the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short spread —
  an overlapping-formation signal is serially correlated, so a plain *t* would overstate
  significance. A one-sample *t* and a pooled Welch *t* (high-ES book vs low-ES book) cross-check.
  A **1,000-permutation placebo** breaks the signal → forward-return link.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Because ES is ~collinear with volatility,
  a survivor panel of today's mega-caps *flatters* a high-ES-earns-more sort — the magnitudes are
  an upper bound.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and the
  short book pays borrow — the honest test of whether the spread survives friction.

## Shared method citations

- **Rockafellar, R. T. & Uryasev, S. (2000)** — optimization of Conditional Value-at-Risk; the
  formal definition of ES/CVaR used here.
- **Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999)** — coherent risk measures (why ES,
  unlike VaR, is sub-additive and tail-sensitive).
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [332-downside-beta](../../332-downside-beta/) — a stock's **systematic** beta *conditional on
  down markets* (co-movement with the market's bad days), a covariance with a common factor.
  Expected Shortfall is a name's **own** left-tail severity, read off its marginal return
  distribution — factor-free, not a beta.
- [505-left-tail-momentum](../../505-left-tail-momentum/) — the short-horizon **continuation** of
  extreme negative returns (dynamics *in* the tail). ES is a *level* statistic — how deep the
  average bad day is — not the persistence of tail events.
- [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — the **whole two-sided
  dispersion** of returns. ES is one-sided: it ignores the right tail and reads only the depth of
  the worst 5% of days. (Empirically the two are correlated, which is exactly the caveat this
  study flags on the mega-cap universe.)

None of the siblings sort on a name's **own one-sided left-tail Expected Shortfall (CVaR)** — the
priced-tail-risk signal — which is this study's own axis.
