# References & literature map — Study 817 (Realized-Volatility Trend)

## The claim under test

- **The idea.** Two names can share the same *level* of volatility yet be moving in
  opposite directions: one's vol is **rising** (its recent short-window realized vol has
  climbed above its longer-window average), the other's is **falling**. The claim is that
  this *change* in vol — vol **momentum**, distinct from the low-vol **level** effect — is
  itself priced: rising-vol names keep **de-rating** (the market re-prices their risk
  upward, and the equity with it) while falling-vol names **re-rate**. A long
  falling-vol / short rising-vol book should therefore earn a positive spread.
- **The measure.** For each name, `vol trend = (trailing 21d realized vol) / (trailing
  63d realized vol) - 1`. Positive = short-window vol above its longer average (rising);
  negative = falling. It is deliberately a *ratio of two vols on the same name*, so it is
  near-orthogonal to the vol **level** (a name can be low-vol-and-rising or
  high-vol-and-falling).
- **Where the idea comes from.** The construct sits at the intersection of two literatures.
  (i) **Volatility-of-volatility / vol trend as a risk signal** — the finance-practitioner
  observation that *accelerating* realized volatility precedes negative drift, a cousin of
  the "vol-of-vol" premium (**Baltussen, Van Bekkum & Van der Grient, 2018**, *Unknown
  Unknowns: Uncertainty About Risk and Stock Returns*, Journal of Financial and Quantitative
  Analysis). (ii) The **low-volatility anomaly** it must be separated from
  (**Ang, Hodrick, Xing & Zhang, 2006**, *The Cross-Section of Volatility and Expected
  Returns*, Journal of Finance; **Blitz & van Vliet, 2007**, *The Volatility Effect*,
  Journal of Portfolio Management), which is a **level** effect — the whole point here is to
  test whether the *trend* adds anything the *level* does not.
- **Volatility clustering — why a trend even exists.** Realized vol is strongly
  autocorrelated (**Engle, 1982**, ARCH; **Bollerslev, 1986**, GARCH), so a 21d/63d vol
  ratio is a meaningful, persistent state variable rather than noise — the pre-condition
  for a tradable vol trend.

## The specific test here

- We take the self-contained daily version on a liquid US cross-section: sort on the
  **trailing vol trend** and measure the forward return of the equal-weight
  long-falling-vol / short-rising-vol book, with a Newey-West *t*, a permutation placebo,
  an explicit **additivity regression against the low-vol level sort** (study 330's signal),
  a two-era robustness cut, a costed timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Realized vol, no free model.** For each name, the rolling 21-day and 63-day sample
  standard deviation of daily simple returns; the signal is their ratio minus one.
- **Point-in-time sort, one documented lag.** The ranking signal is the vol trend **known
  at the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Additivity, not just significance.** Because the desk already has the low-vol *level*
  anomaly (330), a bare *t* on the trend spread is not enough — we regress the trend spread
  on the level spread and read a Newey-West *t* on the residual, so a "signal" that is just
  the level effect in disguise cannot pass.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (falling book vs rising
  book) cross-check. A **1,000-permutation placebo** breaks the signal → forward-return link.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and
  the short book pays borrow — the honest test of whether a small daily spread survives
  friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Bollerslev, T. (1986)** — GARCH, the canonical model of volatility persistence that
  underwrites a meaningful short-vs-long realized-vol ratio.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [330-low-volatility](../../330-low-volatility/) — the low-vol **level** anomaly (low-vol
  names out-earn high-vol names). This study sorts on the **trend** (change) in vol, and
  explicitly regresses its spread against the level spread to test additivity; the two are
  near-orthogonal (corr +0.065 on the real tape).
- [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — the **level** of
  *residual* (market-model) volatility (Ang-Hodrick-Xing-Zhang idiosyncratic-vol puzzle),
  still a level, and residual rather than total vol. This study uses total realized vol and
  its **trend**.
- [6-clockwork-volatility](../../6-clockwork-volatility/) — the **calendar seasonality** of
  aggregate volatility (a time-series clock on the market's own vol), not a
  **cross-sectional** name-by-name trend sort.

None of the siblings sort the cross-section on the **change in a name's own realized vol** —
the vol-momentum axis this study tests.
