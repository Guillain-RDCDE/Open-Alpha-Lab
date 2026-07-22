# References & literature map — Study 788 (Overnight / Intraday Tug of War)

## The claim under test

- **The source paper.** Dong **Lou, Christopher Polk & Spyros Skouras**, *"A Tug of
  War: Overnight Versus Intraday Expected Returns"* (Journal of Financial Economics,
  2019). Their central finding: the cross-section of expected stock returns is pulled
  in **opposite directions overnight versus intraday**. Sorting stocks on a firm
  characteristic (or on the past overnight/intraday component itself), the overnight
  component of returns **persists** while the intraday component **reverses** — so a
  strategy that looks strong on close-to-close returns is often a near-cancellation of
  a persistent overnight leg and a reversing intraday leg. Momentum, in particular,
  they show, is an **overnight** phenomenon; the intraday leg fights it.
- **The specific test here.** We take the sharpest, self-contained version: sort a
  liquid US cross-section on its **trailing overnight return**, then measure the
  **forward** overnight and intraday legs of the high-minus-low portfolio. LPS predict
  a **positive overnight leg** (the past-overnight winners keep winning overnight) and
  a **negative intraday leg** (they reverse intraday). We report the Newey-West *t* of
  each leg on the real tape, plus whether the two roughly cancel close-to-close.
- **Mechanism (why it might be real).** LPS attribute the split to a persistent
  clientele: a class of investors (retail / news-driven) concentrates demand at the
  open and overnight, pushing prices, while a different clientele (institutions,
  liquidity providers) leans against it during the day — a tug of war between the two
  return components that recurs day after day.

## What we measure, and the honesty rails

- **Exact decomposition, no free parameters.** `r_overnight = Open/prevClose − 1`,
  `r_intraday = Close/Open − 1`, via `quantlab.decompose` — the same identity study 01
  is built on. `auto_adjust=True` scales Open and Close by one daily factor, so the
  night/day split survives total-return adjustment.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing-21-day
  mean overnight return **known at the close of `t−1`** (`.shift(1)`); the portfolio is
  held on day `t`. Zero look-ahead: the overnight leg from `Close[t−1]` to `Open[t]` is
  capturable because you enter at the close of `t−1`, the instant the signal is fully
  known.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on each daily leg-spread
  series — the daily long-short spread of an overlapping-formation signal is serially
  correlated, so a plain *t* would overstate significance. A one-sample *t* and a pooled
  Welch *t* (top book vs bottom book, per leg) cross-check. A **1,000-permutation
  placebo** breaks the signal → forward-outcome link (column-permute the forward legs)
  to confirm the persistence isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**. The caveat travels
  with every published number; the qualitative tug is the robust claim, not the exact bps.
- **The timer is graded separately.** Costs are one-way × NAV, charged on every leg the
  overnight-capture book turns over (enter close, exit open = 2 sides × 2 legs/day), and
  the short book pays borrow — the honest reason the tradability stamp is `MIRAGE`.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on every leg-spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share (available in the
  strategy primitives for any conditional hit rate).
- **Knuteson, B. / study 01** — the exact overnight/intraday decomposition this desk
  reuses (`quantlab.decompose`).

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [01-overnight-anomaly](../../01-overnight-anomaly/) — the **aggregate / index-level**
  overnight-vs-intraday split (does the market *as a whole* earn its return at night?).
  That is a single time series' night/day means; this study is the **cross-sectional**
  tug — a *sort* on individual names' past overnight returns and the forward legs of the
  resulting long-short. LPS's contribution is precisely the cross-section, which #01 does
  not touch.
- [640-gold-overnight](../../640-gold-overnight/) — the same night/day decomposition on a
  **single asset** (gold / GLD). No cross-section, no sort, no tug-of-war between
  persistence and reversal across names.
- [116-power-hour](../../116-power-hour/) — an **intraday clock** effect (does the last
  trading hour follow the morning?), on a different timescale. Same intraday *family*, but
  it never sorts the cross-section on overnight returns or contrasts the overnight vs
  intraday legs.

None of the siblings run the **cross-sectional overnight-sorted persistence-vs-reversal**
test — the Lou-Polk-Skouras tug of war — which is this study's own axis.
