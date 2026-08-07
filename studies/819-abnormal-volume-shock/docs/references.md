# References & literature map — Study 819 (Abnormal-Volume Shock)

## The claim under test

- **The source paper.** Jon A. **Garfinkel & Jonathan Sokobin**, *"Volume, Opinion
  Divergence, and Returns: A Study of Post-Earnings Announcement Drift"* (Journal of
  Accounting Research, 2006). Decomposing announcement-period volume into a liquidity part
  and an **unexplained** part, they argue the unexplained (abnormal) volume proxies
  **opinion divergence / disagreement**, and that higher unexplained volume is followed by
  a **larger positive drift** — attention and disagreement resolve into a subsequent
  return. The reading sits in the Karpoff / Harris-Raviv volume-information tradition:
  volume is a footprint of information arrival and belief dispersion.
- **The behavioural / microstructure reading.** A burst of volume that a name's own recent
  norm cannot explain marks a spike of attention and disagreement about fresh information.
  Under limited attention and slow diffusion (Hong-Stein), that information is impounded
  with a lag, producing a short forward drift in the direction the informed side takes.
- **The specific test here.** We take a self-contained daily cross-sectional version: for
  each name, **standardised abnormal volume** `(V − mean_60)/std_60` — how many sigmas
  today's volume is above its own trailing 60-day benchmark — averaged over a **5-day
  formation window**; then sort the cross-section and measure the forward return of the
  equal-weight long-high-abnormal-volume / short-low-abnormal-volume book, with a
  Newey-West *t*, a permutation placebo, a two-era robustness cut, a costed timer, and a
  seeded synthetic positive control. (The original paper conditions on **earnings
  announcements**; an all-days daily proxy is a deliberately weaker, more conservative test
  of the same disagreement mechanism.)

## What we measure, and the honesty rails

- **Abnormal volume, no free model.** For each name, `(Volume − trailing-60d mean) /
  trailing-60d std`, averaged over the last 5 days. Volume is the one column added to the
  shared 50-name panel loader versus the sibling price studies.
- **Point-in-time sort, one documented lag.** The ranking signal is the abnormal-volume
  average **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero
  look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (high book vs low book)
  cross-check. A **1,000-permutation placebo** breaks the signal → forward-return link to
  confirm the (tiny) spread isn't a lucky alignment of the sort.
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
- **Karpoff, J. (1987)** — the price/volume relation survey; **Harris, M. & Raviv, A.
  (1993)** — differences of opinion and trading volume, the theoretical backbone of the
  volume-as-disagreement reading.
- **Hong, H. & Stein, J. (1999)** — gradual information diffusion, the mechanism that would
  turn an attention shock into a forward drift.

## Data sources

- **yfinance daily OHLC + Volume** (`auto_adjust=True`, total-return), 50 liquid US
  large-caps, 2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [512-high-volume-return-premium](../../512-high-volume-return-premium/) — the **level**
  of trading / dollar volume (a liquidity-premium sort). A perennially heavy name is
  high-*level* every day; this study sorts on volume **relative to a name's own trailing
  norm**, so it fires only on the days a name is abnormal *for itself*.
- [141-turnover-anomaly](../../141-turnover-anomaly/) — **share turnover** (volume ÷ shares
  outstanding), a slow liquidity/attention state variable. This study is a short-window
  standardised **shock** off the recent mean, not a level of turnover.
- [254-wsb-mentions](../../254-wsb-mentions/) — an **exogenous** social-media attention
  proxy (Reddit mention counts). Here the attention proxy is **endogenous** to the tape —
  abnormal volume versus a name's own 60-day benchmark — with no alt-data feed.

None of the siblings sort on **standardised abnormal volume versus a name's own trailing
benchmark** — the Garfinkel-Sokobin disagreement shock — which is this study's own axis.
