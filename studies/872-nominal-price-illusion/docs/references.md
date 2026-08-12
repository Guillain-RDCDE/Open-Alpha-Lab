# References & literature map — Study 872 (Nominal-Price Illusion)

## The claim under test

- **The behavioural source.** Alok **Kumar**, *"Who Gambles in the Stock Market?"*
  (Journal of Finance, 2009). Retail investors with a gambling propensity systematically
  over-weight **lottery-type stocks** — low-priced, high-idiosyncratic-volatility,
  high-idiosyncratic-skewness names. Low **nominal price** is one of the three defining
  lottery features: a cheap-looking share is the cheapest lottery ticket to buy in round lots.
- **The nominal-price mechanism.** Justin **Birru & Baolian Wang**, *"Nominal Price Illusion"*
  (Journal of Financial Economics, 2016). Investors mistakenly believe **low-priced** stocks
  have more room to grow — they over-estimate the *skewness* of low-priced names — a pure
  money illusion, since the nominal price (value ÷ shares outstanding) carries **no**
  information about a firm's value or prospects. The illusion is strongest exactly where the
  price level is arbitrary and where retail participation is high.
- **The prediction here.** If lottery / illusion demand over-prices cheap-looking stocks, then
  a cross-section sorted on **nominal price level** should show low-priced names with (a) the
  lottery look — higher volatility and more right-skew — and (b) **lower risk-adjusted forward
  returns**. A book long the cheap names / short the dear names should earn a **negative**
  `lo − hi` return spread.
- **The specific test here.** We sort a liquid US cross-section on its nominal price level
  (the adjusted Close, a proxy — see the honesty rails), long the cheapest 30% / short the
  priciest 30%, and measure the forward `lo − hi` spread with a Newey-West *t*, per-book
  vol / skew / Sharpe, a two-sided permutation placebo, a two-era robustness cut, a costed
  timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Price level, no free model.** The signal is simply each name's price. No factor model, no
  estimation — the money-illusion characteristic is the raw dollar number.
- **Point-in-time sort, one documented lag.** The ranking price is **known at the close of
  `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short spread;
  a one-sample *t* and a pooled Welch *t* (cheap book vs dear book) cross-check; a
  **1,000-permutation two-sided placebo** breaks the signal → forward-return link to confirm
  the spread isn't a lucky alignment of the sort.
- **The risk-adjusted read is reported explicitly.** The claim is about *risk-adjusted*
  underperformance, so we report each book's annualised vol, return skew, and Sharpe — a raw
  return spread alone would hide whether the cheap book merely took more risk.
- **Survivorship + adjusted-price proxy, named on the Signal axis.** The universe is a
  **current-membership** set of ~50 liquid mega-caps (`quantlab.universe` guard,
  `allow_survivorship_bias=True`), so delisted names are absent *and*, critically for this
  signal, **mega-caps are rarely cheap** — no single-digit names exist in the panel, so the
  retail-lottery segment the theory targets is missing (honest low power). The cache is
  `auto_adjust=True`, so the price level is **split/dividend-back-adjusted** — an honest proxy
  for the true nominal trading price that is exact only near the as-of date. A pure-nominal
  replication needs raw unadjusted closes, which the total-return cache does not preserve.
- **The timer is graded separately.** Costs are 2 sides × one-way × NAV on the long-short book,
  and the short (dear) book pays borrow — the honest test of whether a small daily spread
  survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Barberis, N. & Huang, M. (2008)** — cumulative-prospect-theory over-weighting of the tails,
  the utility-side foundation for why a lottery-like (right-skewed, cheap) name is over-priced.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [11-vanishing-penny](../../11-vanishing-penny/) — literal **penny stocks** and the fate of
  the sub-dollar segment. This study is a *continuous cross-sectional* sort on the **price
  level** of *liquid* names, not an examination of the penny-stock tail itself.
- [365-lottery-max-effect](../../365-lottery-max-effect/) — the single **MAX** daily return
  (Bali-Cakici-Whitelaw), a realized-tail order statistic. This study sorts on the **price
  level** itself, a static balance-sheet-arbitrary number, not a realized return feature.
- [250-reverse-split](../../250-reverse-split/) — the **event** of a reverse split, a discrete
  engineered *reset* of the price level. This study never conditions on an event; it ranks the
  standing cross-section by price every day.
- [93-round-numbers](../../93-round-numbers/) — price **round-number** magnetism ($100/$50
  barriers), a *within-name* level-anchoring effect. This study is a *cross-name* cheap-vs-dear
  sort, not a proximity-to-a-round-number signal.

None of the siblings sort the standing cross-section on a name's **nominal price level** — the
Kumar / Birru-Wang money-illusion axis — which is this study's own signal.
