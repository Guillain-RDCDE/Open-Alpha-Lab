# References & literature map — Study 870 (Industry-Leader Lead-Lag)

## The claim under test

- **The source paper.** Kewei **Hou**, *"Industry Information Diffusion and the Lead-Lag
  Effect in Stock Returns"* (Review of Financial Studies, 2007). Hou shows that within an
  industry, the returns of **big firms lead** the returns of **small firms**: information
  is incorporated into large, closely-followed names first and diffuses to their smaller
  industry peers with a lag. A big-firm return this period predicts small-firm returns next
  period, over and above own-firm momentum — the lead-lag is a *within-industry*, size-based
  information-diffusion effect.
- **The behavioural / frictions reading.** Smaller firms are followed by fewer analysts and
  traded by more slow-moving investors, so sector-wide news reaches their prices late (a
  gradual-information-diffusion mechanism, Hong & Stein 1999). The biggest name is the
  bellwether the rest of the industry catches up to.
- **The specific test here.** We take the self-contained weekly version: assign a liquid US
  cross-section to GICS-style sectors, designate the **largest-cap** name in each as the
  **leader**, and test whether the leader's week-`w` return predicts the **followers'**
  week-`w+1` return. Long the followers whose leader rose, short those whose leader fell,
  with a Newey-West *t*, a permutation placebo, a two-era cut, a dollar-volume leader
  re-designation, a costed timer, and a seeded synthetic positive control. (Weekly mega-cap
  returns are a *conservative* test bed — the effect is documented strongest among small,
  thinly-covered firms, which this survivor panel omits.)

## What we measure, and the honesty rails

- **Weekly returns, no free model.** Each name's daily total-return Close resampled to Friday
  (`W-FRI`), simple weekly returns; the leader signal is the leader's own weekly return.
- **Point-in-time lead-lag, one documented lag.** The signal is the leader's return **through
  the close of week `w`**; the followers are held **week `w+1`**. Zero look-ahead into returns.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the weekly spread — the
  sign-based weekly book is serially correlated, so a plain *t* would overstate significance.
  A one-sample *t* and a pooled Welch *t* (follower returns after up- vs down-leader weeks)
  cross-check. A **1,000-permutation placebo** shuffles the lead→lag week alignment to confirm
  the spread isn't a lucky coincidence of the sort.
- **Leader designation is named on the Signal axis.** `LEADERS` is *today's* largest-cap per
  sector, held static across the sample — Hou's "big firm". A robustness cut re-designates
  leaders by trailing **dollar volume** (a size/liquidity proxy read directly off the tape) and
  reaches the same verdict, so the result does not hinge on one designation.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent, so
  the cross-sectional magnitudes are an **upper bound** — and the small illiquid followers where
  the effect is strongest are exactly the ones missing.
- **The timer is graded separately.** Costs are 2 sides × one-way × NAV on the weekly long-short
  book, and the short book pays borrow — the honest test of whether a small weekly spread survives
  friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the weekly spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Hong, H. & Stein, J. (1999)** — gradual-information-diffusion model underpinning lead-lag
  and momentum effects.
- **Lo, A. & MacKinlay, A. C. (1990)** — the classic size-based lead-lag in weekly portfolio
  autocorrelations (big portfolios lead small), the empirical ancestor of Hou's within-industry test.

## Data sources

- **yfinance daily OHLCV** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- **Sector assignment & largest-cap leaders** — GICS-style sectors and a public-record
  market-cap ranking (leader = the largest-cap member of each sector across the sample:
  AAPL, GOOGL, AMZN, WMT, JPM, JNJ, XOM, GE). Encoded as static maps in
  [`leader_lag/data.py`](../leader_lag/data.py) (`SECTORS`, `LEADERS`); market-cap rankings
  are public record (e.g. companiesmarketcap.com sector leaderboards).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [379-etf-lead-lag](../../379-etf-lead-lag/) — a **basket-vs-member** lead-lag (does the sector
  *ETF* lead its own constituents?), an index-arb / fund-flow channel. This study is
  **name-on-name within an industry**: the single largest *firm* leading its smaller peers, no
  fund in the loop.
- [506-industry-momentum](../../506-industry-momentum/) — sorts **industries against each other**
  on their own trailing returns (Moskowitz-Grinblatt), a cross-industry time-series signal. This
  study is **within** one industry: a leader predicting *other names in the same sector*.
- [538-industry-relative-reversal](../../538-industry-relative-reversal/) — a name's **own**
  deviation from its industry mean *reverses* (a contrarian own-return signal). This study is a
  leader's return **predicting a different name's** forward return — cross-name diffusion, not
  own-name reversal.
- [810-price-delay](../../810-price-delay/) — how slowly a name absorbs the **market** factor (a
  name-level lagged-beta R² decomposition). This study is one **industry name** leading another,
  not a name's delay in loading the aggregate market.

None of the siblings sorts on **the largest-cap firm's return predicting its within-industry
followers** — Hou's own axis — which is this study's own.
