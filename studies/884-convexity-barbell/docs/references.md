# References & literature map — Study 884 (Convexity Barbell)

## The claim under test

- **The structure.** A **barbell** holds the short and long ends of the curve; a **bullet**
  holds the belly. Weight the barbell (here SHY + TLT) so its **duration** equals the
  bullet's (IEF), and the two share the same first-order rate exposure — but the barbell has
  **higher convexity**, because convexity is (roughly) proportional to the *square* of
  maturity/duration and a `w·D_short + (1-w)·D_long = D_belly` mix, being spread across the
  wings, carries more curvature than the concentrated belly. The second-order term of the
  price/yield relation, `ΔP/P ≈ −D·Δy + ½·C·(Δy)²`, is therefore larger for the barbell, so
  for any given yield move it should lose less (or gain more) than the bullet: **at equal
  duration, more convexity ⇒ out-performance when yields move a lot.**
- **The catch the desk tests for.** Convexity is not free. (1) The market **prices** it: a
  more convex bond trades at a **lower yield**, so the barbell gives up carry — in a calm
  tape it under-earns. (2) The duration match only neutralises a **parallel** shift; the
  barbell is long the wings and short the belly, so it is exposed to **curve reshaping**
  (a butterfly / "belly cheapens vs wings" move) that the bullet is not. The empirical
  question is whether the convexity pickup survives the yield give-up and the curve risk.
- **The specific test here.** We rebuild the barbell from three liquid iShares Treasury ETFs
  (SHY 1-3y, IEF 7-10y, TLT 20y+) plus BIL cash, estimate each ETF's empirical duration
  (its beta to an equal-weight rates factor), duration-match the SHY+TLT barbell to the IEF
  bullet each day, and compare total return, Sharpe (excess-of-cash), the convexity
  regression + smile, drawdown, a calendar-year table, a two-era cut, a costed timer, and a
  seeded synthetic positive control.

## What we measure, and the honesty rails

- **Empirical duration, no free model.** The rates factor is the equal-weight mean of the
  three bond daily returns; each ETF's empirical duration is the trailing-252d
  `Cov(r_i,f)/Var(f)`, computed vectorised. Duration is measured, not assumed from a
  vendor sheet.
- **The duration match, one documented lag.** The match weight `w = (β_TLT−β_IEF)/(β_TLT−β_SHY)`
  is built from betas **known at the close of `t−1`** (`.shift(1)`) and held on day `t`.
  Zero look-ahead. The residual duration slope of the spread on the factor (≈ 0) confirms
  the first-order match.
- **Total return, excess-vs-excess.** All series are dividend-reinvested total return
  (`auto_adjust=True`). The Sharpe race is run on **excess-of-BIL** returns for both books,
  so a common cash rate cannot flatter either side.
- **Convexity, measured honestly.** The return-based factor is a near-linear proxy for the
  true yield change (`corr ≈ −0.99`), so the `f²` regression slope recovers the *sign and
  relative size* of the convexity pickup — not an absolute analytical convexity, which
  returns-only data cannot pin down. The convexity **smile** (mean spread by absolute-move
  quintile) is the model-free cross-check.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily spread — an
  overlapping-formation signal is serially correlated, so a plain *t* overstates
  significance. A **circular block bootstrap** CI for the mean spread; a **leg-permutation
  placebo** that breaks the day-by-day alignment of the two barbell legs; a **two-era** cut.
- **The timer is graded separately.** One-way cost × the barbell's daily rebalancing
  turnover (the bullet is buy-and-hold) — the honest test of whether any spread survives
  friction.

## Shared method citations

- **Fabozzi, F. J.** — *Bond Markets, Analysis, and Strategies* — the textbook duration /
  convexity decomposition and the barbell-vs-bullet trade-off (a barbell has more convexity
  than a duration-matched bullet, paid for with a lower yield).
- **Ilmanen, A. (2011)** — *Expected Returns* — convexity is priced: more convex positions
  earn a lower yield, so convexity is a *cost* in calm markets and a *benefit* only in
  volatile ones; the net is an empirical question.
- **Litterman, R. & Scheinkman, J. (1991)** — *Common Factors Affecting Bond Returns* — the
  curve moves in level / slope / curvature factors; a duration-matched barbell is neutral to
  *level* but exposed to *curvature* (the butterfly), which is why the convexity is not free.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread).
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap (the
  spread-mean CI).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily total-return closes** (`auto_adjust=True`), SHY / IEF / TLT + BIL,
  2010-01-04 → 2026-06-30, cached under `_cache/`.
- All headline numbers are pinned in [`results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [59-downhill](../../59-downhill/) and [380-curve-roll-down](../../380-curve-roll-down/) —
  **roll-down / riding the curve**: a *carry* trade that harvests the price gain of a bond
  ageing down a upward-sloping curve. This study is a **duration-matched convexity** compare
  — a second-order (curvature) bet, not a first-order roll/carry bet.
- [826-treasury-duration-bab](../../826-treasury-duration-bab/) — a **beta-neutral
  betting-against-beta** long-short across five maturity buckets (levered low-duration vs
  short high-duration). This study is a **long-only, duration-*matched*** barbell-vs-bullet
  compare — no leverage, no beta-neutral short book; the axis is convexity, not the low-risk
  alpha.
- [581-term-premium](../../581-term-premium/) — a **time-series** *when-to-own-duration*
  timer on TLT (a directional level bet). This study takes **no directional duration view**
  — barbell and bullet are duration-matched by construction; the only difference is
  convexity.

None of the siblings run a **duration-matched barbell-vs-bullet convexity compare** — this
study's own axis.
