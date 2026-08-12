# References & literature map — Study 905 (Residual Reversal)

## The claim under test

- **The source paper.** David **Blitz, Joop Huij, Simon Lansdorp & Marno Verbeek**,
  *"Short-Term Residual Reversal"* (Journal of Financial Markets, 2013, vol. 16). They
  show the well-known one-week reversal (Jegadeesh 1990; Lehmann 1990) is largely driven
  by exposure to **common factors** and by microstructure noise. Regressing weekly stock
  returns on a factor model and reversing on the **residual** produces a signal that is
  roughly twice as strong, far more stable, and — they argue — better able to survive
  transaction costs than the raw reversal, because it no longer bets on factor mean-
  reversion or harvests bid-ask bounce.
- **The mechanism.** A raw one-week loser may simply be a high-beta name that fell with
  the market; buying it is a factor bet, not a mispricing bet, and it drags the book into
  systematic exposures. Removing the fitted factor return isolates the **idiosyncratic**
  move — the part most plausibly driven by temporary liquidity-demand / overreaction that
  actually reverses.
- **The microstructure trap.** Reversal measured with last week's close both forming the
  signal and pricing the entry is the textbook **bid-ask bounce** illusion (a name that
  closed on the bid prints a low return and "reverts" upward purely because the next print
  lands on the ask). Bounce is largest in illiquid names, so we add a **dollar-volume
  liquidity screen** (top 60%) — the standard defence.
- **The specific test here.** We take the self-contained version on a liquid US
  cross-section: each name's **weekly market-model residual** (a trailing-52-week rolling
  OLS on the equal-weight market — a single-factor proxy, the coarsest honest version of
  the paper's multi-factor model), sorted point-in-time (signal known at the close of
  `w−1`, one shift), with the **raw** weekly reversal placed beside it as the foil, a
  Newey-West *t*, a permutation placebo, a two-era cut, a costed timer, and a seeded
  synthetic positive control. (A single-factor market residual on 50 mega-caps is a
  conservative floor: the paper's fuller factor model and broader universe would clean
  more, but the raw signal we start from is itself absent here.)

## What we measure, and the honesty rails

- **Market-model residual, no free model.** For each name, a trailing-52-week rolling OLS
  of the weekly return on the equal-weight market, computed vectorised via rolling
  cov/var (`β`) and rolling means (`α`); the residual of week `w` uses only data through
  `w`.
- **Point-in-time sort, one documented lag.** The ranking signal is the residual **known
  at the close of `w−1`** (`.shift(1)`); the book is held on week `w`. Zero look-ahead.
- **Liquidity screen.** Each week only the top 60% of names by trailing dollar volume are
  eligible — bid-ask bounce is an illiquid-name artefact, so this is where the raw
  reversal would otherwise pick up phantom edge.
- **Self-financing spread.** The book is dollar-neutral long-short, so the spread is
  already an excess-of-cash quantity (the cash legs cancel); no separate T-bill subtraction
  is needed on the Signal axis.
- **Robust inference.** Newey-West (HAC, Bartlett, 8-lag) *t* on the weekly long-short
  spread — an overlapping-formation signal is serially correlated. A one-sample *t* and a
  pooled Welch *t* (loser book vs winner book) cross-check. A **1,000-permutation placebo**
  breaks the signal → forward-return link.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). The magnitudes are an **upper
  bound** — and they are still zero.
- **The timer is graded separately.** A weekly reversal turns the book over almost fully;
  costs are 2 sides × one-way × NAV per weekly rebalance, plus borrow on the short.

## Shared method citations

- **Jegadeesh, N. (1990)** — "Evidence of Predictable Behavior of Security Returns"; the
  one-month reversal this study residualises.
- **Lehmann, B. (1990)** — "Fads, Martingales, and Market Efficiency"; the weekly contrarian
  strategy at the root of short-term reversal.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily OHLC + Volume** (`auto_adjust=True`, total-return), 50 liquid US
  large-caps, 2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [329-one-month-reversal](../../329-one-month-reversal/) — the classic **monthly**
  Jegadeesh reversal on **raw** returns, with no factor cleaning. This study residualises
  the **weekly** version and puts the raw weekly reversal beside it as the foil.
- [800-high-frequency-reversal](../../800-high-frequency-reversal/) — very-short-horizon
  (daily/intraday) **raw** reversal. This study works at the **weekly** horizon on the
  market-model **residual**.
- [377-bid-ask-bounce](../../377-bid-ask-bounce/) — isolates the microstructure **bounce**
  itself as the object of study. Here the bounce is the *contaminant* the residual + the
  liquidity screen aim to strip, not the signal being harvested.
- [237-residual-momentum](../../237-residual-momentum/) — residual **momentum** (Blitz-
  Huij-Martens): the *continuation* of the residual over a long formation window. This is
  the opposite sign at the opposite horizon — short-term residual **reversal**.

None of the siblings sort on the **short-horizon weekly market-model residual with a
liquidity screen** — the Blitz-et-al residual-reversal signal — which is this study's own
axis.
