# References & literature map — Study 657 (Larry Portfolio)

## The claim under test

- **The folklore.** Larry Swedroe's "Larry Portfolio" (the name comes from Bill Schultheis,
  who described it in *The Coffeehouse Investor*; Swedroe details and defends it across his
  own books from the early 2000s on, e.g. *Rational Investing in Turbulent Times* (2003) and
  later *Reducing the Risk of Black Swans* (2018, with Kevin Grogan)): concentrate the
  equity-risk budget in the highest-expected-return equity factor — small-cap **value** —
  and hold a SMALL slice of it (~30%), parking the rest (~70%) in safe, short/intermediate
  bonds. The pitch: because small-value has earned a higher expected return per unit of
  equity risk than the broad market, a much smaller equity sleeve can deliver returns
  comparable to a conventional 60/40 while running far less total equity exposure — lower
  volatility, shallower drawdowns, lower correlation to a pure-equity crash.
- **The academic anchor.** Banz (1981, *The relationship between return and market value of
  common stocks*, JFE) for the size premium and Fama & French (1992, 1993, JF/JFE) for the
  value premium and the three-factor model that formalised both as priced risk factors. The
  Larry Portfolio is, in spirit, a leveraged bet on the size-times-value corner of that
  model — small AND cheap, the cell with the largest historical premium in the original
  Fama-French sample-splitting tables.
- **The honest caveat, said upfront.** Both factors are contested. This desk has already
  torn down the plain versions of each: [513-size-effect](../513-size-effect/) finds no
  size premium on a modern survivor basket, and
  [530-book-to-market-value](../530-book-to-market-value/) finds no book-to-market value
  premium either. The Larry Portfolio is a genuinely different test — it doesn't rank a
  basket, it buys a small-cap-value **ETF** (IJS) and asks whether the resulting *portfolio*
  can match a 60/40 — but if the underlying premia are absent, the portfolio-construction
  trick has nothing to lean on.

## What we measure, and the honesty rails

- **The headline race.** A fixed 30% IJS / 70% IEF blend vs a fixed 60% SPY / 40% IEF blend,
  both rebalanced annually (2 bps one-way cost on turnover), CAGR/vol/Sharpe(excess-of-cash)/
  max-drawdown for each, plus the standalone legs (100% SPY, 100% IJS, 100% IEF) for context.
  Identical rebalance convention and Treasury leg (IEF) to sibling study
  [97-balancing-act](../97-balancing-act/), so the 60/40 numbers are directly comparable
  across the two studies (same window, same construction).
- **"Does it match 60/40's return?"** tested two ways: a Newey-West (1987) HAC *t* on the
  daily (Larry − 60/40) return difference, and a circular block bootstrap (21-day blocks,
  2,000 resamples — long enough to span a full trading month of serial correlation) 95% CI
  on the mean-return difference.
- **"Is the risk-adjusted edge real?"** a *separate* bootstrap on the Sharpe-ratio
  *difference* (not the same test as the return-difference HAC *t*, which would just cancel
  the shared cash leg) — because a ratio's sampling distribution isn't the mean-difference's.
- **The equity-risk claim** (vol, drawdown, correlation to SPY) needs no inferential test —
  it follows mechanically from running 30% vs 60% equity weight and is reported as arithmetic
  fact, not a statistical finding.
- **The small-value premium's own decay**, the load-bearing assumption underneath the whole
  construction: whole-sample HAC *t* on the daily (IJS − SPY) spread, plus a **justified,
  not-snooped** era split at **2007-01-01** (documented externally — AQR's and Swedroe's own
  writing both date the value factor's drawdown to the post-GFC years), tested as a Welch *t*
  of the era **difference**, exactly the pattern sibling study 637 uses for its
  press-conference-era split.

## Data sources

- **IJS, IEF, SPY, SHY daily auto-adjusted (total-return) closes** — yfinance (no key),
  cached under `_cache/` (`lp_prices.csv`), 2002-07-30 → 2026-06-30 (IEF/SHY's shared
  inception date is the binding joint-window constraint; IJS itself lists 2000-07-24).
- Swedroe's own case for the portfolio: Larry Swedroe & Kevin Grogan, *Reducing the Risk of
  Black Swans* (2018) and his long-running CBS MoneyWatch / Alpha Architect columns on
  "tilting" a portfolio toward small-cap value.
- Banz (1981); Fama & French (1992, 1993); Asness, Frazzini, Israel & Moskowitz (2015,
  *Fact, Fiction and Value Investing*, JPM) on the value factor's post-2007 drawdown; Fama &
  French (2020s updates, Dartmouth data library) for the long-run factor history this
  desk's real tape (bounded by IEF/SHY's 2002 inception) cannot itself reach back to.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [513-size-effect](../513-size-effect/) — the plain **size** premium (small vs large,
  long-short) on a 40-name survivor stock basket. This study buys a small-cap-value **ETF**
  as one leg of a fixed two-asset *portfolio*, not a long-short factor spread — and finds
  the same underlying absence from a different angle.
- [530-book-to-market-value](../530-book-to-market-value/) — the plain **value** premium
  (book-to-market long-short) on fundamentals data. Same relationship: different test
  design, same "premium not certifiably present" conclusion.
- [655-ivy-portfolio](../655-ivy-portfolio/) — Faber's 5-asset **equal-weight** endowment
  mix (equity/foreign/REITs/bonds/commodities), optionally timed by a 10-month SMA. A
  *diversification* story across five uncorrelated-ish sleeves, not a *concentration* story
  in one high-expected-return factor — the opposite construction philosophy from the Larry
  Portfolio's "small slice of the best factor, big slice of safety."
- [68-all-weather](../68-all-weather/) — volatility-weighted risk parity across a broader
  macro basket. Also diversification-by-uncorrelated-macro-regime, not factor concentration.
- [97-balancing-act](../97-balancing-act/) — the plain fixed **60/40** (SPY/IEF) this study
  races against, using the identical rebalance/cost/window convention so the numbers line up
  directly. 97 asks if 60/40 beats 100% stocks; this study asks if a smaller, factor-tilted
  equity sleeve can match 60/40.

None of the siblings test whether **concentrating a small equity sleeve in small-cap value
lets you shrink total equity exposure for free** — the Larry Portfolio's own claim is this
study's axis, even though its building blocks (the size and value premia) have already been
torn down individually elsewhere on this desk.
