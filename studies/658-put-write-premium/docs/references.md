# References & literature map — Study 658 (Put-Write-Premium)

## The claim under test

- **The folklore.** "Sell a cash-secured, at-the-money S&P 500 put every month, roll it
  forever, and you harvest the **variance risk premium** — implied volatility runs richer than
  what subsequently realizes — for a smoother, better risk-adjusted ride than just holding the
  index." The pitch is sold both retail (option-income newsletters, the "Wheel" community) and
  institutionally (CBOE's own PUT-index marketing, "equity premium income" funds).
- **The academic anchor.** Whaley (2002, *Return and risk of CBOE buy-write monthly index*,
  Journal of Derivatives) and the CBOE's own methodology papers on the **PUT index** (1986→)
  document that, over that long sample, the raw index roughly matched the S&P 500's return at
  about two-thirds the volatility — i.e. a *higher* long-run Sharpe. Israelov & Nielsen (2015,
  *Covered calls uncovered*) and Israelov & Klein (2016) are the standard rebuttal: most of that
  apparent Sharpe gain is explainable as **selling equity beta at a smaller size**, not a
  distinct risk-adjusted edge — precisely the question this study tests directly with a CAPM
  regression rather than assuming it.
- **What we actually test.** Not the 1986→ CBOE PUT *index* (untradeable, no fees, no real
  fills) but **PUTW**, the only liquid ETF that implements the same methodology — its entire
  live history, 2016-02-24 → 2026-06-30. That honesty cuts the sample to ~9.4 years and one
  historically strong equity bull regime; we name that trade-off rather than borrow the index's
  longer, friendlier sample to make the tradable product look better than its own tape.

## What we measure, and the honesty rails

- **Excess-of-cash return, HAC-*t*'d (10 daily lags).** PUTW and SPY both measured against
  **BIL** (the T-bill ETF), the same cash-proxy convention as sibling study 655-ivy-portfolio,
  so every Sharpe and every excess-return number on this page is excess-of-cash, never raw.
- **CAPM alpha/beta, not a naive Sharpe comparison.** The honest version of "is the premium
  real" is whether PUTW earns anything **beyond** what its equity beta already explains — a
  regression of `PUTW − BIL` on `SPY − BIL`, HAC-robust. A significant, positive alpha would be
  the real find; an insignificant one (what the real tape shows) means "the premium" is not
  distinguishable from smaller beta.
- **Crash-conditional beta, not a single point estimate.** An interaction term (a crash-day
  dummy, threshold SPY ≤ −3% on the day, chosen ex-ante and not tuned to the sample) tests
  whether the average-day beta discount survives exactly the days it is supposed to matter —
  the honest version of "does put-writing cushion a crash" instead of eyeballing two named
  drawdown windows in isolation.
- **Sharpe difference via circular block bootstrap** (block = 21 trading days ≈ one option
  roll), not a raw point-estimate gap — so "did it beat SPY risk-adjusted" carries an interval,
  not a single number that could flip sign in a friendlier sub-sample.
- **No survivorship to name.** Both PUTW and SPY are live, currently-listed products measured
  over their entire respective histories in the window — there is no basket of dead peers this
  study conditions away.

## Data sources

- **PUTW / SPY / BIL daily auto-adjusted (total-return) closes** — yfinance (no key), cached
  under `_cache/` (`pwp_prices.csv`), 2016-02-24 → 2026-06-30. WisdomTree PUTW fund page:
  https://www.wisdomtree.com/investments/etfs/alternatives/putw — methodology tracks the CBOE
  S&P 500 PutWrite Index (ticker `PUT`). Cboe PUT index page:
  https://www.cboe.com/index/dashboard/PUT
- **Named crash windows** ("Volmageddon" 2018-01-26 → 2018-02-09; COVID crash 2020-02-19 →
  2020-03-23) are hardcoded, conventional peak-to-trough dates for those episodes, in
  [`data.py`](../put_write_premium/data.py) — no network.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

This study is the **naked-put side** of the option-income family — the desk's other
short-volatility studies each test a *different* leg or a *different* underlying risk, and none
of them runs the PUTW-vs-SPY CAPM/crash-beta test this study runs:

- [62-premium-seller](../62-premium-seller/) — the **covered-call** side (QYLD vs its own QQQ
  underlying): a different leg (short calls, holding the equity) and a different underperformance
  mechanism (return-of-capital NAV erosion). This study never sells a call and never holds the
  equity outright.
- [337-covered-call-etf](../337-covered-call-etf/) — the broader buy-write ETF family (QYLD,
  XYLD, RYLD, JEPI/JEPQ) vs SPY **total return**, same covered-call leg as 62, tested across
  four funds. Again the call side, not the put side.
- [354-the-wheel](../354-the-wheel/) — sells **both** legs in sequence (ATM put, then ATM call
  on assignment, forever), priced synthetically off the real VIX with Black-Scholes rather than
  a real fund's own NAV. This study tests only the put leg, and on a *real, investable ETF's*
  live returns rather than a model-priced replica.
- [130-vol-risk-premium](../130-vol-risk-premium/) — the **raw** variance risk premium itself
  (implied VIX vs trailing realized vol, no option structure, no fund): establishes that the
  premium *exists* in the vol surface (HAC *t* = +22.9) but does not test whether any tradable
  wrapper actually delivers it net of beta. This study is the direct follow-on: does the
  *fund* that is supposed to harvest that premium actually show up with alpha beyond beta? (It
  does not.)
- [617-crash-insurance-cost](../617-crash-insurance-cost/) — the **buyer's** side of the same
  variance risk premium (long-vol/tail-hedge products bleed because sellers are compensated).
  This study is the mirror image: the seller's side, on the specific vehicle (PUTW) that claims
  to bank that same premium as an equity substitute.

None of the siblings run the CAPM-alpha-vs-beta test or the crash-conditional-beta test this
study runs on PUTW's own live tape — that pairing is this study's own axis.
