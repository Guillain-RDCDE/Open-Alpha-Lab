# References & literature map — Study 678 (Random-Walk-Index)

## The claim under test

- **The folklore.** Michael Poulos introduced the **Random Walk Index (RWI)** in *Technical
  Analysis of Stocks & Commodities* (1990): compare how far price has actually traveled over the
  last *n* sessions against how far a *pure random walk* with the market's own recent Average
  True Range would be expected to travel over the same *n* sessions. If the real move exceeds
  the random-walk benchmark — **RWI-high > 1** — the claim is that the move is "statistically
  non-random": a real trend, not noise, and therefore worth riding.
    RWI_high(n, t) = (High_t − Low_{t−n}) / (ATR_n(t) × √n)
    RWI_low(n, t)  = (High_{t−n} − Low_t) / (ATR_n(t) × √n)
  Poulos scans several short lookbacks (his own examples use n = 2..7) and reports the single
  highest reading as "the" indicator — the version this study implements (n = 2..6).
- **The mechanism, steelmanned.** The idea borrows its intuition from the same variance-scaling
  logic as a Sharpe ratio or a random-walk hypothesis test: over *n* independent steps of size
  ~ATR, a pure random walk's *expected* net displacement scales like ATR × √n (a standard
  Brownian-motion identity). A displacement that *exceeds* this benchmark is, in principle, less
  likely under the random-walk null — which is exactly the same "trend-strength gate" logic
  behind ADX, the Choppiness Index, the Vertical-Horizontal-Filter and a rolling Hurst exponent.
  See the dedup map below — this desk has now tested that family four times.
- **What we did *not* find any peer-reviewed academic literature defending.** Unlike the FOMC
  vol-crush or the pre-FOMC drift, the RWI has no meaningful academic anchor — it is a
  practitioner indicator that lives entirely in trading-platform documentation (StockCharts,
  Investopedia, TradingView script libraries) and Poulos' original magazine column. We treat the
  claim at face value, as its own promotional material states it, and test it directly.

## What we measure, and the honesty rails

- **RWI-high flag vs no-flag next-session return** — the direct test of "statistically
  non-random predicts returns": the RWI is computed from data through the close of day *t* (it
  needs that day's own High/Low), so the position is entered **at that same close** and earns
  the close(t) → close(t+1) return — a single execution lag, applied once, exactly the
  METHODOLOGY convention. Welch *t* (daily, effectively non-overlapping observations), a
  Newey-West (1987) 5-lag *t* on the flag-dummy regression, and a Wilson (1927) interval on the
  hit rate.
- **A matched-count random-day placebo**, not a naive "is it significant" test alone: does a
  random subset of the *same size* as the flagged set do as well by chance? 20,000 draws (20
  seeds × 1,000).
- **The tradable book vs a block-shuffled random-entry control**, not just buy & hold. Comparing
  a timing rule only to buy & hold conflates "does the *timing* add value" with "was the market
  up over the period" — a rule that's long 57% of the time in a 21-year bull tape will show a
  positive *number* almost by construction. The fair control here chops the flag series into
  contiguous 21-day blocks and randomly re-orders the blocks (20 seeds): this preserves the
  timer's total days-invested count *and* its turnover/run-length texture while destroying any
  real correlation between "the flag fired here" and what happened at that calendar date — the
  benchmark the claim actually has to beat.
- **Cross-instrument pooling** (SPY, QQQ, IWM, DIA, GLD) so one lucky/unlucky tape can't carry
  the verdict either way.
- **No survivorship bias to name.** SPY/QQQ/IWM/DIA are broad index ETFs (not a stock-picking
  panel) and GLD tracks a physical commodity; none of this study's tests condition on any
  cross-sectional membership.

## Data sources

- **Daily total-return-adjusted OHLC**, SPY + basket (QQQ, IWM, DIA, GLD) — yfinance
  (`auto_adjust=True`, no key), cached under `_cache/` (`rwi_spy.csv` etc.), 2005-01-03 →
  2026-06-30. Adjusting Open/High/Low/Close together (not just Close) matters here specifically
  because the RWI is a *range* statistic — an unadjusted split would otherwise print a fake giant
  range on the split day and poison the True Range/ATR window for weeks around it.
- Poulos, M. (1990). *"Random Walk Index"*, Technical Analysis of Stocks & Commodities. Widely
  reproduced in trading-platform documentation, e.g. StockCharts' ChartSchool RWI page and the
  TradingView built-in RWI script, both consistent with the formula implemented in
  [`random_walk_index/strategy.py`](../random_walk_index/strategy.py).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

The RWI is one member of a small family of **trend-strength gates** this desk has now tested
directly — the same underlying question ("can a statistic that reads 'this move is real, not
noise' actually time or filter a trade?") asked with a different formula each time:

- [108-adx-filter](../108-adx-filter/) — Wilder's **ADX(14) > 25** gate on a moving-average
  cross. Verdict: `None` × `Mirage` — the gate removes 80% of signals and makes the ungated rule
  *worse*, not better. Different formula (directional-movement smoothing), same family question.
- [484-vertical-horizontal-filter](../484-vertical-horizontal-filter/) — White's **VHF** (a
  range-over-path-length ratio) gating a momentum entry. Verdict: `None` × `Mirage` — the gate
  ties a drift-matched random baseline; the "trending now" timing carries no information.
- [397-hurst-regime](../397-hurst-regime/) — a rolling **R/S Hurst exponent** used to switch
  between trend-following and mean-reverting styles. Verdict: `None` × `Mirage` — real-market
  Hurst estimates sit pinned above 0.5 almost always, and where the switch does fire it carries
  no forecasting power.
- **This study (678-random-walk-index)** — Poulos' **RWI** (a realized-displacement-over-ATR
  ratio) used as a **long timer**: be long only when RWI-high > 1. None of the siblings test
  this specific formula, and none frame it as a standalone timer rather than a gate on a separate
  entry rule — but the result rhymes: `None` × `Mirage`, and here the flag-day/no-flag-day split
  even lands **wrong-signed** at pooled significance (Welch *t* = −2.16), the strongest
  disconfirmation of the four.
- [495-choppiness-index](../495-choppiness-index/) — Chande's **Choppiness Index**, the
  logarithmic-range cousin of the VHF, gating the same family of entries. Not yet built at the
  time of writing; when it lands, expect the same rhyme — a different formula testing the same
  "trend-strength gate" hypothesis this desk keeps busting.

Four tests, four formulas (ADX's directional smoothing, VHF's range ratio, Hurst's R/S exponent,
RWI's ATR-scaled displacement), the same verdict every time: a statistic that *looks* like it is
measuring "is this a real trend" does not, in practice, forecast what the trend does next.
