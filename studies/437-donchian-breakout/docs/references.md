# References & literature map — Study 437 (Donchian-Breakout)

## The claim under test

- **Richard Donchian** — the inventor of the channel breakout. The "Donchian channel"
  plots the highest high and lowest low of the prior *N* sessions; the trading rule buys a
  new *N*-day high and sells a new *N*-day low. Donchian's 1970s trend-following work (the
  "5-and-20" and "weekly rule" systems) made channel breakouts the archetypal trend entry.
- **The Turtle Traders (Richard Dennis & William Eckhardt, 1983).** The famous Turtle
  experiment built its **System 1** on a *20-day* Donchian breakout (with a 55-day System 2
  fallback), plus volatility-based position sizing (the "N"/ATR unit), pyramiding, and a
  10-day opposite-channel exit. Curtis Faith's *Way of the Turtle* (2007) and the leaked
  *Original Turtle Trading Rules* document the recipe. The folklore that survives in retail
  trading forums today is the **20-day breakout on its own** — which is exactly what this
  study isolates: *does the 20-day channel hold up sans the rest of the Turtle apparatus?*
- **The steelman.** Channel breakouts are a binary, parameter-light expression of
  time-series momentum: a new *N*-day high is, by construction, evidence that the recent
  trend is up. If markets trend — if returns have positive autocorrelation at the swing
  horizon — then entering on the breakout and riding it should harvest that premium. The
  Turtles reportedly turned a few thousand dollars into tens of millions on this skeleton.

## The statistical and financial literature behind the rule

- **Time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*,
  Journal of Financial Economics — documents a significant trend premium across 58 liquid
  futures. The channel breakout is a coarse, binary proxy for the same signal; where TSMOM
  is strongest (diversified futures), breakouts work best.
- **Moving-average and channel trading rules.** Brock, Lakonishok & LeBaron (1992),
  *Simple Technical Trading Rules and the Stochastic Properties of Stock Returns*, Journal
  of Finance — the canonical early test of MA and channel (trading-range) breakout rules;
  reported pre-1987 outperformance. Sullivan, Timmermann & White (1999), *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap*, Journal of Finance — re-examine
  the *same* rules under White's Reality Check and find the apparent edge largely vanishes
  once you correct for the universe of rules searched. This is the central caution for any
  single-window breakout claim, and the spirit of our permutation placebo.
- **Where breakouts work vs where they don't.** Hurst, Ooi & Pedersen (2017), *A Century of
  Evidence on Trend-Following Investing*, Journal of Portfolio Management — trend rules earn
  a real, diversified premium *across many assets*, but the per-market signal is weak and the
  benefit is mostly diversification, not any single instrument. A 20-day breakout on a single
  long-only equity index (the SPY test here) is the *least* favourable case — equities trend
  up secularly, so a flat-when-below rule mostly just reduces exposure.
- **Post-publication decay / data-snooping.** Bajgrowicz & Scaillet (2012), *Technical
  Trading Revisited: False Discoveries, Persistence Tests, and Transaction Costs*, Journal of
  Financial Economics — after multiple-testing correction and costs, technical rules'
  profitability does not persist out of sample. Our SPY result (timing placebo *p* = 0.83)
  is consistent: the channel's days-in-market are no better than random.

## Why the rule can work — the mechanism, and the synthetic control

- **Return autocorrelation is the necessary ingredient.** A breakout only pays if a new
  *N*-day high predicts above-average forward returns, i.e. if log-returns are positively
  autocorrelated at the swing horizon. Our synthetic generator (`data.synthetic_panel`) is
  an AR(1) in log-returns with a single **`edge`** knob = the autocorrelation coefficient:
  at `edge = 0` the tape is a random walk (no trend to harvest, the null), at `edge = 0.4`
  it has strong persistence. The breakout banks the planted edge only when it exists
  (`t` vs buy-and-hold = +2.24, placebo *p* = 0.044) — proving the harness is faithful and
  that a *zero* edge cannot manufacture significance.
- **Why SPY is the unfriendly case.** A single liquid equity index has tiny daily return
  autocorrelation and a strong upward drift; a long/flat breakout spends ~38% of the time in
  cash and therefore mostly forgoes drift, which is why its Sharpe trails buy-and-hold even
  though it cuts drawdown. Drawdown reduction from a 62%-invested rule is not a free lunch —
  it is just lower beta, available far more cheaply via a static stock/cash split.

## Related desk studies

- **[Study 103 — Turtle-Trader](../../103-turtle-trader/)**: the *full* Turtle system
  (channel entry + ATR sizing + pyramiding + opposite-channel exit). This study is the
  controlled subtraction — the 20-day channel *alone* — so the two read together as
  "how much of the Turtle result is the breakout, and how much is the rest?"
- **[Study 110 — Faber-Timing](../../110-faber-timing/)**: the 200-day SMA in/out rule — the
  same long/flat-timing family, the same random-timing control. Faber lands `REAL` (on the
  *risk* axis) because the slow SMA genuinely times volatility; the fast 20-day channel does
  not. The contrast is instructive: window and mechanism matter.
- **[Study 128 — Keltner-Channel](../../128-keltner-channel/)** and
  **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: the *band*
  cousins — ATR / standard-deviation envelopes rather than the rolling high/low channel.
  Different envelope, same family of "price pierced a band, now what?" questions.
- **[Study 436 — MA-Envelopes](../../436-ma-envelopes/)**: percentage envelopes around a
  moving average — another channel-style breakout/reversion test on the same SPY tape.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.summary`](../donchian_breakout/strategy.py) and `sharpe_diff_tstat`.
- **Return-difference t-stat (Sharpe comparison).** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance — the
  Donchian-vs-BH / vs-SMA / vs-random head-to-heads.
- **Block / permutation testing.** Politis & Romano (1994), *The Stationary Bootstrap*,
  JASA, and White (2000), *A Reality Check for Data Snooping*, Econometrica — the
  block-shuffle placebo in [`strategy.permutation_pvalue`](../donchian_breakout/strategy.py)
  preserves the position-clustering while destroying alignment with the return tape.

## Data sources

- **SPY daily total-return closes** (via `yfinance`, `auto_adjust=True`), 1993-01-29 to
  2026-06-12 — the canonical liquid US-equity test instrument; split- and dividend-adjusted,
  which is essential for the multi-decade buy-and-hold comparison.
- **Panel:** QQQ, IWM, TLT, GLD, USO, UUP (same source, same adjustment) — broad equity,
  small-cap, long Treasuries, gold, crude, dollar — the classic diversified trend basket in
  spirit, to test whether the rule generalises beyond SPY. (IWM's cached history begins
  2016; a short-span caveat applies to that tape only.)
- **Cash leg:** 0% (a conservative flat baseline; the excess-vs-excess race is unaffected
  since both arms net the same cash rate). Borrow on the short leg = 50 bps/yr.
