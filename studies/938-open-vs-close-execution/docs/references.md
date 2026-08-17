# References & literature map — Study 938 (Open or Close)

## The claim under test

- **The practitioner claim.** Every published tactical back-test quietly picks an execution
  venue — "rebalance at the next open" or "rebalance at the next close" — and almost none
  report what the other choice would have done. The folk belief cuts both ways: one camp says
  fill at the **open**, because the signal-confirming move continues through the session and
  waiting until the close means paying up; the other says fill at the **close**, because the
  closing auction is the deepest liquidity event of the day and the opening auction is where
  the spread is widest. Both are testable, and they make opposite predictions about the same
  sliver of return.
- **Why the question is well posed.** With one execution lag fixed (signal through the close
  of `t`, fill on `t+1`), the two venues hold an *identical* book on every non-trade day. The
  entire difference is `Δweight × intraday return` on the trade days. So the venue question
  reduces to: **is the open-to-close move on entry days systematically different from the one
  on exit days?** Everything else cancels.

## Why the answer could be non-zero — the mechanisms

- **The overnight/intraday split is not neutral.** Cooper, Cliff & Gulen (2008), *Return
  Differences between Trading and Non-Trading Hours*, and Lou, Polk & Skouras (2019),
  *A Tug of War: Overnight versus Intraday Expected Returns*, Journal of Financial Economics
  134(1) — nearly all of the long-run equity premium accrues overnight, with the trading
  session roughly flat and prone to reversal. If that asymmetry applied *unconditionally*, a
  venue choice would still not matter here: entries (Δw = +1) and exits (Δw = −1) alternate, so
  a constant intraday drift cancels over a full cycle. Only a **conditional** difference moves
  the needle.
- **Short-horizon continuation right after a signal flip.** Moskowitz, Ooi & Pedersen (2012),
  *Time Series Momentum*, JFE 104(2), and Faber (2007), *A Quantitative Approach to Tactical
  Asset Allocation*, Journal of Wealth Management 9(4) — the rules tested here are Faber's, and
  if trend persistence operates at the daily as well as the monthly horizon then the session
  after a bullish flip should drift up and the session after a bearish flip should drift down,
  making the open fill strictly better. That is exactly the hypothesis this study measures.
- **The opening auction is the expensive one.** Barclay & Hendershott (2003), *Price Discovery
  and Trading After Hours*, Review of Financial Studies, and the large literature on the
  closing-auction share of daily volume (see e.g. Bogousslavsky & Muravyev, 2023, *Who Trades
  at the Close?*, Journal of Financial Economics) — the closing auction concentrates the day's
  liquidity, while the opening auction reopens after an information gap with wide spreads. Any
  measured open-fill advantage has to survive that asymmetric friction, which is why the
  opening penalty is carried as an explicit swept PROXY rather than assumed to be zero.

## Why the answer is plausibly zero

- **Trade counts are tiny.** A 10-month filter on a single ETF flips ~2 times a year. Nineteen
  years of tape buys 31–37 observations — the venue question is *structurally* low-powered, and
  a single |*t*| ≥ 2 among eight tape × rule cells is what chance delivers. Harvey, Liu & Zhu
  (2016), *…and the Cross-Section of Expected Returns*, Review of Financial Studies, on the
  multiple-testing bar this implies.
- **Rebalance timing luck.** Hoffstein, Faber & Braun (2019), *Rebalance Timing Luck: The
  (Dumb) Luck of Smart Beta* — an arbitrary implementation choice inside an identical rule
  produces large, unforecastable dispersion in the realised track record. This study is the
  *intraday* instance of that finding: the venue choice is a coin flip whose realised
  contribution ranges over roughly ±40 bps/yr while its expectation sits on zero.

## Related desk studies (dedup)

- **[Study 01 — The Overnight Anomaly](../../01-overnight-anomaly/)** and
  **[Study 788 — Overnight / Intraday Tug of War](../../788-overnight-intraday-tug-of-war/)**:
  both measure the overnight-versus-intraday split as a *return premium* to be harvested
  (unconditionally in 01, cross-sectionally sorted in 788). Study 938 does not trade the split
  at all — it holds the split fixed and asks whether *where a pre-existing rule fills* moves
  its P&L, which turns on the split's **conditional** behaviour on the handful of trade days.
- **[Study 110 — Faber Timing](../../110-faber-timing/)**: the same moving-average rule, tested
  for whether it beats buy-and-hold. Study 938 takes the rule as given and audits its
  *implementation*.
- **[Study 836 — Rebalance Timing Luck](../../836-timing-luck/)**: the phantom dispersion from
  choosing *which day of the month* to rebalance, on a synthetic null. Study 938 is the
  neighbouring axis on the **real tape**: which *moment of the day* you fill. Same family of
  arbitrary choice, different dimension, and here measured against four live ETF tapes.
- **[Study 937 — Tranched Rebalancing](../../937-tranched-rebalancing/)**: 836's real-tape
  sequel — the same *day-of-month* cone, plus the tranching fix that closes it. The two studies
  bracket 938: 937 measures a **date** lottery worth ~0.26 of a Sharpe point and can be
  *diversified away* by splitting the book across dates; 938 measures the **intraday** lottery
  one level down, which is smaller (~±0.4 pp/yr) and cannot be tranched at all — you fill at
  one moment or the other. Neither is a return; both are dispersion.
- **[Study 352 — Opening Range Breakout](../../352-opening-range-breakout/)** and
  **[Study 80 — Cold Open](../../80-cold-open/)**: intraday rules that *trade* the opening
  session as a signal. Study 938 uses the opening session only as a fill point for a rule
  formed entirely on daily closes.
- **[Study 640 — Gold Overnight](../../640-gold-overnight/)**: the overnight effect in a single
  commodity ETF — a return-premium study, not an execution study.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3) —
  [`strategy.newey_west_t`](../open_close_exec/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA
  89(428) — [`strategy.block_bootstrap_ci`](../open_close_exec/strategy.py); 21-day blocks so
  the clustering of trade days survives resampling.
- **Wilson score interval.** Wilson (1927), *Probable Inference, the Law of Succession, and
  Statistical Inference*, JASA 22(158) — [`strategy.wilson_interval`](../open_close_exec/strategy.py),
  used on the open-fill win rate where the normal approximation is unsafe near 50%. It assumes
  independent draws, which the pooled fills are **not** — so it is reported beside the clustered
  interval, never instead of it.
- **Cluster-robust inference / cluster bootstrap.** Petersen (2009), *Estimating Standard Errors
  in Finance Panel Data Sets*, Review of Financial Studies 22(1), and Cameron, Gelbach & Miller
  (2008), *Bootstrap-Based Improvements for Inference with Clustered Errors*, Review of Economics
  and Statistics 90(3) — [`strategy.cluster_t`](../open_close_exec/strategy.py) and
  [`strategy.cluster_bootstrap_trades`](../open_close_exec/strategy.py). Four correlated ETF tapes
  flip their signals on the same days, so the 538 fills sit on 361 dates; clustering on the trade
  date is the minimum honest correction and it moves the *t* from +0.33 to +0.23.
- **Welch's unequal-variance t.** Welch (1947), *The Generalization of "Student's" Problem…*,
  Biometrika 34 — [`strategy.welch_t`](../open_close_exec/strategy.py), for the entry-versus-exit
  intraday spread.

## Data sources

- **SPY, IWM, EEM, EFA** (US large, US small, emerging, developed ex-US) and **BIL** (1-3 month
  T-bill / cash leg) — daily **adjusted OHLC** via `yfinance` (`auto_adjust=True`), cached in the
  shared desk cache as `ohlc_<TICKER>_1d.parquet`. Yahoo applies one adjustment factor per day
  to all of O/H/L/C, so the **intraday** leg (open → close) is a clean **price-only** return and
  the **overnight** leg absorbs the dividend step; their product is the close-to-close **total**
  return. Both labels are carried through every table.
- **Window 2007-05-30 → 2026-06-30**, gated by BIL's inception — the cash leg has to be a real,
  buyable total return rather than an assumed short rate, because the rules sit in cash for a
  large share of the sample.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps between
  reruns. Fingerprint `f36d90ae4fdc`.
