# References & literature map — Study 937 (Tranches)

## The claim under test

- **Rebalance timing luck.** Corey Hoffstein, Nathan Faber & Steven Braun (Newfound
  Research, 2019), *Rebalance Timing Luck: The Difference Between Hired and Fired* and
  Hoffstein, Sober & Vezeris (2020), *Rebalance Timing Luck: The (Dumb) Luck of Smart
  Beta*, Journal of Index Investing. The same rules-based strategy rebalanced on a
  different day of the period traces a materially different equity curve — dispersion that
  is an artefact of an arbitrary implementation choice, not of skill. Their proposed fix is
  **portfolio tranching** (overlapping portfolios): run N sleeves rebalanced on N staggered
  dates so no single date decides the book. This study tests the *fix*, on the real tape:
  how fast does the cone shrink with N, and what does the fix cost?
- **The steelman for doing nothing.** Tranching multiplies operational complexity, and a
  sceptic can argue the cone is small enough to ignore, or that the extra sleeves cost more
  in tickets and tax than the dispersion is worth. Both are empirical questions; the cost
  side is answered here with a labelled ticket **ASSUMPTION** sweep, since broker tickets
  are not on the price tape.

## Where the overlapping-portfolio idea comes from

- **Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*, Journal of
  Finance.** The original momentum paper already forms **overlapping** portfolios — K
  simultaneous cohorts, one initiated each month — precisely so the result does not hinge
  on a single formation date. Tranching is that construction moved from the academic
  back-test into live portfolio management.
- **Blitz, van der Grient & van Vliet (2010), *Fundamental Indexation: Rebalancing
  Assumptions and Performance*, Journal of Index Investing.** Shows the reported performance
  of fundamental indices depends materially on the (arbitrary) annual reconstitution month,
  and that staggering it removes the dependence — the same phenomenon at annual frequency.
- **Faber (2007), *A Quantitative Approach to Tactical Asset Allocation*, Journal of Wealth
  Management (SSRN 962461).** The 200-day / 10-month rule used here as the sleeve. It is a
  vehicle, not the object of study: the point is that *any* periodically rebalanced rule
  carries a schedule, and the schedule carries luck.
- **Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, Journal of Financial
  Economics**; **Jegadeesh & Titman (2001)** for the 12-1 window used in the cross-check
  sleeve (twelve months of return, skipping the most recent one).

## Why the cone exists at all

- A monthly rule is a **low-frequency sampler** of a high-frequency price path. Two books
  that sample the same signal 21 days apart hold different positions through every
  transition, and those differences compound: the study measures a 95% terminal-wealth
  spread over 23 years from nothing but the sampling phase.
- The dispersion is therefore **path noise, not information** — which is why an
  exposure-matched *random-timing* rule on the same legs shows an even wider cone (sd
  0.103) than the trend rule does (0.066). Any statistic built on one schedule inherits it.
- **Inference consequence.** A reported Sharpe from a single rebalance date carries a
  standard error the back-test never prints. Bailey & López de Prado (2014), *The Deflated
  Sharpe Ratio*, Journal of Portfolio Management, makes the general point for multiple
  testing; timing luck is the version of it hiding inside a *single* strategy.

## Related desk studies (dedup)

- **[Study 836 — Rebalance Timing Luck](../../836-timing-luck/)** measures the phantom
  dispersion on a **synthetic** panel built so momentum has zero genuine edge, and proves
  the lucky offset is unforecastable. Study 937 is the sequel it points at: the same
  question asked of the **real SPY/IEF tape**, with the **fix** as the object of study —
  how fast the cone closes as N goes 1 → 4 → 12 → 21, what tranching costs in turnover and
  in broker tickets, and whether the real-tape lucky date persists (836 could only answer
  that in simulation).
- **[Study 936 — Tolerance Bands](../../936-rebalance-bands/)** asks *when* to rebalance
  (calendar against 5/25 drift bands) on a fixed-weight book. Study 937 holds the schedule
  frequency fixed and varies only its **phase**, and its book is a signal-driven sleeve, not
  a constant mix.
- **[Study 102 — Free Rebalance](../../102-free-rebalance/)** is the *economic* rebalancing
  premium (volatility harvesting); this is a *statistical* artefact of the calendar with no
  economics underneath.
- **[Study 604 — Month-End Rebalancing Flows](../../604-month-end-rebalancing-flows/)** is a
  genuine, predictable *flow* around month-end. It is the reason we report the +0.556
  cross-rule offset-ranking correlation honestly: part of this cone may be that flow rather
  than pure luck — but the payoff to chasing it is smaller than the free gain from
  tranching.
- **[Study 110 — Faber Timing](../../110-faber-timing/)** is the 200-day rule itself,
  evaluated as a strategy. Here the rule is only a carrier for the schedule question.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite …
  Covariance Matrix*, Econometrica — [`strategy.newey_west_t`](../tranching/strategy.py)
  and [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Return-difference (Sharpe comparison) t-stat.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance —
  [`strategy.sharpe_diff_tstat`](../tranching/strategy.py).
- **Circular block bootstrap**, resampled jointly across the 21 books so their
  cross-sectional dependence survives. Politis & Romano (1994), *The Stationary Bootstrap*,
  JASA — [`strategy.dispersion_bootstrap`](../tranching/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources

- **SPY** (risky sleeve), **IEF** (defensive sleeve), **BIL** (tradable cash cross-check) —
  daily **total-return** closes via `yfinance` (`auto_adjust=True`), from the shared desk
  cache `studies/_cache`.
- **^IRX** — the 13-week Treasury bill *discount yield*, accrued daily as the cash leg. It
  is a **PROXY** (a quote, not a fund) used so the excess-of-cash race can reach back to
  IEF's 2002 inception, five years before BIL existed; the BIL cross-check on 2007+ differs
  by 0.008 of a Sharpe point.
- **As-of 2026-06-30** — the partial current month is dropped so the sample never creeps.
  The 200-day warm-up puts the 21 books live from 2003-06-16.
