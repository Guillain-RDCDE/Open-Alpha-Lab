# References & literature map — Study 73 (First-Light)

## The claim under test

- **Zarattini, C. & Aiolfi, M. (2023).** *Can Day Trading Really Be Profitable? Evidence
  from the U.S. Equity Market.* Working paper available at SSRN. The paper reports very
  large back-tested returns (up to several hundred percent per year) on ORB strategies
  applied to QQQ and especially the 3×-leveraged TQQQ, over 2010–2023. It popularised the
  academic framing of the ORB and gave the recipe a peer-reviewed citation. We steelman
  it as: *the first 5-minute opening range provides a directional signal sufficient to
  justify a breakout entry, measured against a random-direction baseline, net of realistic
  costs.* The spectacular headline returns rely on leverage (TQQQ) and either ignore or
  understate financing drag; our study isolates the signal question from the leverage
  question and tests each honestly.

## The underlying effect the ORB leans on

- **Intraday momentum and the opening gap.** Gao, P., Han, X., Li, Y., & Zhou, G. (2018).
  *Market Intraday Momentum.* Journal of Financial Economics 129(2), 394–414. Documents
  that the first half-hour of trading predicts the last half-hour; the mechanism is
  correlated order flow and institutional momentum. The ORB is a coarser version of this
  pattern: if the opening direction persists, a breakout in that direction profits.
- **Heston, S., Korajczyk, R., & Sadka, R. (2010).** *Intraday Patterns in the
  Cross-section of Stock Returns.* Journal of Finance 65(4), 1369–1407. Periodic intraday
  return continuation every ~30 minutes after open; provides a micro-structure rationale
  for session-start directional persistence.
- **Opening gap continuation vs. reversion.** The direction of the opening gap (vs. prior
  close) tends to continue intraday on high-momentum days (Bhattacharya et al. 2020;
  various practitioner works by Toby Crabel). This is the micro-signal the ORB targets:
  the opening range encodes the market's initial directional commitment for the session.

## Why the steelman is fragile

- **Low power from the 60-day window.** Yahoo Finance caps 5-minute intraday history at
  approximately 60 calendar days. At ~1 signal/day per ticker and n = 60 per instrument,
  the study has roughly 30% power to detect an effect at |*t*| = 2 at the typical
  per-trade volatility. The WEAK verdict reflects this structural limitation; see
  Zarattini & Aiolfi (2023) for a multi-year back-test that claims to find significance.
- **Leverage is not a signal amplifier.** The TQQQ version's implied gains in the academic
  paper stem from combining a modestly positive intraday signal with 3× leverage and a
  historically rising TQQQ NAV. Prospectively, daily leverage rebalancing costs (daily
  volatility drag) and current financing rates (~5–6% annualised in 2024–2026) subtract
  roughly 50 bps/day from any TQQQ strategy. Lo & MacKinlay (1990), *When Are
  Contrarian Profits Due to Stock Market Overreaction?* (Review of Financial Studies) and
  the leveraged-ETF decay literature (Avellaneda & Zhang 2010, *Path-Dependence of
  Leveraged ETF Returns,* SIAM Journal on Financial Mathematics) document the mechanism.
- **Selection and look-ahead bias concerns.** Faber, M. (2007), *A Quantitative Approach
  to Tactical Asset Allocation* (Journal of Wealth Management), and Harvey, C., Liu, Y.,
  & Zhu, H. (2016), *...and the Cross-Section of Expected Returns* (Review of Financial
  Studies) — the bar for claiming a signal is high given the number of ORB variants tested
  in the literature; our honest single-variant test is one data point.

## Method lineage

- **HAC / Newey-West t-stat.** Newey, W. & West, K. (1987). *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica 55(3), 703–708. Used in [`strategy.summarize`](../first_light/strategy.py).
- **Sharpe with bootstrap CI.** Lo, A. (2002). *The Statistics of Sharpe Ratios.*
  Financial Analysts Journal 58(4), 36–52. Politis, D. & Romano, J. (1994). *The
  Stationary Bootstrap.* JASA 89(428), 1303–1313. Implemented in
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Barrier backtest / no-overnight discipline.** Chan, E. (2009). *Quantitative Trading.*
  Wiley — the standard reference for implementing a stop-and-target engine without
  look-ahead; our implementation in [`strategy.run_trades`](../first_light/strategy.py)
  follows the same convention (enter at next bar's open, conservative straddling fill).
- **Random-direction control.** The benchmark is a coin flip on the same entry timestamps,
  which is the minimal fair baseline for any directional rule. Used throughout the desk's
  intraday studies; see also [Study 72 — Loaded-Dice](../../72-loaded-dice/).

## Data sources

- **Yahoo! Finance intraday bars** (via `yfinance`), 5-minute fidelity, SPY / QQQ / IWM /
  TQQQ, ~60 calendar days ending 2026-06-12. The power ceiling is structural; the
  offline reproducible core runs on `data.synthetic_5m` only.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) 5-minute scalp —
  the same 5-minute tape, the same random-direction baseline, the same infrastructure.
- **[Study 13 — Crimson-Hour](../../13-crimson-hour/)**: intraday time-of-day effects —
  the opening-hour return anomaly, same family.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: event-window directional signals —
  similar event-triggered entry discipline, different catalyst.
- **[Study 42 — Last-Call](../../42-last-call/)**: end-of-day anomaly — counterpoint,
  asking whether the *closing* period carries predictable drift.
- **[Study 70 — Digital-Gold](../../70-digital-gold/)**: leveraged crypto product —
  the same leverage-drag critique applied to a different asset class.
