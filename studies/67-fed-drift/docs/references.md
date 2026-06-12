# References & literature map — Study 67 (Fed-Drift)

## The effect and its source

- **Lucca, D., & Moench, E. (2015).** *The Pre-FOMC Announcement Drift.* Journal of Finance 70(1) — the
  canonical paper: large, statistically robust equity returns in the 24 hours before scheduled FOMC
  statements, accounting for most of the equity premium since 1994.
- **Cieslak, A., Morse, A., & Vissing-Jørgensen (2019).** *Stock Returns over the FOMC Cycle.* Journal
  of Finance — the even-week / odd-week FOMC-cycle pattern, a related calendar structure.

## Post-publication decay

- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return Predictability?*
  Journal of Finance — anomalies decay ~58% post-publication; the pre-FOMC drift is a vivid case.
- **Open-Alpha-Lab** kin: the decay/timing cousins [56 Tide-Table](../../56-tide-table/) (CAPE) and [66
  Inverted](../../66-inverted/) (yield curve) — real signals, poor live triggers — and the calendar
  studies [55 Summer-Lull](../../55-summer-lull/) / [59 Downhill](../../59-downhill/).

## Data

- **Yahoo! Finance** — SPY daily total returns, 1993–2026. **Federal Reserve** — scheduled FOMC meeting
  calendars (statement-release days); the dates are embedded in
  [`fed_drift/data.py`](../fed_drift/data.py). Daily close-to-close is a noisy proxy for the intraday
  2pm-to-2pm window Lucca-Moench measured — our t-stats are conservative. The offline synthetic world
  injects a known pre-FOMC drift (and a null).

*A calendar/announcement-effect companion to [55 Summer-Lull](../../55-summer-lull/) and a
post-publication-decay case alongside the timing studies [56 Tide-Table](../../56-tide-table/) and [66
Inverted](../../66-inverted/).*
