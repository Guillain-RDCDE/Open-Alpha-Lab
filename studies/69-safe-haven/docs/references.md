# References & literature map — Study 69 (Safe-Haven)

## The claims and the evidence

- **Erb, C., & Harvey, C. (2013).** *The Golden Dilemma.* Financial Analysts Journal — the definitive
  takedown: gold is a poor short/medium-horizon inflation hedge; the "real" gold price mean-reverts over
  very long horizons only.
- **Baur, D., & Lucey, B. (2010).** *Is Gold a Hedge or a Safe Haven?* Financial Review — distinguishes
  a *hedge* (uncorrelated on average) from a *safe haven* (uncorrelated/▲ in crises); gold is more the
  former than the latter.
- **Baur, D., & McDermott, T. (2010).** *Is Gold a Safe Haven? International Evidence.* Journal of
  Banking & Finance — gold's safe-haven role is regime- and region-dependent, not universal.

## Context

- **Open-Alpha-Lab** kin: [68 All-Weather](../../68-all-weather/) (gold as one sleeve of risk parity)
  and [70 Digital-Gold](../../70-digital-gold/) (bitcoin pitched as "digital gold" — the same
  diversifier/hedge questions).

## Data

- **Yahoo! Finance** — GLD (gold) and SPY (US equities) monthly total returns, 2005–2025. **FRED** — US
  CPI (the shared `macro_us` pull). Inflation hedge tested on YoY gold vs YoY CPI and a high/low-inflation
  split; crisis hedge on equity-crash months. The offline synthetic world drives gold off a smooth
  inflation cycle with a tunable loading (and a null where gold ignores inflation).

*A diversifier/hedge companion to [68 All-Weather](../../68-all-weather/) and [70
Digital-Gold](../../70-digital-gold/).*
