# References & literature map — Study 46 (Bargain-Bin)

## The effect and its source

- **Fama, E., & French, K. (1992, 1993).** *The Cross-Section of Expected Stock Returns* / *Common Risk
  Factors…* — value (HML) as a pricing factor.
- **Lakonishok, J., Shleifer, A., & Vishny, R. (1994).** *Contrarian Investment, Extrapolation, and
  Risk.* Journal of Finance — the behavioural case for value.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Value (Book-to-Market) Factor"* (listed Sharpe `0.526`), with a QuantConnect implementation.
  Backlog triage: [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## The lost decade

- **Israel, R., Laursen, K., & Richardson, S. (2021).** *Is (Systematic) Value Investing Dead?* Journal
  of Portfolio Management — the definitive treatment of value's 2007–2020 drawdown.
- **Arnott, R., Harvey, C., Kalesnik, V., & Linnainmaa, J. (2021).** *Reports of Value's Death May Be
  Greatly Exaggerated.* Financial Analysts Journal — the counter-case (the drawdown was re-rating, not
  a dead premium); we stay agnostic and let the tape speak.
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1) — the decay frame.

## Data

- **Yahoo! Finance** — value/growth ETF pairs IVE/IVW (S&P 500), VTV/VUG (broad), RPV/RPG (pure style),
  monthly total return. ETF pairs keep the test tradable and survivorship-free. The offline synthetic
  world injects a regime-switching value premium (and a null) to exercise the machinery offline.

*Companion: [45 Vanishing-Act](../../45-vanishing-act/) — the size premium's disappearance; together
they ask whether the classic factors are alive. [44 Growth-Spurt](../../44-growth-spurt/) — the
asset-growth (investment) factor.*
