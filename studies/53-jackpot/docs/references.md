# References & literature map — Study 53 (Jackpot)

## The effect and its source

- **Bali, T., Cakici, N., & Whitelaw, R. (2011).** *Maxing Out: Stocks as Lotteries and the
  Cross-Section of Expected Returns.* Journal of Financial Economics 99(2) — the MAX (lottery) effect:
  high recent maximum daily returns predict low subsequent returns.
- **Kumar, A. (2009).** *Who Gambles in the Stock Market?* Journal of Finance — the behavioural demand
  for lottery-like stocks, concentrated in retail-heavy, small names.
- **Vendor entry** — the lottery/skewness family on
  [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading); backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## Why it inverts on tradable large caps

- **Hou, K., Xue, C., & Zhang, L. (2020).** *Replicating Anomalies* — lottery/idio-vol effects
  concentrate in micro-caps and weaken sharply with breakpoints and value-weighting.
- **Open-Alpha-Lab [Study 43 Free-Lunch](../../43-free-lunch/)** (betting against beta failed on large
  caps) and **[Study 54 Static](../../54-static/)** (idio-vol, the near-twin of MAX) — the same
  high-risk-won-on-large-caps regime.

## Data

- **Yahoo! Finance** — daily returns for current S&P 500 members with ≥15y history. Survivorship-biased,
  large-cap (the lottery effect's home is small/micro caps — the untradable end). The offline synthetic
  panel makes high-MAX (high-vol) stocks underperform (and a null) so the machinery is provable offline.

*MAX ≈ idiosyncratic volatility: see the near-twin [Study 54 Static](../../54-static/).*
